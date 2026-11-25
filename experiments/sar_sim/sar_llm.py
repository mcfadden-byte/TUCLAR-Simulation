"""LLM-backed decision makers for the search-and-rescue case study (sar_sim.py).

NodeBrain (our method): the planner of sar_sim decides every action; the LLM only fills blank nodes of the task tree.
  likelihood   one node per area, asked once at launch: how likely the area holds the person (from the brief).
               Messages can never fill it, so it matters at every level of communication.
  cover_A      asked at a decision when some teammate is silent, for the first areas on the planned route:
               has a teammate already searched A, or is one heading there? An answer naming teammates sets A aside
               for COVER_TTL rounds, but only when the shared priority rule says this drone yields.
               "open" and "unknown" change nothing, and LLM answers never leave the drone without an area.
LLMOnlyBrain (baseline): the LLM picks every area itself from the same information.
Set SAR_MOCK_LLM=1 to replace the LLM with a fixed stand-in (for testing the plumbing without a GPU).
"""
import json
import re
import math
import os
import time

import sar_sim as S

COVER_TTL = 10
COVER_NODES = 3


# ---------------------------------------------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------------------------------------------
class LLMClient:
    def __init__(self):
        from openai import OpenAI
        self.cli = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key="EMPTY", timeout=60, max_retries=1)
        self.model = os.environ.get("MODEL", "gemma-4-26b-a4b-it")

    def ask(self, system, user, schema):
        """Returns (parsed JSON or None, seconds)."""
        t0 = time.perf_counter()
        for attempt in range(2):          # one retry: a reply now and then runs away into padding before closing
            try:
                r = self.cli.chat.completions.create(
                    model=self.model, temperature=0.2, max_tokens=1000,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    # A regex instead of the JSON schema: the schema lets gemma pad the JSON with newlines up to
                    # max_tokens (vLLM 0.24's xgrammar ignores a per-request disable_any_whitespace).
                    extra_body={"chat_template_kwargs": {"enable_thinking": False, "thinking": False},
                                "structured_outputs": {"regex": schema_regex(schema)}})
                return json.loads(r.choices[0].message.content), time.perf_counter() - t0
            except Exception as e:        # timeouts, bad JSON, server errors: the caller proceeds without an answer
                print(f"LLM call failed (attempt {attempt + 1}): {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None, time.perf_counter() - t0


def schema_regex(s):
    """Regex for compact JSON matching the subset of JSON schema used here (objects with ordered required keys,
    strings, integers, enums, integer arrays)."""
    if "enum" in s:
        return "(" + "|".join(str(v) if isinstance(v, int) else '"' + re.escape(v) + '"' for v in s["enum"]) + ")"
    t = s["type"]
    if t == "object":
        return r"\{" + ",".join(f'"{re.escape(k)}":' + schema_regex(v) for k, v in s["properties"].items()) + r"\}"
    if t == "string":
        return r'"[^"\\\n]*"'
    if t == "integer":
        return r"-?[0-9]{1,3}"
    if t == "array":
        item = schema_regex(s["items"])
        return r"\[(" + item + "(," + item + r"){0,%d})?\]" % (s.get("maxItems", 4) - 1)
    raise ValueError(f"unsupported schema {s}")


class MockLLM:
    """Stand-in for tests: uniform likelihoods, "unknown" for every cover node, nearest area for LLM-only."""

    def ask(self, system, user, schema):
        props = schema["properties"]
        if "likelihood" in props:
            keys = props["likelihood"]["required"]
            return {"reason": "mock", "likelihood": {k: 10 for k in keys}}, 0.0
        if "message" in props:
            return {"reason": "mock", "area": props["area"]["enum"][0], "message": "mock message"}, 0.0
        if "beliefs" in props:
            keys = props["beliefs"]["required"]
            return {"reason": "mock", "beliefs": {k: {"answer": "unknown", "teammates": []} for k in keys}}, 0.0
        return {"reason": "mock", "area": props["area"]["enum"][0]}, 0.0


_CLIENT = None


def client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = MockLLM() if os.environ.get("SAR_MOCK_LLM") == "1" else LLMClient()
    return _CLIENT


# ---------------------------------------------------------------------------------------------------------------
# Shared prompt pieces
# ---------------------------------------------------------------------------------------------------------------
TEAM_RULES = """How the team works. Every drone runs exactly the same procedure:
- All 5 drones start together at the base at round 0 with the same mission brief and the same estimate of where the person is likely to be.
- Each drone picks the area that minimises the expected time to find the person given what it knows: likely areas first, nearby areas first. One drone is enough for an area; a second drone searching the same area only wastes time. The person is found when a drone finishes sweeping the person's area.
- When the link between two drones is up, they exchange their current area and the areas they know to be searched. Links can stay down for long periods."""


def rounds_between(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1]) / S.SPEED


def state_lines(sim, d, prior=None):
    """What drone d knows, written for a prompt, including each area's total trip (flight there, sweep, flight home)."""
    sc, rnd = sim.sc, sim.round
    me = (d.x, d.y)
    lines = [f"You are drone {d.idx}. Round {rnd}. A drone flies {S.SPEED:.0f} m per round; distances are in rounds "
             f"of flight. Sweeping an area takes the number of rounds shown.",
             f"Battery: {d.battery} rounds left (a full charge lasts {S.BATTERY}). Flying, sweeping and hovering each "
             f"use one round of battery per round; recharging at the base takes {S.RECHARGE} rounds. A drone whose "
             f"battery runs out away from the base crashes."]
    if d.status == "fly":
        lines.append(f"You are flying to area {d.target}.")
    elif d.status == "sweep":
        lines.append(f"You are sweeping area {d.target}.")
    else:
        lines.append(f"You are {S.flight(me, (0, 0))} rounds from the base and have no area right now.")

    lines.append("\nTEAMMATES:")
    for j in range(S.N_DRONES):
        if j == d.idx:
            continue
        h = d.heard.get(j)
        claim = d.mate_claim.get(j)
        if h is None:
            lines.append(f"  Agent {j}: silent; nothing heard since round 0, when all drones were at the base")
            continue
        when = "this round" if rnd - h <= 1 else f"{rnd - h} rounds ago (silent since)"
        if j in d.mate_text:
            lines.append(f'  Agent {j}: last message {when}: "{d.mate_text[j][0]}"')
            continue
        a, st, r = claim
        what = ("returning to the base to recharge" if st in ("return", "charge") else "idle") if a is None else (
            f"sweeping area {a}" if st == "sweep" else f"heading to area {a}")
        lines.append(f"  Agent {j}: last message {when}: {what}")

    lines.append("\nAREAS NOT KNOWN TO BE SEARCHED:")
    claims = {}
    for j, (a, st, r) in d.mate_claim.items():
        if a is not None:
            claims.setdefault(a, []).append((j, st, rnd - r))
    tot = sum(prior) if prior else 0
    for ar in sc.areas:
        if ar.idx in d.known_done:
            continue
        there, back = S.flight(me, (ar.x, ar.y)), S.flight((ar.x, ar.y), (0, 0))
        text = f"  Area {ar.idx} ({ar.feature}): {there} from you, {back} from the base"
        text += f", {S.dist(ar, sc.areas[sc.lkp]):.0f} m from the last known point; sweep {ar.sweep} rounds"
        text += f"; flying there, sweeping it and flying back to the base takes {there + ar.sweep + back} rounds"
        if prior:
            text += f"; your likelihood estimate {100 * prior[ar.idx] / tot:.0f}%"
        for j, st, age in claims.get(ar.idx, []):
            text += f"; agent {j} {'sweeping' if st == 'sweep' else 'heading'} there (heard {age} rounds ago)"
        lines.append(text)
    done = sorted(d.known_done)
    if done:
        mine = sorted(d.own_done)
        others = [a for a in done if a not in d.own_done]
        lines.append(f"\nSEARCHED (known to you): by you {mine or 'none'}; reported by teammates {others or 'none'}")
    return lines


# ---------------------------------------------------------------------------------------------------------------
# Our method: node-level reasoning
# ---------------------------------------------------------------------------------------------------------------
ENV_SYSTEM = """You support a drone team that is searching for a missing person. Before the search starts, estimate how likely each search area is to hold the person. Use the mission brief, the terrain of each area and its distance from where the person was last seen, together with what you know about how lost people behave. Give every area a whole-number percentage; the percentages should add up to about 100.

Reply with a single JSON object: a short "reason", then "likelihood" with one entry per area."""

COVER_SYSTEM = """You are the reasoning module of one drone in a search-and-rescue team of 5 drones. Communication between drones is limited, so the drone's planner is missing some facts. The planner lists them as questions about specific search areas. Answer each question from the evidence you are given.

""" + TEAM_RULES + """

Question cover_A: Has a teammate already searched area A, or is a teammate heading to it or sweeping it now? Answer "already_searched" or "teammate_heading_there" and list those teammates; answer "open" if no teammate will search it soon; or answer "unknown".

Infer answers from timing and from the shared procedure, not only from reports: a fact no drone has reported can still follow from the evidence. Every teammate answers the same questions in the same way, so a guess that is not backed by evidence would be made by all of them at once. Answer "unknown" whenever the evidence does not point to an answer; the planner then proceeds as if it had not asked.

Reply with a single JSON object: a short "reason" covering each question, then "beliefs" with one entry per question."""


def env_prompt(sc):
    lines = [f"MISSION BRIEF: {sc.brief}", "", "SEARCH AREAS:"]
    lkp = sc.areas[sc.lkp]
    for ar in sc.areas:
        lines.append(f"  Area {ar.idx}: {ar.feature}; {S.dist(ar, lkp):.0f} m from the last known point "
                     f"(area {sc.lkp}); {math.hypot(ar.x, ar.y):.0f} m from the base")
    lines.append("\nEstimate the probability that the person is in each area.")
    return "\n".join(lines)


def env_schema(n):
    keys = [str(a) for a in range(n)]
    return {"type": "object",
            "properties": {"reason": {"type": "string", "maxLength": 600},
                           "likelihood": {"type": "object", "properties": {k: {"type": "integer"} for k in keys},
                                          "required": keys, "additionalProperties": False}},
            "required": ["reason", "likelihood"], "additionalProperties": False}


def cover_schema(nodes):
    one = {"type": "object",
           "properties": {"answer": {"type": "string",
                                     "enum": ["already_searched", "teammate_heading_there", "open", "unknown"]},
                          "teammates": {"type": "array", "items": {"type": "integer"}}},
           "required": ["answer", "teammates"], "additionalProperties": False}
    return {"type": "object",
            "properties": {"reason": {"type": "string", "maxLength": 600},
                           "beliefs": {"type": "object", "properties": {n: one for n in nodes},
                                       "required": list(nodes), "additionalProperties": False}},
            "required": ["reason", "beliefs"], "additionalProperties": False}


class NodeBrain:
    def __init__(self, i, sim):
        self.i, self.sim = i, sim
        self.prior = [1.0] * S.N_AREAS
        self.env_asked = False
        self.set_aside = {}      # area -> (round, answer, teammates)
        self.cache = {}          # node -> round answered
        self.history = []        # (round, node, answer, teammates)
        self.queries = 0
        self.llm_seconds = 0.0
        self.failures = 0

    def _ask(self, system, user, schema):
        reply, sec = client().ask(system, user, schema)
        self.queries += 1
        self.llm_seconds += sec
        if reply is None:
            self.failures += 1
        return reply

    def decide(self, d):
        sim, sc, rnd = self.sim, self.sim.sc, self.sim.round
        if not self.env_asked:
            self.env_asked = True
            reply = self._ask(ENV_SYSTEM, env_prompt(sc), env_schema(S.N_AREAS))
            lik = reply.get("likelihood") if isinstance(reply, dict) else None
            if isinstance(lik, dict):
                vals = [max(float(lik.get(str(a), 0) or 0), 0.0) for a in range(S.N_AREAS)]
                if sum(vals) > 0:
                    self.prior = [max(v, 0.5) for v in vals]      # never rule an area out completely
                sim.log.append(dict(kind="env", seed=sc.seed, drone=d.idx, prior=vals, person=sc.person,
                                    true_p=[round(p, 4) for p in sc.true_p], reason=reply.get("reason", "")))

        self.set_aside = {a: v for a, v in self.set_aside.items()
                          if rnd - v[0] < COVER_TTL and a not in d.known_done}
        area, route = S.choose_area(d, sc, self.prior, self.set_aside)
        if not route:
            return area
        silent = [j for j in range(S.N_DRONES) if j != d.idx and (d.heard.get(j) is None or rnd - d.heard[j] > 1)]
        if not silent:
            return area
        nodes = [f"cover_{a}" for a in route[:COVER_NODES]
                 if rnd - self.cache.get(f"cover_{a}", -10 ** 9) >= COVER_TTL]
        if not nodes:
            return area
        user = "\n".join(state_lines(sim, d, self.prior) +
                         ([f"Your planned route: " + " -> ".join(f"area {a}" for a in route) + "."] if route else []) +
                         self._history_lines(d) + ["\nQUESTIONS: " + ", ".join(nodes)])
        reply = self._ask(COVER_SYSTEM, user, cover_schema(nodes))
        answers = reply.get("beliefs") if isinstance(reply, dict) else None
        if isinstance(answers, dict):
            for node in nodes:
                a = int(node.split("_")[1])
                ans_obj = answers.get(node) or {}
                ans = ans_obj.get("answer", "unknown")
                mates = sorted({int(j) for j in ans_obj.get("teammates") or []
                                if isinstance(j, int) and 0 <= j < S.N_DRONES and j != d.idx})
                if ans in ("already_searched", "teammate_heading_there") and not mates:
                    ans = "unknown"
                self.cache[node] = rnd
                self.history = (self.history + [(rnd, node, ans, mates)])[-8:]
                if ans == "already_searched" or (ans == "teammate_heading_there" and min(mates) < d.idx):
                    self.set_aside[a] = (rnd, ans, mates)
                heading = sorted(o.idx for o in sim.drones if o.idx != d.idx and o.target == a
                                 and o.status in ("fly", "sweep"))
                sim.log.append(dict(kind="cover", seed=sc.seed, round=rnd, drone=d.idx, area=a, answer=ans,
                                    mates=mates, truth_searched=a in sim.searched, truth_heading=heading,
                                    silent=silent))
            area, route = S.choose_area(d, sc, self.prior, self.set_aside)
        return area

    def _history_lines(self, d):
        if not self.history:
            return []
        lines = ["\nYOUR EARLIER ANSWERS:"]
        for r, node, ans, mates in self.history:
            a = int(node.split("_")[1])
            now = "now known to be searched" if a in d.known_done else "still not known to be searched"
            lines.append(f"  Round {r}: {node} = {ans}{f' (agents {mates})' if mates else ''}; area {a} is {now}")
        return lines


# ---------------------------------------------------------------------------------------------------------------
# Baseline: the LLM chooses every area
# ---------------------------------------------------------------------------------------------------------------
LLM_ONLY_SYSTEM = """You control one drone in a search-and-rescue team of 5 drones that is looking for a missing person. Choose the next area for your drone to search, so that the team finds the person as soon as possible.

Every teammate is a drone like you, controlled the same way and from the same information.
- All 5 drones start together at the base at round 0 with the same mission brief.
- One drone is enough for an area; a second drone searching the same area only wastes time. The person is found when a drone finishes sweeping the person's area.
- When the link between two drones is up, they exchange their current area and the areas they know to be searched. Links can stay down for long periods.
- Your battery must always cover the flight back to the base. A drone whose battery runs out away from the base crashes. Choose -1 to fly back to the base and recharge.

Use the mission brief, the terrain and position of each area, and what you know about how lost people behave and about your teammates.

Reply with a single JSON object: a short "reason", then "area" with the number of the area to search next, or -1 to fly back to the base and recharge."""

# LLM-only as in LLM agents that coordinate through natural-language dialogue: each drone writes its own message.
LLM_ONLY_NL_SYSTEM = LLM_ONLY_SYSTEM.replace(
    "- When the link between two drones is up, they exchange their current area and the areas they know to be searched.",
    "- When the link between two drones is up, each receives the other's latest message.").replace(
    'or -1 to fly back to the base and recharge."""'[:-3],
    'or -1 to fly back to the base and recharge, then "message" with a brief message to your teammates; each teammate '
    'receives it whenever its link to you is up.')
assert LLM_ONLY_NL_SYSTEM.count("latest message") == 1 and LLM_ONLY_NL_SYSTEM.count('"message"') == 1


class LLMOnlyBrain:
    uses_priority_rule = False          # no coordination rule is imposed on the LLM; it sees the same messages

    def __init__(self, i, sim, nl_messages=False):
        self.i, self.sim = i, sim
        self.nl_messages = nl_messages      # teammates receive this drone's own message instead of the task-tree state
        self.prior = [1.0] * S.N_AREAS          # for the result record only
        self.queries = 0
        self.llm_seconds = 0.0
        self.failures = 0

    def decide(self, d):
        sim, sc = self.sim, self.sim.sc
        allowed = [a.idx for a in sc.areas if a.idx not in d.known_done]
        if not allowed:
            return None if S.at_base(d) else S.RETURN
        options = allowed + [S.RETURN]
        schema = {"type": "object",
                  "properties": {"reason": {"type": "string", "maxLength": 600},
                                 "area": {"type": "integer", "enum": options}},
                  "required": ["reason", "area"], "additionalProperties": False}
        if self.nl_messages:
            schema["properties"]["message"] = {"type": "string"}
            schema["required"].append("message")
        user = "\n".join([f"MISSION BRIEF: {sc.brief}", ""] + state_lines(sim, d) +
                         [f"\nChoose one of these areas: {allowed}, or -1 to fly back to the base and recharge."])
        reply, sec = client().ask(LLM_ONLY_NL_SYSTEM if self.nl_messages else LLM_ONLY_SYSTEM, user, schema)
        self.queries += 1
        self.llm_seconds += sec
        a = reply.get("area") if isinstance(reply, dict) else None
        if a not in options:
            self.failures += 1
            a = S.RETURN                                     # no usable answer: fly home, the safe default
        elif self.nl_messages and isinstance(reply.get("message"), str):
            d.message = reply["message"].strip()
        sim.log.append(dict(kind="llm_action", seed=sc.seed, round=sim.round, drone=d.idx, area=a,
                            person=sc.person, **({"message": d.message} if self.nl_messages else {})))
        return a
