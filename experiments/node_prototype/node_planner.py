"""Node-level LLM reasoning for the hybrid planner of TUCLAR-Simulation (prototype).

hybrid_planner.py asks the LLM "what is each silent teammate doing?" and turns every answer into a claim.
Here the LLM is asked about blank nodes of the drone's own task tree instead:

  cover_T   serve(T) under mission: is a teammate already heading to target T?
            "covered" by teammates S sets T aside for a while, but only if the shared priority rule says
            this drone yields to S (the lowest indices keep a target); "not_covered" and "unknown" change nothing
  type_T    method choice of serve(T): how many drones does target T need?
            Exclusive / Joint / Sequential select the method, as a type heard in a message would

A node is asked only when its answer could change the drone's next action:
  cover_T   T is among the next targets on the drone's planned route and at least one teammate is silent
  type_T    T has an unknown type, a teammate holds it, and planning with "T needs 2 drones" changes the action
Answers are cached per node for TTL rounds; an expired node is asked again together with what happened since.
An LLM answer never makes a drone idle while a target it could serve is still open.

Everything that is not an LLM answer (messages, observations, the last-known memory rule, the planner) is the same
as the messages-only ablation (HYBRID_USE_LLM = False), so NODE_USE_LLM = False reproduces that baseline.
"""
import math
import time
from collections import Counter

from environment import REQUIRED_DRONES, TargetType
from htn_hierarchical import HierarchicalHTNPlanner, HTNSolver, State, _label, build_domain, build_tree, render_tree
from htn_planner import UNKNOWN_SLOT_GUESS, PlanningFailed
from hybrid_planner import HybridPlanner

NODE_USE_LLM = True
NODE_TTL = 10            # rounds a node answer stays in force
MAX_COVER_NODES = 2      # cover_T nodes per query: the first targets on the planned route
MAX_NODES = 4

KIND_OF = {"Exclusive": 1, "Joint": 2, "Sequential": 3}
NAME_OF = {1: "Exclusive", 2: "Joint", 3: "Sequential"}


NODE_SYSTEM_PROMPT = """
You are the reasoning module of one drone in a search team. Communication between drones is limited, so the drone's route planner is missing some facts. The planner lists them as questions about specific targets. Answer each question from the evidence you are given.

How the team works. Every drone runs exactly the same procedure:
- All drones start together at the base at round 0 and fly at the same speed.
- Each drone plans the cheapest route over the targets it believes still need drones, nearest first, and flies to the first target on that route. It replans whenever it reaches a target or learns something new.
- A drone tells teammates its current target in a message when the link allows, and sees a teammate's target only when it has line of sight.
- An Exclusive target is cleared as soon as 1 drone arrives. A Joint target is cleared only when 2 drones are there at the same time; a drone that arrives alone waits there for a partner. A Sequential target is cleared after 3 different drones have visited it; each drone visits once and moves on.
- A drone learns a target's type only by arriving there or from a message. Until then it plans as if the target needs 1 drone.
- When more drones head to a target than it needs, the lowest-numbered drones keep it and the others move on.

Questions:
- cover_T: Is any teammate heading to target T right now? Answer "covered" and list the teammates you believe are heading there, "not_covered", or "unknown".
- type_T: What type is target T? Answer "Exclusive", "Joint", "Sequential", or "unknown".

Infer an answer from timing and from what happens at targets, not only from reports: a fact no drone has reported can still follow from the evidence.

Every teammate answers the same questions with the same procedure, so a guess that is not backed by evidence would be made by all of them at once. Answer "unknown" whenever the evidence does not point to an answer; the planner then proceeds as if it had not asked.

Reply with a single JSON object: a short "reason" covering each question, then "beliefs" with one entry per question.
"""


def node_schema(nodes):
    cover = {"type": "object",
             "properties": {"answer": {"type": "string", "enum": ["covered", "not_covered", "unknown"]},
                            "teammates": {"type": "array", "items": {"type": "integer"}}},
             "required": ["answer", "teammates"], "additionalProperties": False}
    kind = {"type": "object",
            "properties": {"answer": {"type": "string", "enum": ["Exclusive", "Joint", "Sequential", "unknown"]}},
            "required": ["answer"], "additionalProperties": False}
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "maxLength": 600},
            "beliefs": {"type": "object",
                        "properties": {n: (cover if n.startswith("cover_") else kind) for n in nodes},
                        "required": list(nodes), "additionalProperties": False},
        },
        "required": ["reason", "beliefs"],
        "additionalProperties": False,
    }


class NodeHTNPlanner(HierarchicalHTNPlanner):
    """HierarchicalHTNPlanner that also reads two node beliefs from the scene:
    scene["believed_kind"]: target -> 1/2/3 for targets of unknown type, and scene["set_aside"]: targets to skip.
    Everything else is HierarchicalHTNPlanner.plan unchanged."""

    def plan(self, agent_index, num_agents, targets, scene):
        self._start_round()
        self.last_tree, self.last_plan = "", []
        if all(t.visited for t in targets):
            return "RECALL", None

        me = agent_index
        known = scene["type_known"]
        believed = scene.get("believed_kind") or {}
        set_aside = scene.get("set_aside") or set()
        kind = {i: (REQUIRED_DRONES[t.target_type] if known(t) else believed.get(i)) for i, t in enumerate(targets)}
        visited_by_me = {i for i, t in enumerate(targets) if me in t.visitors}

        predicted = {p: t for p, t in scene["teammate_targets"].items()
                     if isinstance(t, int) and t >= 0 and not targets[t].cleared}
        claims = Counter(predicted.values())

        open_targets = []
        for i, t in enumerate(targets):
            if t.cleared or i in set_aside:
                continue
            k = kind[i]
            if k == 3 and i in visited_by_me:
                continue
            need = k if k is not None else UNKNOWN_SLOT_GUESS
            if claims[i] >= need:
                continue
            open_targets.append(i)
        if not open_targets:
            return "IDLE", None

        pos = {"cur": tuple(scene["location"]), "home": self.home}
        pos.update({i: targets[i].get_location() for i in open_targets})
        speed = scene["velocity"]

        def dist(a, b):
            return 0.0 if a == b else math.dist(pos[a], pos[b]) / speed

        horizon = math.inf
        if self.horizon_rounds is not None:
            horizon = max(0.0, self.horizon_rounds - scene.get("round", 0))

        methods, operators, heuristic = build_domain(
            open_targets, kind, set(predicted.values()), visited_by_me, dist, self.hold_cost, horizon)
        solver = HTNSolver(methods, operators, heuristic, check_budget=self._remaining)
        root = ("mission", frozenset(open_targets))
        try:
            result = solver.solve(State("cur", 0.0), [root])
        finally:
            self.expanded = solver.expanded
        if result is None:
            raise PlanningFailed("no decomposition satisfies the constraints")
        _, steps = result
        self.last_tree = "\n".join(render_tree(build_tree(root, steps)))
        self.last_plan = [_label(s[1]) for s in steps if s[0] == "P"]
        first = next(s[1] for s in steps if s[0] == "P")
        return "FLY", first[1]


class NodeHybridPlanner(HybridPlanner):
    """HybridPlanner whose LLM fills blank nodes of the task tree (see module docstring)."""

    def __init__(self, agent_index, time_budget_seconds, config=None, planner=None):
        super().__init__(agent_index, time_budget_seconds, config, planner)
        if type(self.htn) is HierarchicalHTNPlanner:
            self.htn.__class__ = NodeHTNPlanner        # same object as the drone's agent.htn
        self.use_node_llm = NODE_USE_LLM
        self.believed_kind = {}     # target -> (kind, round answered)
        self.set_aside = {}         # target -> (round answered, teammates believed there)
        self.node_cache = {}        # node -> (round, answer, teammates)
        self.node_history = []      # (round, node, answer, teammates), most recent last
        self.cleared_round = {}     # target -> first round this drone saw it cleared
        self.streak = {}            # (teammate, target) -> first round of an unbroken claim (message/obs/memory)
        self.last_nodes = []        # (node, answer, teammates) of the latest query, for grading
        self.safety_overrides = 0

    # -- per-round bookkeeping (no LLM) --------------------------------------------------------
    def observe(self, scene, targets):
        super().observe(scene, targets)
        rnd = scene["round"]
        for i, t in enumerate(targets):
            if t.cleared and i not in self.cleared_round:
                self.cleared_round[i] = rnd
        now = {(j, b.target) for j, b in self.beliefs.items() if isinstance(b.target, int)}
        self.streak = {k: self.streak.get(k, rnd) for k in now}
        self.set_aside = {t: v for t, v in self.set_aside.items()
                          if rnd - v[0] < NODE_TTL and not targets[t].cleared}
        known = self._known_with_findings(scene, targets)
        self.believed_kind = {t: v for t, v in self.believed_kind.items()
                              if rnd - v[1] < 2 * NODE_TTL and not targets[t].cleared and not known(targets[t])}

    def believed_slots(self, target_index):
        v = self.believed_kind.get(target_index)
        return v[0] if v else None

    # -- planning ----------------------------------------------------------------------------------
    def _scene(self, scene, targets, llm_aside=True, extra_kind=None):
        ps = dict(scene)
        ps["teammate_targets"] = {j: b.target for j, b in self.beliefs.items()}
        ps["type_known"] = self._known_with_findings(scene, targets)
        ps["believed_kind"] = {t: v[0] for t, v in self.believed_kind.items()}
        if extra_kind:
            ps["believed_kind"].update(extra_kind)
        ps["set_aside"] = set(self.set_aside) if llm_aside else set()
        return ps

    def plan(self, agent_index, num_agents, targets, scene, ask_llm=None):
        rnd = scene["round"]
        if self._observed_round != rnd:
            self.observe(scene, targets)
        self.last_used = dict(self.beliefs)
        self.last_nodes = []
        t0 = time.perf_counter()
        try:
            act = self.htn.plan(agent_index, num_agents, targets, self._scene(scene, targets))
            route = [int(p[len("fly_to("):-1]) for p in self.htn.last_plan if p.startswith("fly_to(")]
            if self.use_node_llm and ask_llm is not None and not all(t.cleared for t in targets):
                nodes = self._relevant_nodes(agent_index, num_agents, targets, scene, act, route)
                if nodes:
                    self.queries += 1
                    reply = ask_llm(NODE_SYSTEM_PROMPT, self._node_prompt(agent_index, targets, scene, nodes, route),
                                    node_schema(nodes))
                    self._apply(reply, nodes, agent_index, targets, scene)
            act = self.htn.plan(agent_index, num_agents, targets, self._scene(scene, targets))
            if act[0] == "IDLE" and self.set_aside:
                alt = self.htn.plan(agent_index, num_agents, targets, self._scene(scene, targets, llm_aside=False))
                if alt[0] != "IDLE":
                    act = alt
                    self.safety_overrides += 1
            return act
        finally:
            self.last_plan_s = time.perf_counter() - t0

    def _fresh(self, node, rnd):
        c = self.node_cache.get(node)
        return c is not None and rnd - c[0] < NODE_TTL

    def _relevant_nodes(self, me, num_agents, targets, scene, act, route):
        rnd = scene["round"]
        nodes = []
        silent = self.silent()
        if act[0] == "FLY" and silent:
            for t in route[:MAX_COVER_NODES]:
                if not self._fresh(f"cover_{t}", rnd):
                    nodes.append(f"cover_{t}")
        known = self._known_with_findings(scene, targets)
        held = Counter(b.target for b in self.beliefs.values() if isinstance(b.target, int))
        for i, t in enumerate(targets):
            if len(nodes) >= MAX_NODES:
                break
            if t.cleared or known(t) or i in self.believed_kind or held[i] < 1 or self._fresh(f"type_{i}", rnd):
                continue
            try:
                alt = self.htn.plan(me, num_agents, targets, self._scene(scene, targets, extra_kind={i: 2}))
            except Exception:
                continue
            if alt != act:
                nodes.append(f"type_{i}")
        return nodes[:MAX_NODES]

    def _apply(self, reply, nodes, me, targets, scene):
        rnd = scene["round"]
        answers = reply.get("beliefs") if isinstance(reply, dict) else None
        if not isinstance(answers, dict):
            return
        known = self._known_with_findings(scene, targets)
        for node in nodes:
            a = answers.get(node)
            if not isinstance(a, dict):
                continue
            t = int(node.split("_")[1])
            ans = a.get("answer")
            if node.startswith("cover_"):
                mates = sorted({int(j) for j in a.get("teammates") or []
                                if isinstance(j, int) and j in self.active and j != me})
                if ans == "covered" and not mates:
                    ans = "unknown"
                self.node_cache[node] = (rnd, ans, mates)
                self.last_nodes.append((node, ans, mates))
                self.node_history.append((rnd, node, ans, mates))
                if ans == "covered":
                    k = REQUIRED_DRONES[targets[t].target_type] if known(targets[t]) else (
                        self.believed_slots(t) or UNKNOWN_SLOT_GUESS)
                    rivals = sorted(set(mates) | {me})
                    if me not in rivals[:k]:
                        self.set_aside[t] = (rnd, mates)
            else:
                self.node_cache[node] = (rnd, ans, [])
                self.last_nodes.append((node, ans, []))
                self.node_history.append((rnd, node, ans, []))
                if ans in KIND_OF:
                    self.believed_kind[t] = (KIND_OF[ans], rnd)
        self.node_history = self.node_history[-8:]

    # -- prompt ---------------------------------------------------------------------------------------
    def _node_prompt(self, me, targets, scene, nodes, route):
        rnd = scene["round"]
        loc = scene["location"]
        speed = scene["velocity"]
        known = self._known_with_findings(scene, targets)

        def rounds(a, b):
            return math.dist(a[:2], b[:2]) / speed

        lines = [f"You are drone {me}. Round {rnd}. Active teammates: {', '.join(map(str, sorted(self.active))) or 'none'}."]
        mine = scene.get("my_target")
        lines.append(f"You are heading to target {mine}." if isinstance(mine, int) and mine >= 0 else "You are idle.")
        if route:
            lines.append("Your planned route: " + " -> ".join(f"target {t}" for t in route) + ".")
        lines.append("Distances are in rounds of flight.")
        far = max(rounds((0, 0), t.get_location()) for t in targets)
        lines.append(f"Every target is at most {far:.0f} rounds of flight from the base, so a drone reaches any target "
                     f"well within {2 * far:.0f} rounds of heading to it.")

        lines.append("\nTEAMMATES:")
        for j in sorted(self.active):
            b = self.beliefs.get(j)
            if b is not None and b.source in ("msg", "obs"):
                where = "idle" if b.target is None else f"heading to target {b.target}"
                how = "message received this round" if b.source == "msg" else "seen this round"
                lines.append(f"  Agent {j}: {where} ({how})")
                continue
            lk = self.last_known.get(j)
            if lk is None:
                lines.append(f"  Agent {j}: silent; nothing heard or seen since round 0, when all drones were at the base")
            else:
                where = "idle" if lk.target is None else f"heading to target {lk.target}"
                how = "a message" if lk.source == "msg" else "an observation"
                lines.append(f"  Agent {j}: silent; last known {where}, from {how} {rnd - lk.round} rounds ago")

        lines.append("\nOPEN TARGETS:")
        for i, t in enumerate(targets):
            if t.cleared:
                continue
            if known(t):
                kind = t.target_type.value if me in t.visitors or i not in self.learned_types else \
                    f"{self.learned_types[i].value} (from a message)"
            elif i in self.believed_kind:
                kind = f"type unknown; your earlier estimate: {NAME_OF[self.believed_kind[i][0]]}"
            else:
                kind = "type unknown"
            tl = t.get_location()
            text = (f"  Target {i} ({kind}): {rounds(loc, tl):.0f} from you, {rounds((0, 0), tl):.0f} from the base")
            if t.target_type == TargetType.SEQUENTIAL and known(t) and me in t.visitors:
                text += ", you have already visited it"
            holders = [(j, s) for (j, tt), s in self.streak.items() if tt == i]
            for j, since in sorted(holders):
                src = self.beliefs[j].source if j in self.beliefs else "memory"
                how = {"msg": "message", "obs": "observation", "memory": "last known"}.get(src, src)
                text += f"; agent {j} heading there since round {since} ({rnd - since} rounds, {how})"
            if i in self.set_aside:
                r, mates = self.set_aside[i]
                text += f"; you set it aside at round {r} for agents {mates}"
            lines.append(text)

        cleared = sorted((r, i) for i, r in self.cleared_round.items())
        if cleared:
            lines.append("\nCLEARED TARGETS:")
            for r, i in cleared:
                t = targets[i]
                there = "you were there" if me in t.visitors else "you were not there"
                lines.append(f"  Target {i}: cleared by round {r}; {there}")

        if self.node_history:
            lines.append("\nYOUR EARLIER ANSWERS:")
            for r, node, ans, mates in self.node_history:
                t = int(node.split("_")[1])
                state = "now cleared" if targets[t].cleared else "still open"
                extra = f" (agents {mates})" if mates else ""
                lines.append(f"  Round {r}: {node} = {ans}{extra}; target {t} is {state}")

        lines.append("\nQUESTIONS: " + ", ".join(nodes))
        return "\n".join(lines)
