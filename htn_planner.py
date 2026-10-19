"""Per-agent HTN planners. Each drone owns one planner and its own wall-clock budget.

Backends:
  RuleHTNPlanner  deterministic role-assignment rules, no LLM calls
  LLMHTNPlanner   LLM decomposes the mission into subtasks, then primitive actions

Both plan from the shared scene only. Drones never read each other's plans.
"""
import json
import re
import time

from openai import APITimeoutError

from environment import TargetType


class PlanningBudgetExceeded(Exception):
    pass


class PlanningFailed(Exception):
    pass


class _BudgetedPlanner:
    def __init__(self, time_budget_seconds: float):
        self.time_budget = time_budget_seconds
        self.started_at = None  # set on this drone's first plan() call

    # Check how much time is remaining; if no time remains, throw an exception
    def _remaining(self):
        now = time.perf_counter()
        if self.started_at is None:
            self.started_at = now
        remaining = self.time_budget - (now - self.started_at)
        if remaining <= 0:
            raise PlanningBudgetExceeded(f"over {self.time_budget:.1f}s budget")
        return remaining


# ---------------------------------------------------------------------------
# Rule-based backend
# ---------------------------------------------------------------------------

"""
The following is a rule-based HTN planner that assigns tasks to agents.

The assignments are deterministic based on index and target type, which is 
publicly available to all agents. 

Since all geometric details were obfuscated, the HTN planner has too few details
to work with, and the hierarchy is skin-deep. Details may be re-added in the future.
"""

class RuleHTNPlanner(_BudgetedPlanner):
    def plan(self, agent_index, num_agents, targets, scene=None):
        # Works through the rules and decides what a drone ought to do.
        # Decides what EVERY drone ought to do, but only outputs for one drone
        # Outputs String & None or String & int.
        # "RECALL", None
        # "IDLE", None
        # "Fly", int

        # Check if too much time has passed
        self._remaining()

        # Check if all targets have been visited
        if all(t.visited for t in targets):
            return "RECALL", None

        exclusive_seen = 0
        sequential_seen = 0
        joint_seen = 0
        for idx, target in enumerate(targets):
            if target.visited:
                continue
            roles, exclusive_seen, sequential_seen, joint_seen = self._method_for(
                target, num_agents, exclusive_seen, sequential_seen, joint_seen)
            if agent_index in roles and not self._already_counted(agent_index, target):
                return "FLY", idx
        return "IDLE", None

    # Using this method, iterating over every target, an agent is assigned to every target.
    # This method is deterministic based on information available to all agents and it
    # doesn't matter if it's centralized or decentralized. However, future changes may 
    # make that distinction matter.
    @staticmethod
    def _method_for(target, num_agents, exclusive_seen, sequential_seen, joint_seen):
        if target.target_type == TargetType.JOINT:
            return {joint_seen % num_agents, (joint_seen + 1) % num_agents}, exclusive_seen, sequential_seen, joint_seen + 1
        if target.target_type == TargetType.EXCLUSIVE:
            return {exclusive_seen % num_agents}, exclusive_seen + 1, sequential_seen, joint_seen
        roles = {(sequential_seen + s) % num_agents for s in range(3)}
        return roles, exclusive_seen, sequential_seen + 1, joint_seen

    # Check if an agent has already visited a certain target. Only applicable for Sequential targets.
    @staticmethod
    def _already_counted(agent_index, target):
        return target.target_type == TargetType.SEQUENTIAL and agent_index in target.visitors


# ---------------------------------------------------------------------------
# LLM-driven backend
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_HTN = """
You are a Hierarchical Task Network (HTN) planner for ONE agent in a team of drones.
You decompose a mission into ordered tasks, and tasks into smaller subtasks, until every
leaf is a primitive action. Plan only for yourself. Teammates run the same planner on the
same scene, cannot see your plan, and cannot be messaged reliably.
If the scene lists numbered slots, Agent k is to take Slot k (slots count up across tasks).
Assume lower-numbered agents have already claimed their slots, even if shown as Unclaimed.

Primitive actions: "IDLE", or "FLY <n>" where n is the number of an INCOMPLETE task.
Reply with a single JSON object and nothing else.
"""

TOP_PROMPT = """
Decompose the mission into an ordered list of tasks for yourself.
Set "action" to a primitive action when you can perform the task directly, or null for a
compound task that needs further decomposition.
Format: {"tasks": [{"task": "<description>", "action": "FLY 2" or null}, ...]}
"""

EXPAND_PROMPT = """
Decompose this compound task into an ordered list of subtasks for yourself: {task}
Same format: {{"tasks": [{{"task": "<description>", "action": "FLY 2" or null}}, ...]}}
"""

SELECT_PROMPT = """
Below are candidate decompositions for your mission. Choose the one most likely to
complete the mission quickly without colliding with teammates who apply the same rules.
Format: {"choice": <candidate number>}

"""

_decoder = json.JSONDecoder(strict=False)


def _parse(text, key):
    """Last JSON object in text containing `key`, or None."""
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for start in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            obj, _ = _decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    return None


class LLMHTNPlanner(_BudgetedPlanner):
    def __init__(self, time_budget_seconds, client, model, max_depth=2, num_candidates=2,
                 max_replans=2, max_tokens=1024, candidate_temperature=0.7):
        super().__init__(time_budget_seconds)
        self.client = client
        self.model = model
        self.max_depth = max_depth
        self.num_candidates = num_candidates
        self.max_replans = max_replans
        self.max_tokens = max_tokens
        self.candidate_temperature = candidate_temperature
        self.queue = []  # cached primitive steps, e.g. ["FLY 3", "FLY 5"]

    def plan(self, agent_index, num_agents, targets, scene=None):
        """Returns (kind, target_index). Raises PlanningBudgetExceeded or PlanningFailed."""
        self._remaining()
        if all(t.visited for t in targets):
            self.queue = []
            return "RECALL", None

        self._prune(agent_index, targets)
        attempts = 0
        while not self.queue:
            if attempts > self.max_replans:
                raise PlanningFailed(f"no valid plan after {attempts} attempts")
            attempts += 1
            try:
                self.queue = self._decompose(agent_index, num_agents, targets, scene or "")
            except PlanningFailed as e:
                print(f"[HTN-LLM] drone {agent_index}: re-planning ({e})")

        step = self.queue[0]
        if step == "IDLE":
            self.queue.pop(0)  # an idle step lasts one round
            return "IDLE", None
        return "FLY", int(step.split()[1])

    # Drop leading steps that are no longer useful
    def _prune(self, agent_index, targets):
        while self.queue and self.queue[0] != "IDLE":
            target = targets[int(self.queue[0].split()[1])]
            counted = target.target_type == TargetType.SEQUENTIAL and agent_index in target.visitors
            if target.visited or counted:
                self.queue.pop(0)
            else:
                break

    def _decompose(self, agent_index, num_agents, targets, scene):
        pending = {i for i, t in enumerate(targets) if not t.visited}
        header = f"You are agent {agent_index} of {num_agents}.\n\n{scene}\n"

        temp = self.candidate_temperature if self.num_candidates > 1 else 0.0
        candidates = [self._ask(header + TOP_PROMPT, "tasks", temp)
                      for _ in range(self.num_candidates)]

        tasks = candidates[0]
        if len(candidates) > 1:
            listing = "\n".join(f"Candidate {i}: {json.dumps(c)}" for i, c in enumerate(candidates))
            try:
                tasks = candidates[int(self._ask(header + SELECT_PROMPT + listing, "choice", 0.0))]
            except (ValueError, TypeError, IndexError):
                pass  # keep the first candidate

        steps = self._expand(tasks, 1, header, pending)
        if not steps:
            raise PlanningFailed("empty plan")
        return steps

    def _expand(self, tasks, depth, header, pending):
        if not isinstance(tasks, list):
            raise PlanningFailed("tasks is not a list")
        steps = []
        for task in tasks:
            if not isinstance(task, dict):
                raise PlanningFailed("malformed task")
            action = task.get("action")
            if action:
                action = str(action).strip()
                if not self._legal(action, pending):
                    raise PlanningFailed(f"illegal action {action!r}")
                steps.append(action)
            elif depth < self.max_depth:
                sub = self._ask(header + EXPAND_PROMPT.format(task=task.get("task", "")),
                                "tasks", 0.0)
                steps += self._expand(sub, depth + 1, header, pending)
            else:
                raise PlanningFailed("compound task at max depth")
        return steps

    @staticmethod
    def _legal(action, pending):
        if action == "IDLE":
            return True
        m = re.fullmatch(r"FLY (\d+)", action)
        return bool(m) and int(m.group(1)) in pending

    def _ask(self, prompt, key, temperature):
        remaining = self._remaining()  # raises if the budget is already spent
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT_HTN},
                          {"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=temperature,
                timeout=remaining,  # a slow call cannot outlast the budget
                extra_body={"chat_template_kwargs": {"enable_thinking": False, "thinking": False}},
            )
        except APITimeoutError as e:
            raise PlanningBudgetExceeded("LLM call hit the budget") from e
        data = _parse(response.choices[0].message.content, key)
        if data is None:
            raise PlanningFailed(f"could not parse {key!r}")
        return data[key]


def make_htn_planner(backend, time_budget, client, model, **llm_options):
    if backend == "rule":
        return RuleHTNPlanner(time_budget)
    if backend == "llm":
        return LLMHTNPlanner(time_budget, client, model, **llm_options)
    raise ValueError(f"Unknown HTN_BACKEND: {backend!r}")
