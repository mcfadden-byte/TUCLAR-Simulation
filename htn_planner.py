"""Per-agent HTN planner for the mission.

Each drone owns one HTNPlanner with its own wall-clock budget. Planning uses only the
shared scene (the target list, this drone's index, and the team size). Every drone runs
the same deterministic decomposition on the same scene, so the team reaches a consistent
division of labor without messaging. No shared object coordinates the drones.

Assumes at least 3 drones (sequential targets need 3 distinct visitors).
"""
import time

from environment import TargetType


class PlanningBudgetExceeded(Exception):
    pass


class HTNPlanner:
    def __init__(self, time_budget_seconds: float):
        self.time_budget = time_budget_seconds
        self.started_at = None  # set on this drone's first plan() call

    def _check_budget(self):
        now = time.perf_counter()
        if self.started_at is None:
            self.started_at = now
        elapsed = now - self.started_at
        if elapsed > self.time_budget:
            raise PlanningBudgetExceeded(f"{elapsed:.1f}s > {self.time_budget:.1f}s")

    def plan(self, agent_index, num_agents, targets):
        """Decompose ACCOMPLISH_MISSION into a primitive action.

        Returns (kind, target_index) where kind is "FLY", "IDLE", or "RECALL".
        Raises PlanningBudgetExceeded once this drone's wall-clock budget has run out.
        """
        self._check_budget()
        if all(t.visited for t in targets):
            return "RECALL", None

        # Roles are assigned over pending targets in canonical (index) order, so every
        # drone derives the same table from the same scene.
        exclusive_seen = 0
        sequential_seen = 0
        for idx, target in enumerate(targets):
            if target.visited:
                continue
            roles, exclusive_seen, sequential_seen = self._method_for(
                target, num_agents, exclusive_seen, sequential_seen)
            if agent_index in roles and not self._already_counted(agent_index, target):
                return "FLY", idx
        return "IDLE", None

    @staticmethod
    def _method_for(target, num_agents, exclusive_seen, sequential_seen):
        """Method decomposition per target type. Returns (role set, updated counters)."""
        if target.target_type == TargetType.JOINT:
            # Converge: the first two drones go together
            return {0, 1}, exclusive_seen, sequential_seen
        if target.target_type == TargetType.EXCLUSIVE:
            # Divide by item: one drone per exclusive target, round-robin by role
            return {exclusive_seen % num_agents}, exclusive_seen + 1, sequential_seen
        # SEQUENTIAL: three distinct drones, taken in consecutive role order
        roles = {(sequential_seen + s) % num_agents for s in range(3)}
        return roles, exclusive_seen, sequential_seen + 1

    @staticmethod
    def _already_counted(agent_index, target):
        # A drone that already counted toward a sequential target moves on
        return target.target_type == TargetType.SEQUENTIAL and agent_index in target.visitors
