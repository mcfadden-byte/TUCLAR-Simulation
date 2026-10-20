"""Per-agent HTN planners. Each drone owns one planner and its own wall-clock budget.

Backends:
  RuleHTNPlanner  deterministic role-assignment rules, no LLM calls
  RulePlanner   Non-HTN planner

Each drone has one planner; details from planners are not shared with each other.
Planners are deterministic, and will be consistent between drones.
"""
import json
import re
import time
import numpy as np

from openai import APITimeoutError

from environment import TargetType, TargetLocation, Environment


class PlanningBudgetExceeded(Exception):
    pass


class PlanningFailed(Exception):
    pass


class _BudgetedPlanner:
    def __init__(self, time_budget_seconds: float):
        self.time_budget = time_budget_seconds
        self.started_at = None  # Time that the planner started; set on this drone's first plan() call

    # Check how much time is remaining; if no time remains, throw an exception
    def _remaining(self):
        now = time.perf_counter()
        if self.started_at is None:
            self.started_at = now
        remaining = self.time_budget - (now - self.started_at)
        if remaining <= 0:
            raise PlanningBudgetExceeded(f"over {self.time_budget:.1f}s budget")
        return remaining


class Area:

    # targets: a list of TargetLocations (from environment.py)
    def __init__(self, targets):
        
        if len(targets) == 1:
            self.singular = True
        else:
            self.singular = False

        self.targets = targets
        
        # A list of sub-areas within the area.
        self.areas = []


class RuleHTNPlanner(_BudgetedPlanner):
    def plan(self, agent_index, num_agents, environment):
        self._remaining()

        if all (target.visited for target in environment.get_target_locations()):
            return "RECALL", None

        total_area = Area(environment.get_target_locations())

        self.divide(total_area)

        # TODO: Assign tasks to drones

        # Keep track of how many agents must visit an area
            # A0,0: one
            # A0,1: two
            # A0: two


    

    
    # Planning on making a more advanced version of divide_half below.
    def divide(self, area):
        if area.singular:
            return

        # TODO: Decide how to subdivide areas, then subdivide areas and add into area.areas

        for subarea in area.areas:
            self.divide(subarea)

    # Splits the given area, with an equal number of targets in each half.
    # Tries to optimize such that the center of each half is as far apart as possible
    # Recurses until there is one target in the area.
    def divide_half(self, area):
        if area.singular:
            return

        pts = np.array([t.get_location() for t in area.targets], dtype=float)
        n = len(pts)
        k = n // 2  # sizes are k and n - k (equal when n is even)

        # Start from the principal axis plus a few random directions.
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        rng = np.random.default_rng(0)
        starts = [vt[0]] + list(rng.normal(size=(8, 3)))

        best_dist = -1.0
        best_split = None

        for u in starts:
            prev_key = None
            while True:
                # For a fixed direction u, the best split puts the k points
                # with the largest projection on u in one group.
                order = np.argsort(-(pts @ u), kind="stable")
                idx_a, idx_b = order[:k], order[k:]
                diff = pts[idx_a].mean(axis=0) - pts[idx_b].mean(axis=0)
                dist = np.linalg.norm(diff)

                if dist > best_dist:
                    best_dist = dist
                    best_split = (idx_a.copy(), idx_b.copy())

                key = tuple(sorted(idx_a.tolist()))
                if key == prev_key or dist == 0:
                    break
                prev_key = key
                u = diff  # re-aim at the current mean-to-mean direction

        idx_a, idx_b = best_split
        area.areas = [
            Area([area.targets[i] for i in idx_a]),
            Area([area.targets[i] for i in idx_b]),
        ]

        for subarea in area.areas:
            self.divide(subarea)

# ---------------------------------------------------------------------------
# Rule-based backend
# ---------------------------------------------------------------------------
"""
The following is a non-hierarchical task planner.

The assignments are deterministic based on index and target type, which is 
publicly available to all agents. 

Since all geometric details were obfuscated, the HTN planner has too few details
to work with, and the hierarchy is skin-deep. Details may be re-added in the future.
"""

class RulePlanner(_BudgetedPlanner):
    def plan(self, agent_index, num_agents, targets, scene=None):
        # Works through the targets and decides what a drone ought to do.
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


