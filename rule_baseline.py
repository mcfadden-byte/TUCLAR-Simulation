"""Decentralized rule-based baseline (no LLM, no central coordinator).

Each drone owns its own policy instance and decides from an Observation containing only what the
LLM prompt shows: teammates' last orders, received messages, target status, and (optionally)
coordinates. Decisions in a round are made simultaneously from a snapshot taken before anyone
acts, so a drone never sees a teammate's current-round choice. Movement and visit mechanics reuse
the simulator's own methods, as in optimal.py.
"""
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import utils_geometry as geo
from agent import Agent, AgentHandler, SHOW_OWN_LOCATION, SHOW_TARGET_COORDS
from environment import Environment, TargetLocation, TargetType

HOME = [0, 0, 0]
HOME_TOLERANCE = 1
MAX_ROUNDS = 400


# ---------------------------------------------------------------------------
# What a drone is allowed to know
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TargetView:
    index: int
    kind: TargetType
    cleared: bool
    required_drones: int
    visited_by_me: bool
    claimants: tuple                      # agent indices (incl. me) whose last order was this target
    location: Optional[tuple] = None      # (north, east, alt), only if coordinates are shown
    distance: Optional[float] = None      # from me, only if coordinates are shown


@dataclass(frozen=True)
class TeammateView:
    index: int
    last_target: object                   # int target index, None (idle), or -1 (unknown yet)
    message: Optional[str]                # None if out of range or comms disabled


@dataclass(frozen=True)
class Observation:
    agent_index: int
    num_agents: int
    current_target: object                # my own last order: int, None, or -1
    own_location: Optional[tuple]         # only if SHOW_OWN_LOCATION
    teammates: tuple                      # active (not recalled) teammates only
    targets: tuple
    all_cleared: bool

    def available_targets(self):
        """Targets worth flying to: not cleared, and not a Sequential I already counted toward."""
        return [t for t in self.targets
                if not t.cleared and not (t.kind == TargetType.SEQUENTIAL and t.visited_by_me)]

    def others_on(self, target: TargetView):
        """Teammates whose last order was this target."""
        return tuple(i for i in target.claimants if i != self.agent_index)


@dataclass
class Decision:
    action: str                           # "IDLE", "RECALL", or "FLY <target index>"
    message: str = ""                     # broadcast to teammates (subject to comms quality/range)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------
class DronePolicy:
    """One instance per drone. Keep any memory on self; instances never share state."""

    def __init__(self, agent_index: int, num_agents: int, rng: random.Random):
        self.agent_index = agent_index
        self.num_agents = num_agents
        self.rng = rng
        self.memory = {}

    def decide(self, obs: Observation) -> Decision:
        raise NotImplementedError


class RulePolicy(DronePolicy):

    def decide(self, obs: Observation) -> Decision:
        # ------------------------------------------------------------------
        # IMPLEMENT YOUR RULES HERE.
        # Everything you may use is in `obs` (and self.memory / self.rng):
        #   obs.agent_index, obs.num_agents, obs.current_target
        #   obs.targets[i]: kind, cleared, required_drones, visited_by_me, claimants,
        #                   location/distance (only if coordinates are shown)
        #   obs.teammates[k]: index, last_target, message
        #   obs.available_targets(), obs.others_on(target), obs.all_cleared
        # ------------------------------------------------------------------

        # Stand-in so the framework runs end to end; replace everything below.
        if obs.all_cleared:
            return Decision("RECALL")        # required for the mission to finish
        available = obs.available_targets()
        if not available:
            return Decision("IDLE")          # only Sequentials I already visited remain
        return Decision(f"FLY {available[0].index}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
@dataclass
class RuleResult:
    trials: int
    success_rate: float
    avg_distance: Optional[float]         # over successful trials
    avg_rounds: Optional[float]           # over successful trials
    invalid_actions: int                  # total across all trials


def _observe(me, agents, targets, comms_quality, comms_distance, show_coords, show_own_location):
    teammates = []
    for other in agents:
        if other is me or other.inaction:
            continue
        message = None
        if comms_quality > 0.0:
            in_range = (comms_distance == -1 or
                        geo.geometric_mean_3d(*other.location, *me.location) <= comms_distance)
            message = other.displayed_message if in_range else None
        teammates.append(TeammateView(other.agent_index, other.last_target, message))

    views = []
    for j, t in enumerate(targets):
        claimants = tuple(a.agent_index for a in agents if a.last_target == j and not a.inaction)
        loc = dist = None
        if show_coords:
            loc = t.get_location()
            dist = geo.point_distance(me.location, loc)
        views.append(TargetView(j, t.target_type, t.cleared, t.required_drones,
                                me.agent_index in t.visitors, claimants, loc, dist))

    return Observation(
        agent_index=me.agent_index,
        num_agents=len(agents),
        current_target=me.last_target,
        own_location=tuple(me.location) if show_own_location else None,
        teammates=tuple(teammates),
        targets=tuple(views),
        all_cleared=all(t.cleared for t in targets),
    )


def _apply(agent, decision, targets, comms_quality, rng):
    """Apply a decision; returns False if the action was invalid (the drone then idles)."""
    action = decision.action.strip()
    valid = True
    if action == "IDLE":
        agent.target_index = None
    elif action == "RECALL":
        agent.target_index = "Home"
        agent.inaction = True
    elif action.startswith("FLY "):
        try:
            idx = int(action[4:])
            t = targets[idx] if 0 <= idx < len(targets) else None
        except ValueError:
            t = None
        if (t is None or t.cleared or
                (t.target_type == TargetType.SEQUENTIAL and agent.agent_index in t.visitors)):
            valid = False
        else:
            agent.target_index = idx
    else:
        valid = False
    if not valid:
        agent.target_index = None

    agent.displayed_message = (decision.message if rng.random() < comms_quality
                               else "Communication Failure.")
    return valid


def run_once(spec, num_agents, velocity, start, policy_factory,
             comms_quality, comms_distance, show_coords, show_own_location, seed):
    rng = random.Random(seed)
    targets = [TargetLocation(n, e, a, kind) for n, e, a, kind in spec]
    env = SimpleNamespace(target_locations=targets)
    handler = SimpleNamespace(drone_velocity=velocity, environment=env)
    agents = [Agent("", None, "", list(start), i) for i in range(num_agents)]
    policies = [policy_factory(i, num_agents, random.Random(rng.getrandbits(32)))
                for i in range(num_agents)]
    invalid = 0

    for rounds in range(1, MAX_ROUNDS + 1):
        # Snapshot first: every drone decides from the same pre-round state
        observations = {a.agent_index: _observe(a, agents, targets, comms_quality, comms_distance,
                                                show_coords, show_own_location)
                        for a in agents if not a.inaction}
        decisions = {i: policies[i].decide(obs) for i, obs in observations.items()}

        for i, decision in decisions.items():
            if not _apply(agents[i], decision, targets, comms_quality, rng):
                invalid += 1
        for a in agents:
            if not a.inaction:
                a.last_target = a.target_index

        for a in agents:
            AgentHandler.update_velocity(handler, a)
            a.step()
        Environment.check_visits(env, agents)

        if all(t.cleared for t in targets) and all(
                geo.point_distance(a.location, HOME) < HOME_TOLERANCE for a in agents):
            return True, sum(a.distance_traveled for a in agents), rounds, invalid
    return False, None, MAX_ROUNDS, invalid


def evaluate(environment, num_agents, velocity, start=(0, 0, 1000), policy_factory=RulePolicy,
             trials=1, comms_quality=0.0, comms_distance=1000,
             show_coords=SHOW_TARGET_COORDS, show_own_location=SHOW_OWN_LOCATION, seed=0):
    spec = [(t.north_m, t.east_m, t.alt, t.target_type) for t in environment.target_locations]
    dists, rounds_list, invalid_total = [], [], 0
    for k in range(trials):
        ok, dist, rounds, invalid = run_once(spec, num_agents, velocity, start, policy_factory,
                                             comms_quality, comms_distance,
                                             show_coords, show_own_location, seed + k)
        invalid_total += invalid
        if ok:
            dists.append(dist)
            rounds_list.append(rounds)
    n = len(dists)
    return RuleResult(trials, n / trials,
                      sum(dists) / n if n else None,
                      sum(rounds_list) / n if n else None,
                      invalid_total)
