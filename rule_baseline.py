"""Decentralized rule-based baseline (no LLM, no central coordinator).

Each drone owns its own policy instance and decides from an Observation containing only what the
LLM prompt shows: teammates' last orders, received messages, target status, and (optionally)
coordinates. Decisions in a round are made simultaneously from a snapshot taken before anyone
acts, so a drone never sees a teammate's current-round choice. Movement and visit mechanics reuse
the simulator's own methods, as in optimal.py.
"""
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace

import utils_geometry as geo
from agent import (
    MAX_ROUNDS,
    SHOW_OWN_LOCATION,
    SHOW_TARGET_COORDS,
    TARGET_TYPE_VISIBILITY,
    Agent,
    AgentHandler,
    type_is_known,
)
from environment import Environment, TargetLocation, TargetType

HOME = [0, 0, 0]
HOME_TOLERANCE = 1


# ---------------------------------------------------------------------------
# What a drone is allowed to know
# ---------------------------------------------------------------------------

# TargetView: Represents all information related to a target
@dataclass(frozen=True)
class TargetView:
    index: int
    kind: TargetType | None
    cleared: bool
    required_drones: int | None
    visited_by_me: bool
    claimants: tuple                      # agent indices (incl. me) whose last order was this target
    location: tuple | None = None      # (north, east, alt), only if coordinates are shown
    distance: float | None = None      # from me, only if coordinates are shown


# TeammateView: Represents all information related to a teammate
@dataclass(frozen=True)
class TeammateView:
    index: int
    last_target: object                   # int target index, None (idle), or -1 (unknown yet)
    message: str | None                # None if out of range or comms disabled

# Observation: the entirety of information a drone is given
@dataclass(frozen=True)
class Observation:
    agent_index: int
    num_agents: int
    current_target: object                # Target the agent selected last round
    own_location: tuple | None         # only if SHOW_OWN_LOCATION
    teammates: tuple                      # active teammates only; recalled teammates do no appear
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
# These classes relate to how the drones make decisions

class DronePolicy:
    """One instance per drone. Keep any memory on self; instances never share state."""

    def __init__(self, agent_index: int, num_agents: int, rng: random.Random):
        self.agent_index = agent_index
        self.num_agents = num_agents
        self.rng = rng # A seeded random number generator
        self.memory = {}

    def decide(self, obs: Observation) -> Decision:
        raise NotImplementedError


# The rules that dictate how drones decide what to do
class RulePolicy(DronePolicy):
    UNKNOWN_SLOT_GUESS = 1

    def decide(self, obs: Observation) -> Decision:
        # If no targets left, go home
        if obs.all_cleared:
            return Decision("RECALL")

        me = obs.agent_index
        target = {t.index: t for t in obs.targets}

        def holds(last):
            """True if an order of `last` is still a live claim on an uncleared target."""
            return isinstance(last, int) and last >= 0 and not target[last].cleared

        # 1. Keep an existing claim, unless it is a Sequential this drone has already visited
        mine = obs.current_target
        released = False
        if holds(mine):
            t = target[mine]
            if t.kind == TargetType.SEQUENTIAL and t.visited_by_me:
                released = True
            else:
                return Decision(f"FLY {mine}", f"Continuing to target {mine}.")

        # 2. Who is going to what, who is free (currently the same for every drone)
        orders = {me: mine, **{m.index: m.last_target for m in obs.teammates}}
        claims = {t.index: 0 for t in obs.targets}
        free = []
        for i in sorted(orders):
            if holds(orders[i]):
                claims[orders[i]] += 1
            else:
                free.append(i)

        # 3. Open slots, listed by target index (a target appears once per unclaimed slot)
        slots = []
        for t in obs.targets:
            if not t.cleared:
                need = t.required_drones if t.required_drones is not None else self.UNKNOWN_SLOT_GUESS
                slots += [t.index] * max(0, need - claims[t.index])


        # 4. Free drones take slots in agent order: the k-th free drone takes the k-th open slot.
        #    A drone leaving a Sequential isn't in the others' free list, so it takes the next slot after theirs.
        rank = len(free) if released else free.index(me)
        for j in slots[rank:]:
            t = target[j]
            if t.kind == TargetType.SEQUENTIAL and t.visited_by_me:
                continue             # can't count twice toward the same Sequential
            return Decision(f"FLY {j}", f"Taking target {j}.")

        # More free drones than open slots; using IDLE instead of RECALL in case more spots open in the future
        # Adding targets mid-simulation currently isn't a thing, but may be in the future
        return Decision("IDLE")

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
# The implementation for the simulation
@dataclass
class RuleResult:
    trials: int
    success_rate: float
    avg_distance: float | None         # over successful trials
    avg_rounds: float | None           # over successful trials
    invalid_actions: int                  # total across all trials
    avg_ns_per_drone_round: float | None = None


def _observe(me, agents, targets, comms_quality, comms_distance, show_coords, show_own_location, failed_inference_chance=0.0, noise_rng=None, type_visibility=TARGET_TYPE_VISIBILITY, vision_obstacles=()):
    teammates = []
    for other in agents:
        if other is me or other.inaction:
            continue

        last = other.last_target
        if any(o.blocks(me.location, other.location) for o in vision_obstacles):
            last = -1   # no line of sight: target unknown
        elif noise_rng is not None and noise_rng.random() < failed_inference_chance:
            last = noise_rng.randrange(len(targets))

        message = None
        if comms_quality > 0.0:
            in_range = (comms_distance == -1 or
                        geo.geometric_mean_3d(*other.location, *me.location) <= comms_distance)
            message = other.displayed_message if in_range else None
        teammates.append(TeammateView(other.agent_index, last, message))

    seen = {me.agent_index: me.last_target, **{t.index: t.last_target for t in teammates}}

    views = []
    for j, t in enumerate(targets):
        known = type_is_known(t, me.agent_index, type_visibility)
        claimants = tuple(sorted(i for i, lt in seen.items() if lt == j))
        loc = dist = None
        if show_coords:
            loc = t.get_location()
            dist = geo.point_distance(me.location, loc)
        views.append(TargetView(j,
                                t.target_type if known else None,
                                t.cleared,
                                t.required_drones if known else None,
                                me.agent_index in t.visitors,
                                claimants,
                                loc,
                                dist))

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


def run_once(spec, num_agents, velocity, start, policy_factory, comms_quality, comms_distance, show_coords, show_own_location, seed, failed_inference_chance=0.0, type_visibility=TARGET_TYPE_VISIBILITY, vision_obstacles=()):
    rng = random.Random(seed)
    targets = [TargetLocation(n, e, a, kind) for n, e, a, kind in spec]
    env = SimpleNamespace(target_locations=targets)
    handler = SimpleNamespace(drone_velocity=velocity, environment=env)
    agents = [Agent("", None, "", list(start), i) for i in range(num_agents)]
    policies = [policy_factory(i, num_agents, random.Random(rng.getrandbits(32)))
                for i in range(num_agents)]
    noise_rng = random.Random(rng.getrandbits(32))
    invalid = 0
    t0 = time.perf_counter_ns()

    for rounds in range(1, MAX_ROUNDS + 1):
        # Snapshot first: every drone decides from the same pre-round state
        observations = {a.agent_index: _observe(a, agents, targets, comms_quality, comms_distance, show_coords, show_own_location, failed_inference_chance, noise_rng, type_visibility, vision_obstacles)
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
            return True, sum(a.distance_traveled for a in agents), rounds, invalid, time.perf_counter_ns() - t0
    return False, None, MAX_ROUNDS, invalid, time.perf_counter_ns() - t0


def evaluate(environment, num_agents, velocity, start=(0, 0, 1000), policy_factory=RulePolicy,
             trials=1, comms_quality=0.0, comms_distance=1000,
             show_coords=SHOW_TARGET_COORDS, show_own_location=SHOW_OWN_LOCATION, seed=0, failed_inference_chance=0.0, type_visibility=TARGET_TYPE_VISIBILITY):
    spec = [(t.north_m, t.east_m, t.alt, t.target_type) for t in environment.target_locations]
    dists, rounds_list, invalid_total = [], [], 0
    total_ns = total_drone_rounds = 0
    for k in range(trials):
        ok, dist, rounds, invalid, elapsed_ns = run_once(spec,
                                            num_agents,
                                            velocity,
                                            start,
                                            policy_factory,
                                            comms_quality,
                                            comms_distance,
                                            show_coords,
                                            show_own_location,
                                            seed + k,
                                            failed_inference_chance,
                                            type_visibility,
                                            environment.vision_obstacles)
        total_ns = elapsed_ns
        total_drone_rounds += num_agents * rounds
        invalid_total += invalid
        if ok:
            dists.append(dist)
            rounds_list.append(rounds)
    n = len(dists)
    return RuleResult(  trials,
                        n / trials,
                        sum(dists) / n if n else None,
                        sum(rounds_list) / n if n else None,
                        invalid_total,
                        total_ns / total_drone_rounds if total_drone_rounds else None)
