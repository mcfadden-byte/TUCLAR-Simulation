"""Best-known solution finder.

Instead of solving the routing problem analytically, this runs many stripped-down simulations
(no LLM, no plotting, no I/O) with a randomized coordinator in charge of the drones, and keeps
the run with the smallest total distance flown.

The movement and visit rules are NOT reimplemented here: the runs call the simulator's own
AgentHandler.update_velocity, Agent.step, Environment.check_visits and TargetLocation.record_visits,
so the benchmark follows the same mechanics (visit radius, Joint/Sequential clearing, RECALL to
(0, 0, 0)) as the real experiment. If those change, this stays correct automatically.

The result is the best of N random tries, not a proof of optimality; more iterations can only
improve it.
"""
import math
import random
import time
from dataclasses import dataclass
from types import SimpleNamespace

import utils_geometry as geo
from agent import Agent, AgentHandler
from environment import Environment, TargetLocation, TargetType, VISIT_RADIUS_METERS

HOME = [0, 0, 0]        # where RECALL sends a drone in AgentHandler.update_velocity
HOME_TOLERANCE = 1      # same "home" test as Simulation.check_all_agents_home
SCENARIO_0_RADIUS = 50  # same success radius as Simulation.check_success, scenario 0
MAX_ROUNDS = 400        # a run that takes longer than this is discarded


@dataclass
class OptimalSolution:
    distance: float      # smallest total distance flown by any successful run (meters)
    rounds: int          # fewest rounds taken by any successful run (may come from a different run)
    iterations: int      # runs actually executed
    successes: int       # runs that finished the mission


def solve(environment, scenario: int, num_agents: int, velocity: float,
          start=(0, 0, 1000), iterations: int = 2000, time_limit: float = 20.0, seed: int = 0):
    """Return an OptimalSolution for the scenario, or None if no run ever succeeded."""
    match scenario:
        case 0:
            return _solve_single_target(environment, velocity, start)
        case 1:
            return _solve_multi_target(environment, num_agents, velocity, start, iterations, time_limit, seed)
    raise NotImplementedError(f"No optimal-solution finder for scenario {scenario}")


# Scenario 0: one drone, one target; flying straight at it is optimal.
def _solve_single_target(environment, velocity, start):
    target = environment.target_locations[0].get_location()
    agent = Agent("", None, "", list(start), 0)
    agent.target_index = 0
    handler = SimpleNamespace(drone_velocity=velocity, environment=environment)

    rounds = 0
    while geo.point_distance(agent.location, target) >= SCENARIO_0_RADIUS and rounds < MAX_ROUNDS:
        AgentHandler.update_velocity(handler, agent)
        agent.step()
        rounds += 1
    return OptimalSolution(agent.distance_traveled, rounds, 1, 1)


# Scenario 1: many randomized runs, keep the one with the smallest distance.
def _solve_multi_target(environment, num_agents, velocity, start, iterations, time_limit, seed):
    rng = random.Random(seed)
    spec = [(t.north_m, t.east_m, t.alt, t.target_type) for t in environment.target_locations]

    best_distance = best_distance_rounds = best_rounds = None
    successes = ran = 0
    deadline = time.perf_counter() + time_limit

    for _ in range(iterations):
        if time.perf_counter() > deadline:
            break
        ran += 1
        result = _run_once(rng, spec, num_agents, velocity, start, alpha=rng.uniform(0, 8))
        if result is None:
            continue
        distance, rounds = result
        successes += 1
        if best_distance is None or (distance, rounds) < (best_distance, best_distance_rounds):
            best_distance, best_distance_rounds = distance, rounds
        if best_rounds is None or rounds < best_rounds:
            best_rounds = rounds

    if best_distance is None:
        return None
    return OptimalSolution(best_distance, best_rounds, ran, successes)


def _run_once(rng, spec, num_agents, velocity, start, alpha):
    """One stripped-down mission. alpha sets how strongly drones prefer nearby targets
    (0 = uniformly random, large = nearest first). Returns (total distance, rounds) or None."""
    targets = [TargetLocation(n, e, a, kind) for n, e, a, kind in spec]
    env = SimpleNamespace(target_locations=targets)
    handler = SimpleNamespace(drone_velocity=velocity, environment=env)
    agents = [Agent("", None, "", list(start), i) for i in range(num_agents)]

    for rounds in range(1, MAX_ROUNDS + 1):
        order = list(range(num_agents))
        rng.shuffle(order)
        for i in order:
            _decide(rng, agents[i], agents, targets, alpha)

        for agent in agents:
            AgentHandler.update_velocity(handler, agent)
            agent.step()
        Environment.check_visits(env, agents)

        if all(t.cleared for t in targets) and all(
            geo.point_distance(a.location, HOME) < HOME_TOLERANCE for a in agents
        ):
            return sum(a.distance_traveled for a in agents), rounds
    return None


def _decide(rng, agent, agents, targets, alpha):
    """Omniscient coordinator: give a drone a new order only when it needs one."""
    if agent.inaction:
        return

    current = agent.target_index
    if isinstance(current, int) and current >= 0:
        target = targets[current]
        if not target.cleared:
            arrived = geo.point_distance(agent.location, target.get_location()) < VISIT_RADIUS_METERS
            if not arrived:
                return                                   # still flying
            if target.target_type == TargetType.JOINT:
                return                                   # arrived; hold for the partner

    # Needs a new order: pick among targets that can still use another drone
    candidates = []
    for j, target in enumerate(targets):
        if target.cleared:
            continue
        sequential = target.target_type == TargetType.SEQUENTIAL
        if sequential and agent.agent_index in target.visitors:
            continue                                     # already counted toward this one
        heading = sum(
            1 for other in agents
            if other is not agent and not other.inaction and other.target_index == j
            and not (sequential and other.agent_index in target.visitors)
        )
        if heading < target.slots_remaining():
            candidates.append(j)

    if not candidates:
        agent.target_index = "Home"
        agent.inaction = True
        return

    weights = [max(1.0, geo.point_distance(agent.location, targets[j].get_location())) ** -alpha
               for j in candidates]
    agent.target_index = rng.choices(candidates, weights=weights)[0]
