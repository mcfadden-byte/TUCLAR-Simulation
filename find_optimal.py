# Code written entirely by AI.

"""
Min-max multi-depot routing.

Problem:
    - n drones, each with a fixed start location (its "depot").
    - m targets that must each be visited by exactly one drone, exactly once.
    - Each drone returns to its own start location after finishing its targets.
    - Minimize the MAXIMUM total travel distance among all drones (min-max,
      not min-sum) -- this balances the workload instead of just minimizing
      total fleet distance.

Two solvers are provided:

1. solve_exact(drones, targets)
   Optimal via dynamic programming over (assignment-of-targets-to-drones x
   subset-of-targets-per-drone). Practical up to ~12-14 targets total
   (2^m growth), regardless of drone count.

2. solve_heuristic(drones, targets, iters=2000)
   For larger instances: greedy min-max assignment + 2-opt/relocate local
   search to balance and shorten routes. No optimality guarantee, but scales
   to hundreds of targets.

Both return the same shape: a list of length n, where result[i] is the
ordered list of location indices (using a combined indexing scheme, see
below) visited by drone i, starting and ending at that drone's depot.

Indexing convention used in the returned "matrix" (list of lists):
    Each entry is a target index (0..m-1) in visiting order. The implicit
    start/end at the drone's own depot is NOT repeated in the list -- if you
    want the full path including start/end, use `full_path()` below, which
    inserts the drone's own [x,y,z] at the front and back.
"""

import math
import itertools
from functools import lru_cache


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


# ---------------------------------------------------------------------------
# Exact solver (optimal), via DP.
#
# Approach:
#   Step 1: For a *single* depot and a given subset S of targets, the optimal
#           route cost is the classic TSP-path (open loop start=end=depot)
#           cost for that subset. We compute this with the standard
#           Held-Karp DP: dp[S][j] = min cost to start at depot, visit
#           exactly the targets in S, ending at target j.
#           route_cost(depot, S) = min_j dp[S][j] + dist(j, depot)
#           This is O(2^k * k^2) per depot, k = |targets|.
#
#   Step 2: We need to partition all m targets among n drones (each subset
#           assigned to exactly one drone) to minimize the max over drones of
#           route_cost(depot_i, S_i). This is itself solved by DP over
#           bitmask of targets assigned so far:
#               best[mask] = for each prefix of drones (processed in some
#               fixed order 0..n-1), the minimum possible "max route cost so
#               far" achievable by distributing exactly `mask` targets among
#               drones 0..i-1.
#           We do this drone by drone: dp_drone[i][mask] = min possible
#           max-cost using drones 0..i to cover exactly `mask` targets.
#           dp_drone[i][mask] = min over submask s of mask of
#               max( dp_drone[i-1][mask ^ s], route_cost(depot_i, s) )
#           Final answer = dp_drone[n-1][full_mask].
#           This is O(n * 3^m) (submask enumeration) -- fine for m <= ~14-16
#           and moderate n.
#
# Total complexity: O(n * 2^m * m^2) for step 1 + O(n * 3^m) for step 2.
# Reasonable for m up to ~14 or so.
# ---------------------------------------------------------------------------

def _held_karp_all_subsets(depot, targets):
    """
    For a single depot, compute for every subset S of target-indices:
        - the min route cost (depot -> visit all of S -> depot)
        - the actual visiting order achieving it
    Returns dict: mask -> (cost, order_list)
    """
    k = len(targets)
    if k == 0:
        return {0: (0.0, [])}

    d_depot = [dist(depot, t) for t in targets]
    d_tt = [[dist(targets[i], targets[j]) for j in range(k)] for i in range(k)]

    # dp[mask][j] = min cost to start at depot, visit exactly targets in mask,
    # ending at target j (j must be in mask)
    size = 1 << k
    NEG = math.inf
    dp = [[NEG] * k for _ in range(size)]
    parent = [[-1] * k for _ in range(size)]

    for j in range(k):
        m0 = 1 << j
        dp[m0][j] = d_depot[j]

    for mask in range(size):
        for j in range(k):
            if not (mask & (1 << j)):
                continue
            cur = dp[mask][j]
            if cur == NEG:
                continue
            for nxt in range(k):
                if mask & (1 << nxt):
                    continue
                nmask = mask | (1 << nxt)
                cand = cur + d_tt[j][nxt]
                if cand < dp[nmask][nxt]:
                    dp[nmask][nxt] = cand
                    parent[nmask][nxt] = j

    # For every non-empty subset (as a mask), find best ending target and
    # reconstruct order; cost includes return to depot.
    result = {0: (0.0, [])}
    full_index_list = list(range(k))
    for mask in range(1, size):
        best_cost = NEG
        best_j = -1
        for j in range(k):
            if not (mask & (1 << j)):
                continue
            if dp[mask][j] == NEG:
                continue
            c = dp[mask][j] + d_depot[j]
            if c < best_cost:
                best_cost = c
                best_j = j
        if best_j == -1:
            continue
        # reconstruct
        order = []
        m, j = mask, best_j
        while j != -1:
            order.append(j)
            pj = parent[m][j]
            m ^= (1 << j)
            j = pj
        order.reverse()
        result[mask] = (best_cost, order)

    return result


def solve_exact(drones, targets):
    """
    drones: list of [x,y,z] start/end positions, one per vehicle.
    targets: list of [x,y,z] positions that must each be visited once.

    Returns: list of length len(drones); each element is the ordered list of
    target indices (into `targets`) that drone visits, in visiting order.
    An empty list means that drone visits no targets.
    """
    n = len(drones)
    m = len(targets)
    if m == 0:
        return [[] for _ in range(n)]
    if n == 0:
        raise ValueError("Need at least one drone to cover targets.")

    # Step 1: per-drone route table over all subsets
    per_drone_routes = [_held_karp_all_subsets(drones[i], targets) for i in range(n)]

    full_mask = (1 << m) - 1

    # Step 2: DP across drones to partition targets minimizing the max cost.
    # dp[i][mask] = min possible max-route-cost using drones 0..i-1 to cover
    # exactly `mask` targets (i drones "used up" so far, 0-indexed prefix).
    NEG = math.inf
    size = 1 << m
    dp = [[NEG] * size for _ in range(n + 1)]
    choice = [[None] * size for _ in range(n + 1)]  # store the submask given to drone i
    dp[0][0] = 0.0

    for i in range(n):
        routes_i = per_drone_routes[i]
        for mask in range(size):
            base = dp[i][mask]
            if base == NEG:
                continue
            remaining = full_mask & ~mask
            # enumerate submasks of `remaining` to assign to drone i
            sub = remaining
            while True:
                if sub in routes_i:
                    cost_i = routes_i[sub][0]
                    cand = max(base, cost_i)
                    nmask = mask | sub
                    if cand < dp[i + 1][nmask]:
                        dp[i + 1][nmask] = cand
                        choice[i + 1][nmask] = sub
                if sub == 0:
                    break
                sub = (sub - 1) & remaining

    if dp[n][full_mask] == NEG:
        raise RuntimeError("No feasible assignment found (unexpected).")

    # Backtrack to find each drone's assigned submask
    result = [[] for _ in range(n)]
    mask = full_mask
    for i in range(n, 0, -1):
        sub = choice[i][mask]
        if sub is None:
            sub = 0
        order = per_drone_routes[i - 1].get(sub, (0.0, []))[1]
        result[i - 1] = order
        mask ^= sub

    return result


# ---------------------------------------------------------------------------
# Heuristic solver (scales to larger m): greedy assignment + local search.
# ---------------------------------------------------------------------------

def _route_cost(depot, order, targets):
    if not order:
        return 0.0
    c = dist(depot, targets[order[0]])
    for a, b in zip(order, order[1:]):
        c += dist(targets[a], targets[b])
    c += dist(targets[order[-1]], depot)
    return c


def _nearest_insertion_order(depot, indices, targets):
    """Build a reasonable route for a set of target indices via nearest
    neighbor + simple improvement, used as a fast approximate TSP-path."""
    if not indices:
        return []
    remaining = list(indices)
    # nearest neighbor from depot
    order = []
    cur = depot
    while remaining:
        nxt = min(remaining, key=lambda j: dist(cur, targets[j]))
        order.append(nxt)
        remaining.remove(nxt)
        cur = targets[nxt]
    return order


def _two_opt(depot, order, targets):
    """Standard 2-opt local search on a single route (open path, depot fixed
    endpoint on both ends)."""
    improved = True
    n = len(order)
    if n < 2:
        return order
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                if _route_cost(depot, new_order, targets) < _route_cost(depot, order, targets) - 1e-9:
                    order = new_order
                    improved = True
    return order


def solve_heuristic(drones, targets, iters=500, seed=0):
    """
    Greedy min-max assignment (assign each target to the drone for which
    adding it increases that drone's route cost the least, always choosing
    to grow whichever drone currently has the smallest projected route),
    then relocate/2-opt local search to reduce the maximum route length.
    """
    import random
    rng = random.Random(seed)

    n = len(drones)
    m = len(targets)
    if m == 0:
        return [[] for _ in range(n)]

    routes = [[] for _ in range(n)]

    unassigned = list(range(m))
    # Greedy: repeatedly assign the target whose cheapest insertion (over all
    # drones/positions) is cheapest, biased toward keeping max route low.
    while unassigned:
        best = None  # (added_cost_to_that_drone, drone_i, target_j, insert_pos, new_route_cost, resulting_max)
        cur_costs = [_route_cost(drones[i], routes[i], targets) for i in range(n)]
        for j in unassigned:
            for i in range(n):
                route = routes[i]
                for pos in range(len(route) + 1):
                    trial = route[:pos] + [j] + route[pos:]
                    c = _route_cost(drones[i], trial, targets)
                    added = c - cur_costs[i]
                    resulting_max = max(c, max(cur_costs[k] for k in range(n) if k != i))
                    key = (resulting_max, added)
                    if best is None or key < best[0]:
                        best = (key, i, j, pos)
        _, i, j, pos = best
        routes[i] = routes[i][:pos] + [j] + routes[i][pos:]
        unassigned.remove(j)

    # Local search: 2-opt within each route
    for i in range(n):
        routes[i] = _two_opt(drones[i], routes[i], targets)

    # Local search: relocate single targets between routes if it reduces the max
    def total_max():
        return max(_route_cost(drones[i], routes[i], targets) for i in range(n))

    improved = True
    it = 0
    while improved and it < iters:
        improved = False
        it += 1
        cur_max = total_max()
        costs = [_route_cost(drones[i], routes[i], targets) for i in range(n)]
        src = max(range(n), key=lambda i: costs[i])  # try to relieve the bottleneck drone
        if not routes[src]:
            break
        for j_idx, j in enumerate(routes[src]):
            trial_src = routes[src][:j_idx] + routes[src][j_idx+1:]
            src_cost = _route_cost(drones[src], trial_src, targets)
            for dst in range(n):
                if dst == src:
                    continue
                for pos in range(len(routes[dst]) + 1):
                    trial_dst = routes[dst][:pos] + [j] + routes[dst][pos:]
                    dst_cost = _route_cost(drones[dst], trial_dst, targets)
                    others_max = max(
                        (costs[k] for k in range(n) if k != src and k != dst),
                        default=0.0
                    )
                    new_max = max(src_cost, dst_cost, others_max)
                    if new_max < cur_max - 1e-9:
                        routes[src] = trial_src
                        routes[dst] = trial_dst
                        routes[src] = _two_opt(drones[src], routes[src], targets)
                        routes[dst] = _two_opt(drones[dst], routes[dst], targets)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return routes


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def full_path(drones, targets, routes, drone_i):
    """Return the full path (list of [x,y,z]) for one drone, including its
    own start and end position."""
    order = routes[drone_i]
    return [drones[drone_i]] + [targets[j] for j in order] + [drones[drone_i]]


def route_lengths(drones, targets, routes):
    return [_route_cost(drones[i], routes[i], targets) for i in range(len(drones))]


def solve(drones, targets, exact_limit=13):
    """
    Convenience dispatcher: uses the exact solver when the number of targets
    is small enough to be tractable (<= exact_limit), otherwise falls back
    to the heuristic.
    """
    if len(targets) <= exact_limit:
        return solve_exact(drones, targets)
    return solve_heuristic(drones, targets)


# ---------------------------------------------------------------------------
# Example / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    drones = [
        [0, 0, 0],
        [10, 0, 0],
    ]
    targets = [
        [1, 1, 0],
        [2, 2, 0],
        [9, 1, 0],
        [8, 2, 0],
        [5, 5, 0],
        [1, 5, 0],
    ]

    routes = solve_exact(drones, targets)
    lengths = route_lengths(drones, targets, routes)
    print("Exact solution:")
    for i, r in enumerate(routes):
        print(f"  Drone {i}: targets {r}  (route length {lengths[i]:.3f})")
    print(f"  Max route length (objective): {max(lengths):.3f}")

    routes_h = solve_heuristic(drones, targets)
    lengths_h = route_lengths(drones, targets, routes_h)
    print("\nHeuristic solution:")
    for i, r in enumerate(routes_h):
        print(f"  Drone {i}: targets {r}  (route length {lengths_h[i]:.3f})")
    print(f"  Max route length (objective): {max(lengths_h):.3f}")
