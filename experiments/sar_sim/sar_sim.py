"""Search-and-rescue case study: a UAV team with one kind of task (search an area) under limited communication.

World
  - N_AREAS search areas around the base. Each has a terrain feature and takes 4-8 rounds to sweep.
  - One missing person is in one area. The area is drawn from a lost-person profile: a preference over terrain
    features times a decay with distance from the last known point (LKP). The mission brief describes the
    person and the LKP in words only; the numbers behind it are never shown to any method.
  - A drone flies SPEED m per round. The person is found when a drone finishes sweeping the person's area.
    Sweeping an area that is already searched, or one that a teammate is sweeping, only wastes time.
  - Constraint: a full battery lasts BATTERY rounds of flying, sweeping or hovering, and recharging at the base
    takes RECHARGE rounds. A drone whose battery runs out away from the base crashes and is lost.
Communication
  - Each pair of drones has a link that is up or down: a two-state Markov chain that is up a fraction q of the
    time, with up-windows of UP_WINDOW rounds on average. While a link is up, the two drones exchange their
    current claim and the areas they know to be searched (codebook messages claim(a), done(a, ...)).
    A drone that takes up a new claim broadcasts it at once over the links that are up.
  - Link states depend only on the seed and q, so every method sees the same outages.
Task tree of every drone
  mission -> search(a) -> fly_to(a), sweep(a)     a ranges over the areas this drone believes still open
  Blanks: which areas teammates have searched or claimed (messages; LLM reasoning for silent teammates) and
  how likely each area is to hold the person (LLM reasoning from the brief; uniform without it).
"""
import math
import random
from dataclasses import dataclass, field

N_DRONES = 5
N_AREAS = 12
HALF = 1250.0          # areas lie in [-HALF, HALF]^2 around the base at (0, 0)
MIN_SEP = 420.0
SPEED = 100.0          # m per round
UP_WINDOW = 4.0        # mean rounds a link stays up
MAX_ROUNDS = 300
# A task-level message from the shared task tree: sender (3 bits), claimed area (4 bits: 12 areas or none),
# status (3 bits) and the searched areas as a 12-bit bitmap.
STATE_BITS = 3 + 4 + 3 + N_AREAS
BATTERY = 60           # rounds of flying, sweeping or hovering on a full charge
RECHARGE = 8           # rounds to recharge at the base
RETURN = -1            # action: fly back to the base and recharge
PLAN_DEPTH = 4         # areas looked ahead by the planner
PLAN_CANDIDATES = 8

FEATURES = ["trail", "stream", "lake shore", "ridge", "forest", "meadow", "road", "cabin"]

# Lost-person profiles. Terrain weights follow general lost-person behaviour (injured hikers stay by paths,
# young children shelter in cover close by, people with dementia follow roads and paths and end up in brush,
# anglers and people out of water are drawn to water). decay: metres over which the weight falls by e.
PROFILES = {
    "injured_hiker": dict(
        decay=450, lkp=["trail", "ridge"],
        w={"trail": 4, "meadow": 2, "ridge": 2, "cabin": 2, "forest": 1, "stream": 1, "lake shore": 1, "road": 1},
        brief="A 68-year-old hiker twisted his knee and was last seen at the {lkp}. On a short phone call he told "
              "his partner that he would sit down near the path and wait for help."),
    "child": dict(
        decay=500, lkp=["meadow", "cabin"],
        w={"forest": 4, "cabin": 3, "meadow": 2, "stream": 1, "trail": 1, "lake shore": 0.5, "road": 0.5,
           "ridge": 0.3},
        brief="A 5-year-old girl wandered off from a family picnic at the {lkp} an hour ago. A large dog had "
              "frightened her just before she disappeared."),
    "dementia": dict(
        decay=1000, lkp=["road"],
        w={"road": 4, "trail": 3, "forest": 3, "stream": 1, "meadow": 1, "cabin": 1, "lake shore": 0.5,
           "ridge": 0.2},
        brief="An 81-year-old man with dementia walked away from his car at the {lkp} two hours ago. His family "
              "says he likes long walks and does not respond well to his name being called."),
    "angler": dict(
        decay=1000, lkp=["road", "trail"],
        w={"stream": 5, "lake shore": 5, "trail": 1, "forest": 0.5, "meadow": 0.5, "road": 0.5, "cabin": 0.5,
           "ridge": 0.2},
        brief="A 40-year-old man left his car at the {lkp} this morning to go fishing and has not come back. His "
              "rod and tackle box are not in the car."),
    "runner": dict(
        decay=1200, lkp=["trail"],
        w={"stream": 4, "lake shore": 4, "trail": 3, "cabin": 2, "road": 1, "meadow": 1, "forest": 1, "ridge": 0.5},
        brief="A 30-year-old trail runner was last seen at the {lkp} on a hot afternoon. A friend she passed says "
              "she had already finished her water."),
}


@dataclass
class Area:
    idx: int
    x: float
    y: float
    feature: str
    sweep: int


@dataclass
class Scenario:
    seed: int
    areas: list
    profile: str
    lkp: int
    brief: str
    true_p: list          # the profile's probability of each area (used only for scoring)
    person: int


def make_scenario(seed):
    rng = random.Random(seed * 7919 + 17)
    pts = []
    while len(pts) < N_AREAS:
        x, y = rng.uniform(-HALF, HALF), rng.uniform(-HALF, HALF)
        if math.hypot(x, y) < 300 or any(math.hypot(x - a, y - b) < MIN_SEP for a, b in pts):
            continue
        pts.append((x, y))
    feats = FEATURES + [rng.choice(FEATURES) for _ in range(N_AREAS - len(FEATURES))]
    rng.shuffle(feats)
    areas = [Area(i, x, y, f, rng.randint(4, 8)) for i, ((x, y), f) in enumerate(zip(pts, feats))]
    profile = rng.choice(sorted(PROFILES))
    prof = PROFILES[profile]
    cands = [a.idx for a in areas if a.feature in prof["lkp"]]
    lkp = rng.choice(cands)
    w = [prof["w"][a.feature] * math.exp(-dist(areas[lkp], a) / prof["decay"]) for a in areas]
    s = sum(w)
    p = [v / s for v in w]
    person = rng.choices(range(N_AREAS), weights=p)[0]
    brief = prof["brief"].format(lkp=f"{areas[lkp].feature} in area {lkp}")
    return Scenario(seed, areas, profile, lkp, brief, p, person)


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


class Links:
    """Pairwise link states; identical for every method with the same seed and q."""

    def __init__(self, seed, q, n=N_DRONES):
        self.q = q
        self.rng = random.Random(seed * 104729 + int(round(q * 1000)))
        self.pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if q <= 0:
            self.up = {p: False for p in self.pairs}
        elif q >= 1:
            self.up = {p: True for p in self.pairs}
        else:
            self.p_down = 1.0 / UP_WINDOW                       # up -> down
            self.p_up = q * self.p_down / (1.0 - q)              # down -> up, so that the chain is up a fraction q
            self.up = {p: self.rng.random() < q for p in self.pairs}

    def is_up(self, i, j):
        return self.up[(min(i, j), max(i, j))]

    def step(self):
        if 0 < self.q < 1:
            for p in self.pairs:
                r = self.rng.random()
                self.up[p] = (r >= self.p_down) if self.up[p] else (r < self.p_up)


@dataclass
class Drone:
    idx: int
    x: float = 0.0
    y: float = 0.0
    target: int = None
    status: str = "idle"                     # idle | fly | sweep | return | charge | lost
    sweep_left: int = 0
    battery: int = BATTERY
    charge_left: int = 0
    travelled: float = 0.0
    sweep_rounds: int = 0
    wasted_sweep: int = 0
    own_done: set = field(default_factory=set)
    known_done: set = field(default_factory=set)
    mate_claim: dict = field(default_factory=dict)   # j -> (area or None, status, round heard)
    heard: dict = field(default_factory=dict)        # j -> last round a message from j arrived
    message: str = ""                                # natural-language message (LLM-only); empty for the planner methods
    mate_text: dict = field(default_factory=dict)    # j -> (j's natural-language message, round heard)
    decisions: int = 0


# ---------------------------------------------------------------------------------------------------------------
# Planner: the task tree mission -> search(a) -> fly_to(a), sweep(a), searched by branch-and-bound
# ---------------------------------------------------------------------------------------------------------------
def flight(p, q):
    """Rounds to fly from p to q (a drone covers SPEED m per round and arrives in the round it gets within reach)."""
    return math.ceil(math.hypot(p[0] - q[0], p[1] - q[1]) / SPEED - 1e-9)


def at_base(d):
    return math.hypot(d.x, d.y) < 1e-6


def open_areas(d, sc, set_aside=()):
    claimed = {c[0] for c in d.mate_claim.values() if c[0] is not None}
    return [a.idx for a in sc.areas if a.idx not in d.known_done and a.idx not in claimed and a.idx not in set_aside]


def plan_route(d, sc, prior, candidates):
    """Order of searches that minimises the expected time to find the person (truncated at PLAN_DEPTH):
    sum over planned areas of p(a) * completion(a), plus the probability left in unplanned open areas times
    the completion time of the last planned area. Returns the best sequence of area indices."""
    if not candidates:
        return []
    pos = (d.x, d.y)
    mass = {a: prior[a] for a in candidates}
    total = sum(mass.values()) or 1.0
    mass = {a: v / total for a, v in mass.items()} if sum(mass.values()) > 0 else {a: 1 / len(candidates)
                                                                                   for a in candidates}

    def leg(p, a):
        ar = sc.areas[a]
        return flight(p, (ar.x, ar.y)) + ar.sweep

    def home(p):
        return flight(p, (0.0, 0.0))

    # constraint: a search enters the plan only if the battery also covers the flight home that closes it
    feasible = [a for a in candidates if leg(pos, a) + home((sc.areas[a].x, sc.areas[a].y)) <= d.battery]
    # keep the candidates with the best ratio of probability to time from here
    ranked = sorted(feasible, key=lambda a: (-mass[a] / leg(pos, a), a))[:PLAN_CANDIDATES]
    best = [math.inf, None]
    depth = min(PLAN_DEPTH, len(ranked))

    def dfs(p, t, cost, left, seq, used):
        if len(seq) == depth:
            total_cost = cost + left * t
            if total_cost < best[0]:
                best[0], best[1] = total_cost, list(seq)
            return
        if cost + left * t >= best[0]:       # every remaining area finishes after t
            return
        nxt = [a for a in ranked if a not in used
               and t + leg(p, a) + home((sc.areas[a].x, sc.areas[a].y)) <= d.battery]
        if not nxt:                          # the battery ends this sortie: the rest waits for a recharge
            total_cost = cost + left * (t + home(p) + RECHARGE)
            if seq and total_cost < best[0]:
                best[0], best[1] = total_cost, list(seq)
            return
        for a in sorted(nxt, key=lambda a: leg(p, a)):
            ar = sc.areas[a]
            t2 = t + leg(p, a)
            seq.append(a)
            used.add(a)
            dfs((ar.x, ar.y), t2, cost + mass[a] * t2, left - mass[a], seq, used)
            seq.pop()
            used.discard(a)

    dfs(pos, 0.0, 0.0, 1.0, [], set())
    return best[1] or []


def choose_area(d, sc, prior, set_aside=()):
    """First area of the planned route. If nothing is open, fall back first to ignoring LLM set-asides, then to
    areas a teammate has claimed (a drone never idles while an area is not known to be searched). RETURN when the
    battery cannot cover any remaining search and the flight home."""
    for cands in (open_areas(d, sc, set_aside), open_areas(d, sc),
                  [a.idx for a in sc.areas if a.idx not in d.known_done]):
        if cands:
            route = plan_route(d, sc, prior, cands)
            return (route[0], route) if route else (RETURN, [])
    return (None if at_base(d) else RETURN), []


# ---------------------------------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------------------------------
class Sim:
    def __init__(self, seed, q, method, llm=None, log=None):
        self.sc = make_scenario(seed)
        self.links = Links(seed, q)
        self.q = q
        self.method = method                 # htn | llm | llm_nl | node
        self.llm = llm                       # sar_llm.LLMClient or a mock, for llm/node methods
        self.log = log if log is not None else []
        self.drones = [Drone(i) for i in range(N_DRONES)]
        self.round = 0
        self.searched = set()                # ground truth: areas finished by some drone
        self.found_round = None
        self.crashes = 0                     # drones whose battery ran out away from the base
        self.messages = 0                    # messages received by teammates
        self.msg_bits = 0                    # their total size
        self.brains = [make_brain(method, i, self) for i in range(N_DRONES)]

    # -- communication -------------------------------------------------------------------------------------------
    def deliver(self, i, j):
        """j receives i's current claim and searched areas, or i's natural-language message (LLM-only)."""
        di, dj = self.drones[i], self.drones[j]
        if getattr(self.brains[i], "nl_messages", False):
            if di.message:
                dj.mate_text[i] = (di.message, self.round)
                dj.heard[i] = self.round
                self.messages += 1
                self.msg_bits += 8 * len(di.message.encode("utf-8"))
            return
        self.messages += 1
        self.msg_bits += STATE_BITS
        dj.mate_claim[i] = (di.target, di.status, self.round)
        dj.heard[i] = self.round
        dj.known_done |= di.known_done
        # a claim on an area j now knows to be searched is void
        for k, c in list(dj.mate_claim.items()):
            if c[0] is not None and c[0] in dj.known_done and k != i:
                dj.mate_claim[k] = (None, "idle", c[2])
        if di.target is not None and di.target in dj.known_done and di.status != "sweep":
            dj.mate_claim[i] = (None, "idle", self.round)

    def exchange(self):
        for i in range(N_DRONES):
            for j in range(N_DRONES):
                if i != j and self.links.is_up(i, j) and "lost" not in (self.drones[i].status, self.drones[j].status):
                    self.deliver(i, j)

    def broadcast(self, i):
        for j in range(N_DRONES):
            if j != i and self.links.is_up(i, j) and self.drones[j].status != "lost":
                self.deliver(i, j)

    # -- decisions ---------------------------------------------------------------------------------------------------
    def needs_decision(self, d):
        if d.status in ("return", "charge", "lost"):
            return False
        if d.status == "idle":
            return True
        if d.status == "fly":
            if d.target in d.known_done:
                return True
            if not getattr(self.brains[d.idx], "uses_priority_rule", True):
                return False                     # LLM-only: the planner's claim protocol does not apply
            for j, (a, st, _) in d.mate_claim.items():
                if a == d.target and (st == "sweep" or (st == "fly" and j < d.idx)):
                    return True
        return False

    def decide(self, d):
        d.decisions += 1
        area = self.brains[d.idx].decide(d)
        if area is None:
            d.target, d.status = None, "idle"
        elif area == RETURN:
            d.target, d.status = None, "return"
        else:
            if area != d.target or d.status == "idle":
                d.target, d.status = area, "fly"
        self.broadcast(d.idx)

    # -- one round --------------------------------------------------------------------------------------------------
    def step(self):
        self.exchange()
        for d in self.drones:
            if self.needs_decision(d):
                self.decide(d)
        for d in self.drones:
            if d.status == "lost":
                continue
            if d.status == "charge":
                d.charge_left -= 1
                if d.charge_left <= 0:
                    d.battery, d.status = BATTERY, "idle"
                continue
            if not (d.status == "idle" and at_base(d)):
                d.battery -= 1                   # flying, sweeping, returning and hovering all drain the battery
            if d.status == "return":
                gap = math.hypot(d.x, d.y)
                step = min(SPEED, gap)
                if gap > 0:
                    d.x -= d.x * step / gap
                    d.y -= d.y * step / gap
                d.travelled += step
                if gap <= SPEED:
                    d.x, d.y, d.status, d.charge_left = 0.0, 0.0, "charge", RECHARGE
            elif d.status == "fly":
                ar = self.sc.areas[d.target]
                gap = math.hypot(ar.x - d.x, ar.y - d.y)
                step = min(SPEED, gap)
                if gap > 0:
                    d.x += (ar.x - d.x) * step / gap
                    d.y += (ar.y - d.y) * step / gap
                d.travelled += step
                if gap <= SPEED:
                    d.status, d.sweep_left = "sweep", ar.sweep
            elif d.status == "sweep":
                d.sweep_rounds += 1
                if d.target in self.searched or any(o is not d and o.status == "sweep" and o.target == d.target
                                                    and o.sweep_left < d.sweep_left for o in self.drones):
                    d.wasted_sweep += 1
                d.sweep_left -= 1
                if d.sweep_left <= 0:
                    a = d.target
                    if a == self.sc.person and self.found_round is None:
                        self.found_round = self.round + 1
                    self.searched.add(a)
                    d.own_done.add(a)
                    d.known_done.add(a)
                    d.target, d.status = None, "idle"
            if d.battery <= 0 and not at_base(d) and d.status != "charge":
                d.status, d.target = "lost", None        # constraint violated: the drone crashes
                self.crashes += 1
        self.links.step()
        self.round += 1

    def run(self):
        while self.found_round is None and self.round < MAX_ROUNDS:
            self.step()
        return self.result()

    def result(self):
        sc = self.sc
        order = sorted(range(N_AREAS), key=lambda a: -sc.true_p[a])
        prior0 = self.brains[0].prior
        prior_rank = sorted(range(N_AREAS), key=lambda a: (-prior0[a], a)).index(sc.person) + 1
        return dict(
            seed=sc.seed, q=self.q, method=self.method, profile=sc.profile,
            found=self.found_round is not None, rounds=self.found_round or MAX_ROUNDS,
            person_true_rank=order.index(sc.person) + 1, person_prior_rank=prior_rank,
            prior_mass_on_person=round(prior0[sc.person] / (sum(prior0) or 1), 4),
            sweep_rounds=sum(d.sweep_rounds for d in self.drones),
            wasted_sweep=sum(d.wasted_sweep for d in self.drones),
            distance=round(sum(d.travelled for d in self.drones)),
            areas_searched=len(self.searched),
            decisions=sum(d.decisions for d in self.drones),
            llm_queries=sum(getattr(b, "queries", 0) for b in self.brains),
            llm_seconds=round(sum(getattr(b, "llm_seconds", 0.0) for b in self.brains), 1),
            llm_failures=sum(getattr(b, "failures", 0) for b in self.brains),
            crashes=self.crashes,
            messages=self.messages,
            msg_bits=self.msg_bits,
        )


# ---------------------------------------------------------------------------------------------------------------
# Decision makers ("brains")
# ---------------------------------------------------------------------------------------------------------------
class HTNBrain:
    """Planner only, with a uniform prior over the areas."""

    def __init__(self, i, sim, prior):
        self.i, self.sim, self.prior = i, sim, prior

    def decide(self, d):
        return choose_area(d, self.sim.sc, self.prior)[0]


def make_brain(method, i, sim):
    uniform = [1.0] * N_AREAS
    if method == "htn":
        return HTNBrain(i, sim, uniform)
    import sar_llm
    if method == "llm":
        return sar_llm.LLMOnlyBrain(i, sim)
    if method == "llm_nl":                   # LLM-only that writes its own natural-language messages to teammates
        return sar_llm.LLMOnlyBrain(i, sim, nl_messages=True)
    if method == "node":
        return sar_llm.NodeBrain(i, sim)
    raise ValueError(method)
