"""Summarise <out>/results.csv (and log.jsonl) of run_sar.py.

  python aggregate_sar.py runs/x [runs/y ...]
Rounds to find the person: runs that never find the person count as MAX_ROUNDS (shown in the found column).
Paired differences use the same seeds (same scenario and link outages) in both methods.
"""
import csv
import json
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict

ORDER = ["htn", "llm", "llm_nl", "node"]
NAMES = {"htn": "HTN-only", "llm": "LLM-only", "llm_nl": "LLM-only, own messages", "node": "Ours"}

rows = []
for d in sys.argv[1:]:
    with open(os.path.join(d, "results.csv")) as f:
        rows += list(csv.DictReader(f))
data = defaultdict(dict)
for r in rows:
    data[(r["method"], float(r["q"]))][int(r["seed"])] = r
methods = [m for m in ORDER if any(k[0] == m for k in data)]
qs = sorted({k[1] for k in data})


def ms(vals):
    if not vals:
        return "-"
    if len(vals) < 2:
        return f"{vals[0]:.1f}"
    return f"{st.mean(vals):.1f}±{1.96 * st.stdev(vals) / math.sqrt(len(vals)):.1f}"


print("Rounds until the person is found (mean ± 95% CI); found/total if any run failed")
print(f"{'comms':>5}  " + "  ".join(f"{NAMES[m]:>20}" for m in methods))
for q in qs:
    cells = []
    for m in methods:
        runs = data.get((m, q), {})
        vals = [float(r["rounds"]) for r in runs.values()]
        nf = sum(r["found"] == "True" for r in runs.values())
        c = ms(vals) + ("" if nf == len(runs) else f" ({nf}/{len(runs)})")
        cells.append(c)
    print(f"{q:5.1f}  " + "  ".join(f"{c:>20}" for c in cells))

print("\nWasted sweep rounds per run (sweeping an area already searched or being finished by a teammate)")
print(f"{'comms':>5}  " + "  ".join(f"{NAMES[m]:>20}" for m in methods))
for q in qs:
    print(f"{q:5.1f}  " + "  ".join(f"{ms([float(r['wasted_sweep']) for r in data.get((m, q), {}).values()]):>20}"
                                     for m in methods))

print("\nConstraint violations: drones crashed (battery ran out away from the base), total over all runs")
print(f"{'comms':>5}  " + "  ".join(f"{NAMES[m]:>20}" for m in methods))
for q in qs:
    print(f"{q:5.1f}  " + "  ".join(f"{sum(int(r.get('crashes') or 0) for r in data.get((m, q), {}).values()):>20}" for m in methods))

print("\nPaired differences in rounds (row method minus column method; negative = row is faster), t in brackets")
pairs = [(a, b) for a, b in [("node", "htn"), ("node", "llm"), ("llm", "htn")] if a in methods and b in methods]
print(f"{'comms':>5}  " + "  ".join(f"{a + '-' + b:>18}" for a, b in pairs))
for q in qs:
    cells = []
    for a, b in pairs:
        A, B = data.get((a, q), {}), data.get((b, q), {})
        d = [float(A[s]["rounds"]) - float(B[s]["rounds"]) for s in A if s in B]
        if len(d) < 2:
            cells.append("-")
            continue
        se = st.stdev(d) / math.sqrt(len(d))
        cells.append(f"{st.mean(d):+.1f} ({st.mean(d) / se if se else 0:+.1f})")
    print(f"{q:5.1f}  " + "  ".join(f"{c:>18}" for c in cells))

print("\nPrior quality: rank of the person's area under the prior used (1 = searched first), and prior mass on it")
for m in methods:
    runs = [r for (mm, q), v in data.items() if mm == m for r in v.values()]
    print(f"  {NAMES[m]:>22}: rank {st.mean(float(r['person_prior_rank']) for r in runs):.2f}, "
          f"mass {st.mean(float(r['prior_mass_on_person']) for r in runs):.3f}, "
          f"LLM queries/run {st.mean(float(r['llm_queries']) for r in runs):.1f}, "
          f"failed calls {sum(int(r['llm_failures']) for r in runs)}")

cover = Counter()
for d in sys.argv[1:]:
    p = os.path.join(d, "log.jsonl")
    if not os.path.exists(p):
        continue
    for line in open(p):
        e = json.loads(line)
        if e.get("kind") != "cover":
            continue
        ans = e["answer"]
        if ans == "already_searched":
            ok = e["truth_searched"]
        elif ans == "teammate_heading_there":
            ok = bool(set(e["mates"]) & set(e["truth_heading"]))
        elif ans == "open":
            ok = not e["truth_searched"] and not e["truth_heading"]
        else:
            ok = None
        cover[(e["method"], e["q"], ans, ok)] += 1
if cover:
    print("\nCover answers (graded afterwards against the truth): method, comms -> answer: right/wrong")
    for (m, q) in sorted({(k[0], k[1]) for k in cover}):
        parts = []
        for ans in ["already_searched", "teammate_heading_there", "open", "unknown"]:
            r, w, u = cover[(m, q, ans, True)], cover[(m, q, ans, False)], cover[(m, q, ans, None)]
            if r + w + u:
                parts.append(f"{ans} {r}/{w}" if ans != "unknown" else f"unknown {u}")
        print(f"  {m:10s} {q:.1f}: " + ", ".join(parts))
