"""Compare node-level runs with baseline runs of main.py on the same seeds (commit 105940f, seeds 1-20).

usage: python3 compare_node.py <baseline out dir> <node out dir>
  baseline dirs: <block>_<nollm|full|llm>_c<q>/summary_log.csv   node dirs: node_<block>_c<q>/{summary_log.csv,nodes.log}
Columns are read by position (the CSV header of commit 105940f is misaligned at positions 27-32).
"""
import csv
import glob
import math
import os
import re
import statistics as st
import sys
from collections import Counter, defaultdict

DIST, SUCCESS, ROUNDS, QUERIES, SEED = 0, 2, 3, 41, 47
NAMES = {"nollm": "HTN-only", "llm": "LLM-only", "full": "teammate-level", "node": "node-level"}


def load(root, pattern):
    data = {}
    for path in glob.glob(os.path.join(root, pattern, "summary_log.csv")):
        name = os.path.basename(os.path.dirname(path))
        parts = name.split("_")
        if parts[0] == "node":
            block, method, q = parts[1], "node", float(parts[2][1:])
        else:
            block, method, q = parts[0], parts[1], float(parts[2][1:])
        with open(path, newline="") as f:
            rows = list(csv.reader(f))[1:]
        data[(block, method, q)] = {int(float(r[SEED])): r for r in rows}
    return data


def cell(runs):
    ok = [r for r in runs.values() if r[SUCCESS] == "True"]
    if not ok:
        return f"0/{len(runs)}"
    rd = [float(r[ROUNDS]) for r in ok]
    ci = 1.96 * st.stdev(rd) / math.sqrt(len(rd)) if len(rd) > 1 else 0
    s = f"{st.mean(rd):.1f}±{ci:.1f}"
    return s if len(ok) == len(runs) else f"{s} ({len(ok)}/{len(runs)})"


def paired(a, b):
    d = [float(a[s][ROUNDS]) - float(b[s][ROUNDS]) for s in a
         if s in b and a[s][SUCCESS] == "True" and b[s][SUCCESS] == "True"]
    if len(d) < 2:
        return "-"
    se = st.stdev(d) / math.sqrt(len(d))
    return f"{st.mean(d):+.1f} (t={st.mean(d) / se if se else 0:+.1f})"


base = load(sys.argv[1], "*_*_c*")
node = load(sys.argv[2], "node_*_c*")
data = {**{k: v for k, v in base.items() if k[1] != "node"}, **node}
for block in ["asis", "noobs"]:
    print(f"\n=== {block}: rounds (mean±95% CI over successful runs; success shown if <100%)")
    print(f"{'comms':>5}  " + "  ".join(f"{NAMES[m]:>18}" for m in ["nollm", "llm", "full", "node"])
          + f"  {'node-HTN paired':>18}  {'queries/run':>11}")
    for q in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        cells = [cell(data[(block, m, q)]) if (block, m, q) in data else "-" for m in ["nollm", "llm", "full", "node"]]
        nd = data.get((block, "node", q))
        pq = paired(nd, data[(block, "nollm", q)]) if nd else "-"
        qr = f"{st.mean(float(r[QUERIES]) for r in nd.values()):.1f}" if nd else "-"
        print(f"{q:5.1f}  " + "  ".join(f"{c:>18}" for c in cells) + f"  {pq:>18}  {qr:>11}")

print("\n=== node answers (graded against ground truth after the fact)")
for path in sorted(glob.glob(os.path.join(sys.argv[2], "node_*_c*", "nodes.log"))):
    name = os.path.basename(os.path.dirname(path))
    c = defaultdict(Counter)
    for line in open(path):
        m = re.search(r" (cover|type)_\d+ ans=(\S+).* ok=(\S+)", line)
        if m:
            c[m.group(1)][(m.group(2), m.group(3))] += 1
    parts = []
    for kind in ["cover", "type"]:
        tot = sum(c[kind].values())
        if not tot:
            continue
        unk = sum(v for (a, ok), v in c[kind].items() if a == "unknown")
        ans = {a: sum(v for (aa, ok), v in c[kind].items() if aa == a) for a, _ in c[kind]}
        right = sum(v for (a, ok), v in c[kind].items() if ok == "True")
        wrong = sum(v for (a, ok), v in c[kind].items() if ok == "False")
        acc = f"{right / (right + wrong):.2f}" if right + wrong else "-"
        parts.append(f"{kind}: n={tot} unknown={unk / tot:.2f} acc(answered)={acc} {dict(sorted(ans.items()))}")
    print(f"{name:16s} " + " | ".join(parts))
