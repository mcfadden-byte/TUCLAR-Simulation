"""Run the search-and-rescue comparison.

  python run_sar.py --methods htn,llm,node --comms 0,0.2,0.4,0.6,0.8,1 --seeds 1-30 --workers 24 --out runs/x
Each run appends one row to <out>/results.csv and its LLM answers to <out>/log.jsonl; runs already in
results.csv are skipped, so an interrupted batch can be restarted with the same command.
Environment: LLM_BASE_URL and MODEL for the LLM methods; SAR_MOCK_LLM=1 replaces the LLM with a stand-in.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sar_sim as S  # noqa: E402

FIELDS = ["method", "q", "seed", "profile", "found", "rounds", "person_true_rank", "person_prior_rank",
          "prior_mass_on_person", "sweep_rounds", "wasted_sweep", "distance", "areas_searched", "decisions",
          "llm_queries", "llm_seconds", "llm_failures", "crashes", "messages", "msg_bits", "wall_s"]


def seeds_arg(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def run_one(task):
    method, q, seed = task
    t0 = time.perf_counter()
    sim = S.Sim(seed, q, method)
    res = sim.run()
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    return res, sim.log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="htn")
    ap.add_argument("--comms", default="0,0.2,0.4,0.6,0.8,1")
    ap.add_argument("--seeds", default="1-30")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res_path, log_path = os.path.join(args.out, "results.csv"), os.path.join(args.out, "log.jsonl")
    done = set()
    if os.path.exists(res_path):
        with open(res_path) as f:
            done = {(r["method"], float(r["q"]), int(r["seed"])) for r in csv.DictReader(f)}
    tasks = [(m, float(q), s) for m in args.methods.split(",") for q in args.comms.split(",")
             for s in seeds_arg(args.seeds) if (m, float(q), s) not in done]
    print(f"{len(tasks)} runs to do ({len(done)} already done), {args.workers} workers", flush=True)
    new = not os.path.exists(res_path)
    t0 = time.time()
    with open(res_path, "a", newline="") as fr, open(log_path, "a") as fl, \
            mp.get_context("fork").Pool(args.workers, maxtasksperchild=20) as pool:
        w = csv.DictWriter(fr, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for k, (res, log) in enumerate(pool.imap_unordered(run_one, tasks), 1):
            w.writerow({f: res.get(f) for f in FIELDS})
            fr.flush()
            for e in log:
                e.update(method=res["method"], q=res["q"])
                fl.write(json.dumps(e) + "\n")
            fl.flush()
            if k % 20 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)} runs, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
