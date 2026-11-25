"""Latency and token counts per LLM call, and planner time per decision, with one request in flight at a time.
Wraps sar_sim / sar_llm without changing them. Needs LLM_BASE_URL; writes timing.json."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sar_sim as S, sar_llm as L

recs = []
cur = {}

def kind_of(system):
    if system.startswith("You support"): return "likelihood"
    if system.startswith("You are the reasoning module"): return "cover"
    if system.startswith("You control one drone"): return "llm_only"
    return "other"

cli = L.client()
create = cli.cli.chat.completions.create
def timed(**kw):
    t0 = time.perf_counter(); r = create(**kw); dt = time.perf_counter() - t0
    recs.append(dict(type="llm", method=cur["m"], q=cur["q"], seed=cur["s"], kind=kind_of(kw["messages"][0]["content"]),
                     s=dt, prompt_tok=r.usage.prompt_tokens, out_tok=r.usage.completion_tokens))
    return r
cli.cli.chat.completions.create = timed

choose = S.choose_area
def timed_choose(*a, **k):
    t0 = time.perf_counter(); out = choose(*a, **k)
    recs.append(dict(type="plan", method=cur["m"], q=cur["q"], seed=cur["s"], s=time.perf_counter() - t0))
    return out
S.choose_area = timed_choose

for m in ["htn", "llm", "node"]:
    for q in [0.0, 0.4, 1.0]:
        for s in range(101, 106):
            cur.update(m=m, q=q, s=s)
            S.Sim(s, q, m).run()
            print(m, q, s, len(recs), flush=True)
json.dump(recs, open("timing.json", "w"))
print("done")
