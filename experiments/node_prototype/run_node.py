"""Run TUCLAR-Simulation (main.py) with the node-level hybrid planner. The repository is not edited.

  METHOD=node    node-level LLM reasoning (node_planner.py)
  METHOD=nollm   the same code with the LLM switched off (equals the messages-only ablation, i.e. HTN-only)
  NO_OBS=1       drones cannot observe teammates' targets by line of sight
All other arguments go to main.py, e.g. --planner hybrid --htn-backend hierarchical --comms-quality 0 --seed-file s.txt
"""
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
repo = os.environ["TUCLAR_REPO"]
os.chdir(repo)                      # osmnx reads the map from ./cache
sys.path.insert(0, repo)
sys.path.insert(1, here)

import agent            # noqa: E402
import node_planner     # noqa: E402
from environment import TargetType  # noqa: E402

method = os.environ.get("METHOD", "node")
agent.HYBRID_USE_LLM = False
agent.HYBRID_PRINT_BELIEFS = False
node_planner.NODE_USE_LLM = method == "node"
agent.HybridPlanner = node_planner.NodeHybridPlanner       # Drone.__init__ builds one per drone

_orig_schema = agent.build_belief_schema
agent.build_belief_schema = lambda x: x if isinstance(x, dict) else _orig_schema(x)   # node queries pass their schema

if os.environ.get("NO_OBS") == "1":
    def _no_observation(self, agent_index):
        return {idx: "blocked" for idx, mate in enumerate(self.agents) if idx != agent_index and not mate.inaction}
    agent.AgentHandler._observe_teammate_targets = _no_observation


def _is_surplus(self, ag, target):
    """AgentHandler._is_surplus, with a node-level type belief standing in for an unknown type."""
    me = ag.agent_index
    if target.target_type == TargetType.SEQUENTIAL and me in target.visitors:
        return True
    hp = ag.hybrid
    if agent.type_is_known(target, me) or hp.type_learned(ag.target_index):
        slots = target.required_drones
    else:
        slots = hp.believed_slots(ag.target_index) or agent.UNKNOWN_SLOT_GUESS
    views = hp.fresh_claims()
    rivals = [me] + [i for i, t in views.items() if t == ag.target_index and not self.agents[i].inaction]
    if len(rivals) <= slots:
        return False
    rivals.sort()
    return me not in rivals[:slots]


agent.AgentHandler._is_surplus = _is_surplus

_orig_score = agent.AgentHandler._score_beliefs


def _score_beliefs(self, agent_index):
    """Grade each node answer against the ground truth (logging only; the drone never sees this)."""
    _orig_score(self, agent_index)
    hp = self.agents[agent_index].hybrid
    targets = self.environment.target_locations
    for node, ans, mates in hp.last_nodes:
        t = int(node.split("_")[1])
        if node.startswith("cover_"):
            truth = sorted(j for j, a in enumerate(self.agents)
                           if j != agent_index and not a.inaction and a.last_target == t)
            ok = "unknown" if ans == "unknown" else str((ans == "covered") == bool(truth))
            print(f"[NODE] r={self.num_rounds} d={agent_index} {node} ans={ans} mates={mates} truth={truth} ok={ok}")
        else:
            truth = targets[t].target_type.value
            ok = "unknown" if ans == "unknown" else str(ans == truth)
            print(f"[NODE] r={self.num_rounds} d={agent_index} {node} ans={ans} truth={truth} ok={ok}")
    if hp.last_nodes and os.environ.get("PRINT_NODE_PROMPTS") == "1":
        print(f"[NODE-PROMPT] d={agent_index}\n{getattr(hp, 'last_prompt', '')}\n[NODE-REPLY] {getattr(hp, 'last_reply', '')}")


agent.AgentHandler._score_beliefs = _score_beliefs

if os.environ.get("PRINT_NODE_PROMPTS") == "1":     # keep the latest prompt and reply on the planner for printing
    _orig_prompt = node_planner.NodeHybridPlanner._node_prompt
    _orig_apply = node_planner.NodeHybridPlanner._apply

    def _node_prompt(self, *a, **k):
        self.last_prompt = _orig_prompt(self, *a, **k)
        return self.last_prompt

    def _apply(self, reply, *a, **k):
        self.last_reply = reply
        return _orig_apply(self, reply, *a, **k)

    node_planner.NodeHybridPlanner._node_prompt = _node_prompt
    node_planner.NodeHybridPlanner._apply = _apply

print(f"[run_node] METHOD={method} NO_OBS={os.environ.get('NO_OBS', '0')} model={agent.LLM_MODEL} "
      f"url={agent.LLM_BASE_URL} log_dir={agent.LOG_DIR}", flush=True)
sys.argv = [os.path.join(repo, "main.py")] + sys.argv[1:]
runpy.run_path(os.path.join(repo, "main.py"), run_name="__main__")
