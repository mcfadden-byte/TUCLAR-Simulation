#!/usr/bin/env bash
# Runs conditions of the node-level prototype against an OpenAI-compatible vLLM server (gemma-4-26b-a4b-it):
#   LLM_BASE_URL=http://<host>:<port>/v1 bash run_node.sh <conditions file> <conditions at a time> <runs at a time>
# Each line of the conditions file: <name> <node|nollm> <no_obs 0|1> <comms quality> <seed file> <print prompts 0|1>
# Results go to $OUT_DIR/out/<name> (default ./node_runs). Seed files are relative to this directory.
P=$(cd "$(dirname "$0")" && pwd)
D=${OUT_DIR:-$PWD/node_runs}
export PYTHONUNBUFFERED=1
export TUCLAR_REPO=${TUCLAR_REPO:-$(cd "$P/../.." && pwd)}
export LLM_BASE_URL=${LLM_BASE_URL:-http://127.0.0.1:8000/v1} MODEL=${MODEL:-gemma-4-26b-a4b-it} P D RUNS_PAR=$3
run_cond() {
  local out=$D/out/$1 seeds=$5
  [[ $seeds = /* ]] || seeds=$P/$seeds
  [ -f $out/DONE ] && return 0
  mkdir -p $out
  METHOD=$2 NO_OBS=$3 PRINT_NODE_PROMPTS=$6 LOG_DIR=$out python $P/run_node.py --planner hybrid \
    --htn-backend hierarchical --comms-quality $4 --seed-file $seeds --parallel $RUNS_PAR > $out/batch.log 2>&1
  local rc=$?
  grep -h "^\[NODE\]" $out/*/run_*/output.txt > $out/nodes.log 2>/dev/null
  [ $rc = 0 ] && touch $out/DONE
  echo "$(date '+%H:%M:%S') end $1 rc=$rc"
}
export -f run_cond
xargs -P $2 -L 1 bash -c 'run_cond "$@"' _ < $1
echo ALLDONE
