#!/usr/bin/env bash
# Runs run_sar.py against an OpenAI-compatible vLLM server serving gemma-4-26b-a4b-it (HTN-only needs no server):
#   LLM_BASE_URL=http://<host>:<port>/v1 bash run_sar.sh <run_sar.py arguments>
export PYTHONUNBUFFERED=1
export LLM_BASE_URL=${LLM_BASE_URL:-http://127.0.0.1:8000/v1} MODEL=${MODEL:-gemma-4-26b-a4b-it}
python "$(dirname "$0")/run_sar.py" "$@"
