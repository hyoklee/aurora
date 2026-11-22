#!/bin/bash
# usage: run_agent_golf.sh <model>
# CLIO Agent (ALCF Tara model) answers the golf question from data that
# `clio-cae golf_omni.yaml` put into clio-core. Runs outside the repo.
m=$1; tag=agent_${m//\//_}
here=$(cd "$(dirname "$0")" && pwd)
q=/home/hyoklee/aurora/bench/qick
: "${GOLF_SCRATCH:?set GOLF_SCRATCH to the scratch dir holding golf/ and venv/}"
export QICK_RUN_ROOT=${GOLF_RUN_ROOT:?set GOLF_RUN_ROOT outside the repo}
w=$QICK_RUN_ROOT/$tag
rm -rf "$w"; mkdir -p "$w"
cp "$here/golf_omni.yaml" "$here/cte_golf.py" "$here/prompt.txt" "$w/"
cp "$GOLF_SCRATCH/golf/clio_env.sh" "$w/clio_env.sh"
cp "$GOLF_SCRATCH/golf/paper/paper.txt" "$w/paper.txt"
( source "$q/env_agent.sh" "$m"
  export ALCF_INFERENCE_TOKEN=$(~/.local/share/clio/clio-agent/.venv/bin/python -c \
    "from clio_agent.providers import argonne_auth as a; print(a.get_access_token(allow_interactive=False))")
  clio stop >/dev/null 2>&1; clio start >/dev/null 2>&1
  timeout 300 bash -c 'until curl --noproxy "*" -s -m 3 http://127.0.0.1:17800/v1/health | grep -q "clio_core_attach\",\"status\":\"ready"; do sleep 3; done'
  timeout 300 bash -c 'until grep -q "agent ready" ~/.local/share/clio/clio-server.$(hostname -s).log; do sleep 3; done'
  export CLIO_LM_MAX_TOKENS=32768
  # Pre-flight: the golf runtime must still hold the ingested data now that
  # the clio server has restarted (the first run found it gone).
  n=$( cd "$w" && source ./clio_env.sh && CTP_LOG_LEVEL=error timeout 120 \
       "$GOLF_SCRATCH/venv/bin/python" -c "import cte_golf as cg; print(len(cg.tags('golf.*')))" 2>/dev/null | tail -1 )
  echo "preflight: $n golf tags" | tee "$w/preflight.txt"
  [ "$n" = 40486 ] || { echo '{"model":"'$m'","rc":99,"error":"preflight"}' > "$w/meta.json"; exit 99; }
  cd "$w"; start=$(date +%s)
  timeout 2700 ~/.local/share/clio/clio-agent/.venv/bin/python "$q/clio_agent_drive.py" "$w" "$w/prompt.txt" "$w/out.json" > "$w/stdout.log" 2> "$w/stderr.log"
  rc=$?
  echo "{\"model\":\"$m\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-start ))}" > "$w/meta.json"; cat "$w/meta.json" )
