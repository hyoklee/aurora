#!/bin/bash
# usage: run_agent_ice.sh <model>
# CLIO Agent (ALCF Tara model) regenerates Figure 10 of arXiv:1909.08623 from
# the data release that `clio-cae ice_omni.yaml` put into clio-core. Runs
# outside the repo.
m=$1; tag=agent_${m//\//_}
here=$(cd "$(dirname "$0")" && pwd)
q=/home/hyoklee/aurora/bench/qick
: "${ICE_SCRATCH:?set ICE_SCRATCH to the scratch dir holding ice/ and venv/}"
export QICK_RUN_ROOT=${ICE_RUN_ROOT:?set ICE_RUN_ROOT outside the repo}
w=$QICK_RUN_ROOT/$tag
[ -e "$w" ] && mv "$w" "$w.old.$(date +%s)"
mkdir -p "$w"
cp "$here/ice_omni.yaml" "$here/cte_ice.py" "$here/prompt.txt" "$w/"
cp "$ICE_SCRATCH/ice/clio_env.sh" "$w/clio_env.sh"
cp "$ICE_SCRATCH/ice/paper/paper.txt" "$w/paper.txt"
( source "$q/env_agent.sh" "$m"
  export ALCF_INFERENCE_TOKEN=$(~/.local/share/clio/clio-agent/.venv/bin/python -c \
    "from clio_agent.providers import argonne_auth as a; print(a.get_access_token(allow_interactive=False))")
  clio stop >/dev/null 2>&1; clio start >/dev/null 2>&1
  timeout 300 bash -c 'until curl --noproxy "*" -s -m 3 http://127.0.0.1:17800/v1/health | grep -q "clio_core_attach\",\"status\":\"ready"; do sleep 3; done'
  timeout 300 bash -c 'until grep -q "agent ready" ~/.local/share/clio/clio-server.$(hostname -s).log; do sleep 3; done'
  export CLIO_LM_MAX_TOKENS=32768
  # Pre-flight: the ice runtime must still hold the ingested release.
  n=$( cd "$w" && source ./clio_env.sh && CTP_LOG_LEVEL=error timeout 120 \
       "$ICE_SCRATCH/venv/bin/python" -c "import cte_ice as ci; print(len(ci.tags('ice.*')))" 2>/dev/null | tail -1 )
  echo "preflight: $n ice tags" | tee "$w/preflight.txt"
  [ "$n" = 9 ] || { echo '{"model":"'$m'","rc":99,"error":"preflight"}' > "$w/meta.json"; exit 99; }
  cd "$w"; start=$(date +%s)
  timeout 2700 ~/.local/share/clio/clio-agent/.venv/bin/python "$q/clio_agent_drive.py" "$w" "$w/prompt.txt" "$w/out.json" > "$w/stdout.log" 2> "$w/stderr.log"
  rc=$?
  echo "{\"model\":\"$m\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-start ))}" > "$w/meta.json"; cat "$w/meta.json" )
