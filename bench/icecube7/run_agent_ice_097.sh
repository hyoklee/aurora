#!/bin/bash
# usage: run_agent_ice_097.sh <model>
# Same task as run_agent_ice.sh, on clio-agent 0.9.7.x installed from source
# (~/src/clio-agent, editable) under its own prefix and home, so it never
# touches the 0.9.4 install in ~/.local/share/clio. Uses the built-in
# unattended `clio-agent -q --workspace` (grants every permission), so no
# custom driver is needed.
m=$1; tag=agent097_${m//\//_}
here=$(cd "$(dirname "$0")" && pwd)
: "${ICE_SCRATCH:?set ICE_SCRATCH to the scratch dir holding ice/ and venv/}"
: "${ICE_RUN_ROOT:?set ICE_RUN_ROOT outside the repo}"
export CLIO_PREFIX=${CLIO_DEV_PREFIX:-$HOME/.local/share/clio-dev}
# One agent home per run: a Stop-hook re-drive still queued when this run's
# server stops can then never resume under the next model (seen 2026-11-25).
export CLIO_AGENT_HOME=${ICE_AGENT_HOME:-$CLIO_PREFIX/home}/$tag
V=$CLIO_PREFIX/clio-agent/.venv
w=$ICE_RUN_ROOT/$tag
[ -e "$w" ] && mv "$w" "$w.old.$(date +%s)"
mkdir -p "$w" "$CLIO_AGENT_HOME"
cp "$here/ice_omni.yaml" "$here/cte_ice.py" "$here/prompt.txt" "$w/"
cp "$ICE_SCRATCH/ice/clio_env.sh" "$w/clio_env.sh"
cp "$ICE_SCRATCH/ice/paper/paper.txt" "$w/paper.txt"
# Nothing but this run's directory may show the agent an answer: lock the
# reference results, the paper PDF, raw downloads and the other runs while it
# works (2026-11-25: mistral-large-4 walked up from its workspace into the
# scratch tree and read the reference grid and the paper's Figure 10).
# ICE_LOCK adds paths; everything is restored on exit.
locked=()
for p in "$ICE_SCRATCH"/ice/{ref,paper,dl,dvcache,cmp,cmp097*,verify,x,refvenv} "$ICE_RUN_ROOT"/*/ $ICE_LOCK; do
  p=${p%/}; [ -e "$p" ] && [ "$p" != "$w" ] && [ "$(stat -c %a "$p")" != 0 ] || continue
  locked+=("$p:$(stat -c %a "$p")"); chmod 000 "$p"
done
unlock() { for e in "${locked[@]}"; do chmod "${e##*:}" "${e%:*}"; done; }
trap unlock EXIT
(
  export ALCF_INFERENCE_TOKEN=$("$V/bin/python" -c \
    "from clio_agent.providers import argonne_auth as a; print(a.get_access_token(allow_interactive=False))")
  export CLIO_LM_PROVIDER=argonne_metis
  export CLIO_LM_API_BASE=https://inference-api.alcf.anl.gov/resource_server/tara/api/v1
  export CLIO_LM_MODEL=$m CLIO_LM_CONTEXT_WINDOW=1000000 CLIO_LM_MAX_TOKENS=32768
  export CLIO_ALLOWED_ROOTS="$ICE_RUN_ROOT:/tmp" CLIO_NO_UPDATE_CHECK=1
  # The shell tool has no timeout by default. On Aurora a model's
  # `find / -name latex` walks Lustre/DAOS and ran 44 min (beam, 0.9.7.1).
  # 600 s still let beam spend 30 min in three `find /` calls; 120 s is ample
  # for every command this task needs (the slowest, reading clio-core, takes ~20 s).
  export CLIO_SHELL_MAX_TIMEOUT_S=${CLIO_SHELL_MAX_TIMEOUT_S:-120}
  # The ALCF endpoint for nemotron accepts one image per prompt; a second
  # view_image made every later request fail (HTTP 400, returned as the answer).
  case $m in nemotron*) export CLIO_LM_MAX_REQUEST_IMAGES=${CLIO_LM_MAX_REQUEST_IMAGES:-1};; esac
  # A Stop hook re-drives a turn (at most 3 times) while fig10.py / fig10.png /
  # answer.md are missing: beam-1.0-fp8 often ends a turn with a plan in prose
  # ("Let me look at the paper text around Figure 10.") and no tool call.
  # ICE_STOP_HOOK=0 turns it off. Needs clio-agent e823d8cad: before it,
  # `clio-agent -q` returned on the blocked turn, `clio stop` below cancelled
  # the queued one (or, with a shared home, it resumed under the NEXT model).
  if [ "${ICE_STOP_HOOK:-1}" = 1 ]; then
    cat > "$CLIO_AGENT_HOME/ice_hooks.json" <<EOF
{"hooks": [{"id": "ice-deliverables", "on": ["Stop"],
  "run": {"type": "command", "command": "/usr/bin/python3", "args": ["$here/deliverables_stop_hook.py", "$w"]},
  "timeoutMs": 10000, "loopLimit": 3}]}
EOF
    export CLIO_HOOKS_CONFIG=$CLIO_AGENT_HOME/ice_hooks.json
  fi
  "$CLIO_PREFIX/clio" stop >/dev/null 2>&1; "$CLIO_PREFIX/clio" start > "$w/server_start.log" 2>&1
  timeout 300 bash -c 'until curl --noproxy "*" -s -m 3 http://127.0.0.1:17800/v1/health | grep -q "\"healthy\":true"; do sleep 3; done'
  # Pre-flight: the ice runtime must still hold the ingested release.
  n=$( cd "$w" && source ./clio_env.sh && CTP_LOG_LEVEL=error timeout 120 \
       "$ICE_SCRATCH/venv/bin/python" -c "import cte_ice as ci; print(len(ci.tags('ice.*')))" 2>/dev/null | tail -1 )
  echo "preflight: $n ice tags" | tee "$w/preflight.txt"
  [ "$n" = 9 ] || { echo '{"model":"'$m'","rc":99,"error":"preflight"}' > "$w/meta.json"; exit 99; }
  cd "$w"; start=$(date +%s)
  # The ALCF gateway can take 1-2 min per call (mistral-large-4 was still
  # checking its figure at 45 min), so the cap is 75 min.
  timeout "${ICE_TIMEOUT:-4500}" "$V/bin/clio-agent" -q "$(cat prompt.txt)" --json --workspace "$w" --port 17800 \
    > "$w/out.json" 2> "$w/stderr.log"
  rc=$?
  "$CLIO_PREFIX/clio" stop >/dev/null 2>&1  # a timed-out turn keeps running in the server
  echo "{\"model\":\"$m\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-start ))}" > "$w/meta.json"; cat "$w/meta.json" )
