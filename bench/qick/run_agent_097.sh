#!/bin/bash
# usage: run_agent_097.sh <model>
# CLIO Agent rerun of the QICK question on clio-agent 0.9.7.x from source
# (~/src/clio-agent, branch fix/model-shell-cte-pool-binding = iowarp/clio-agent#2402),
# same harness as bench/icecube7/run_agent_ice_097.sh: built-in unattended
# `clio-agent -q --workspace`, one agent home per run, Stop hook, answers locked.
# Same prompt and workspace contents as run_agent.sh (0.9.4, 2026-11-21).
m=$1; tag=agent097_${m//\//_}
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../.." && pwd)
: "${QICK_SCRATCH:?set QICK_SCRATCH to the scratch dir (holds venv/)}"
: "${QICK_RUN_ROOT:?set QICK_RUN_ROOT outside the repo}"
export CLIO_PREFIX=${CLIO_DEV_PREFIX:-$HOME/.local/share/clio-dev}
export CLIO_AGENT_HOME=${QICK_AGENT_HOME:?set QICK_AGENT_HOME (scratch; each home holds a 1.1 GB store)}/qick_$tag
V=$CLIO_PREFIX/clio-agent/.venv
w=$QICK_RUN_ROOT/$tag
# Earlier runs and homes go to an archive outside the run root. A locked (000)
# sibling is not enough: on 2026-11-27 beam listed the run root, found its own
# earlier try (left there as $w.old.*) and ran `chmod -R +rx` on it.
A=${QICK_ARCHIVE:-$QICK_SCRATCH/qick_archive}/$(date +%Y%m%d_%H%M%S)
for d in "$QICK_RUN_ROOT" "$QICK_AGENT_HOME"; do
  [ -d "$d" ] || continue
  for e in "$d"/* "$d"/.[!.]*; do
    [ -e "$e" ] || continue
    mkdir -p "$A/$(basename "$d")"; mv "$e" "$A/$(basename "$d")/"
  done
done
# The venv cannot be locked, because the prompt points the agent at its python.
# It must hold no earlier agent's scripts: a 2026-11-21 run left its analysis there,
# and on 2026-11-26 beam found it and re-ran it.
stray=$(find "$QICK_SCRATCH/venv" -maxdepth 1 -type f ! -name pyvenv.cfg ! -name CACHEDIR.TAG ! -name .gitignore ! -name .lock)
[ -z "$stray" ] || { echo "stray files in the venv: $stray" >&2; exit 98; }
mkdir -p "$w" "$CLIO_AGENT_HOME"
cp "$repo/data/qick/2025-10-19_12-46-02_ss_ge_results_batch_1_Num_per_batch1.h5" "$w/ss_ge.h5"
cp "$here/fermilab-pub-26-0193.txt" "$w/"
hook=$CLIO_AGENT_HOME/deliverables_stop_hook.py
cp "$here/../icecube7/deliverables_stop_hook.py" "$hook"
# Lock everything that could show the agent an answer while it works:
# - the repo's ground truth, write-ups and earlier answers;
# - older CLIO homes, which hold earlier QICK sessions;
# - every other scratch entry and run.
# In 2026-11-21 run 3, CLIO Agent + beam found src/qick/ss_fidelity.py.
# QICK_LOCK adds paths. Everything is restored on exit.
locked=()
for p in "$repo"/src/qick "$repo"/doc "$repo"/CLAUDE.md "$repo"/bench/icecube7 \
         "$here"/{runs,answers,invalid,results.json,plots_grid.png,README.md} \
         "$HOME"/.local/share/clio/gact "$CLIO_PREFIX"/home* \
         "$QICK_SCRATCH"/* "$QICK_RUN_ROOT"/*/ "$QICK_AGENT_HOME"/*/ $QICK_LOCK; do
  p=${p%/}
  case $p in "$w"|"$CLIO_AGENT_HOME"|"$QICK_SCRATCH/venv"|"$QICK_RUN_ROOT"|"$QICK_AGENT_HOME") continue;; esac
  [ -e "$p" ] && [ "$(stat -c %a "$p")" != 0 ] || continue
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
  export CLIO_ALLOWED_ROOTS="$QICK_RUN_ROOT:/tmp" CLIO_NO_UPDATE_CHECK=1
  export CLIO_SHELL_MAX_TIMEOUT_S=${CLIO_SHELL_MAX_TIMEOUT_S:-120}
  case $m in nemotron*) export CLIO_LM_MAX_REQUEST_IMAGES=${CLIO_LM_MAX_REQUEST_IMAGES:-1};; esac
  # beam is text-only on ALCF. One view_image led to HTTP 400 "not a multimodal model",
  # returned as the answer (2026-11-26). Needs clio-agent 5624f32e3.
  case $m in beam*) export CLIO_LM_ACCEPTS_IMAGES=${CLIO_LM_ACCEPTS_IMAGES:-false};; esac
  # The Stop hook re-drives a turn (at most 3 times) while readout.png is missing
  # or the final message has no report (no "VERDICT").
  # The hook script was copied into the agent home before bench/icecube7 was locked.
  if [ "${QICK_STOP_HOOK:-1}" = 1 ]; then
    cat > "$CLIO_AGENT_HOME/qick_hooks.json" <<EOF
{"hooks": [{"id": "qick-deliverables", "on": ["Stop"],
  "run": {"type": "command", "command": "/usr/bin/python3",
          "args": ["$hook", "$w", "readout.png", "--answer-contains", "VERDICT"]},
  "timeoutMs": 10000, "loopLimit": 3}]}
EOF
    export CLIO_HOOKS_CONFIG=$CLIO_AGENT_HOME/qick_hooks.json
  fi
  "$CLIO_PREFIX/clio" stop >/dev/null 2>&1; "$CLIO_PREFIX/clio" start > "$w/server_start.log" 2>&1
  timeout 300 bash -c 'until curl --noproxy "*" -s -m 3 http://127.0.0.1:17800/v1/health | grep -q "\"healthy\":true"; do sleep 3; done'
  cd "$w"; start=$(date +%s)
  timeout "${QICK_TIMEOUT:-4500}" "$V/bin/clio-agent" -q "$(cat "$here/prompt.txt")" --json --workspace "$w" --port 17800 \
    > "$w/out.json" 2> "$w/stderr.log"
  rc=$?
  "$CLIO_PREFIX/clio" stop >/dev/null 2>&1
  # Agents also write scripts at the top of /tmp. Move what this run made into the
  # run dir, so a later run cannot find it (2026-11-26: beam re-ran a 2026-11-21
  # run's /tmp/readout_plot.py).
  mkdir -p "$w/tmp_leftovers"
  find /tmp -mindepth 1 -maxdepth 1 -user "$USER" -newer "$w/server_start.log" \
    \( -type f \( -name '*.py' -o -name '*.png' -o -name '*.txt' -o -name '*.md' \
                 -o -name '*.json' -o -name '*.npy' -o -name '*.csv' \) \
       -o -type d \( -name scratchpad -o -name 'qick*' -o -name 'readout*' \) \) \
    -exec mv -t "$w/tmp_leftovers" {} + 2>/dev/null
  # The same goes for the venv and the run root, the scratch paths left open.
  find "$QICK_SCRATCH/venv" "$QICK_RUN_ROOT" -mindepth 1 -maxdepth 1 -newer "$w/server_start.log" \
    ! -path "$w" ! -name bin ! -name lib ! -name lib64 ! -name share \
    -exec mv -t "$w/tmp_leftovers" {} + 2>/dev/null
  echo "{\"model\":\"$m\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-start ))}" > "$w/meta.json"; cat "$w/meta.json" )
