#!/bin/bash
# usage: run_agent.sh <model>  -> clio-agent --query (CLIO Agent + clio-core) in a fresh workspace
m=$1; tag=agent_${m//\//_}
here=$(cd "$(dirname "$0")" && pwd)
# Run outside the git repo (no parent CLAUDE.md / ground-truth script); copied back to runs/.
w=${QICK_RUN_ROOT:?set QICK_RUN_ROOT to a directory outside the repo}/$tag
rm -rf "$w"; mkdir -p "$w"
cp /home/hyoklee/aurora/data/qick/2025-10-19_12-46-02_ss_ge_results_batch_1_Num_per_batch1.h5 "$w/ss_ge.h5"
cp "$here/fermilab-pub-26-0193.txt" "$w/"
source "$here/env_agent.sh" "$m"
cd "$w"; start=$(date +%s)
timeout 1800 ~/.local/share/clio/clio-agent/.venv/bin/python "$here/clio_agent_drive.py" "$w" "$here/prompt.txt" "$w/out.json" > "$w/stdout.log" 2> "$w/stderr.log"
rc=$?
echo "{\"model\":\"$m\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-start ))}" > "$w/meta.json"; cat "$w/meta.json"
rm -rf "$here/runs/$tag"; cp -r "$w" "$here/runs/$tag"
