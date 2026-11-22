#!/bin/bash
# usage: run_genesis.sh <model>  -> "Genesis Agent": Claude Code + Genesis Agent Skills on ALCF Tara
m=$1; tag=genesis_${m//\//_}
here=$(cd "$(dirname "$0")" && pwd)
# Run OUTSIDE the git repo so Claude Code does not load the repo's CLAUDE.md or find the
# ground-truth script (first attempt did both; see invalid/). Results are copied back to runs/.
RUN_ROOT=${QICK_RUN_ROOT:?set QICK_RUN_ROOT to a directory outside the repo}
w=$RUN_ROOT/$tag
G=${GENESIS_CC_DIR:?set GENESIS_CC_DIR to an isolated Claude Code config dir with skills/}
rm -rf "$w"; mkdir -p "$w"
cp /home/hyoklee/aurora/data/qick/2025-10-19_12-46-02_ss_ge_results_batch_1_Num_per_batch1.h5 "$w/ss_ge.h5"
cp "$here/fermilab-pub-26-0193.txt" "$w/"
export CLAUDE_CONFIG_DIR=$G
export ANTHROPIC_BASE_URL=https://inference-api.alcf.anl.gov/resource_server/tara/api
export ANTHROPIC_AUTH_TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.globus/app/58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944/alcf_agentics_workflow/tokens.json'))['data']['DEFAULT']['681c10cc-f684-4540-bcd7-0b4df3bc26ef']['access_token'])")
unset ANTHROPIC_API_KEY
export ANTHROPIC_MODEL=$m ANTHROPIC_DEFAULT_OPUS_MODEL=$m ANTHROPIC_DEFAULT_SONNET_MODEL=$m \
       ANTHROPIC_DEFAULT_HAIKU_MODEL=$m ANTHROPIC_SMALL_FAST_MODEL=$m CLAUDE_CODE_SUBAGENT_MODEL=$m
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 DISABLE_AUTOUPDATER=1
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16000
cd "$w"; start=$(date +%s)
timeout 1800 claude -p "$(cat "$here/prompt.txt")" --model "$m" --permission-mode acceptEdits \
  --allowedTools "Read" "Write" "Edit" "Glob" "Grep" "Skill" "TodoWrite" "Bash(/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/venv/bin/python:*)" "Bash(ls:*)" "Bash(head:*)" "Bash(file:*)" "Bash(wc:*)" \
  --output-format stream-json --verbose --max-turns 80 < /dev/null > "$w/stream.jsonl" 2> "$w/stderr.log"
rc=$?
echo "{\"model\":\"$m\",\"rc\":$rc,\"wall_s\":$(( $(date +%s)-start ))}" > "$w/meta.json"; cat "$w/meta.json"
rm -rf "$here/runs/$tag"; cp -r "$w" "$here/runs/$tag"
