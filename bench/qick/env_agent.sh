# source me: env for CLIO Agent on ALCF Tara. $1 = model
export ALCF_INFERENCE_TOKEN=$(python3 -c "import json;print(json.load(open('$HOME/.globus/app/58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944/alcf_agentics_workflow/tokens.json'))['data']['DEFAULT']['681c10cc-f684-4540-bcd7-0b4df3bc26ef']['access_token'])")
export CLIO_LM_PROVIDER=argonne_metis
export CLIO_LM_API_BASE=https://inference-api.alcf.anl.gov/resource_server/tara/api/v1
export CLIO_LM_MODEL=$1 CLIO_LM_CONTEXT_WINDOW=1000000 CLIO_LM_MAX_TOKENS=16384
export CLIO_ALLOWED_ROOTS="${QICK_RUN_ROOT:?}:/tmp"
export CLIO_NO_UPDATE_CHECK=1
