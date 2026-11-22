#!/bin/bash
# Run CLIO Agent for each model serially; the server's LM is fixed at start, so restart per model.
here=$(cd "$(dirname "$0")" && pwd)
for m in "$@"; do
  ( source "$here/env_agent.sh" "$m"; clio stop >/dev/null 2>&1; clio start >/dev/null 2>&1
    timeout 300 bash -c 'until curl --noproxy "*" -s -m 3 http://127.0.0.1:17800/v1/health | grep -q "clio_core_attach\",\"status\":\"ready"; do sleep 3; done'
    timeout 300 bash -c 'until grep -q "agent ready" ~/.local/share/clio/clio-server.$(hostname -s).log; do sleep 3; done'
    "$here/run_agent.sh" "$m" )
done
