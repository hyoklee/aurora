#!/usr/bin/env python3
"""CLIO Stop hook: do not let the turn end before the three deliverables exist.

Wire contract (clio-agent docs/HOOKS.md): the JSON envelope arrives on stdin;
exit 2 denies (re-drives the turn, bounded by loopLimit) and stderr is the
reason the model reads; exit 0 with empty stdout allows.

Why: on clio-agent 0.9.7.1 beam-1.0-fp8 wrote its plan as visible text
("... Let me write the test.") with no tool call; the loop takes a reply
without tool calls as the final answer and ended the turn with nothing made.
"""
import json
import os
import sys

# usage: deliverables_stop_hook.py <run dir> [file ...]; the IceCube-7 files by default.
DELIVERABLES = tuple(sys.argv[2:]) or ("fig10.py", "fig10.png", "answer.md")

try:
    envelope = json.load(sys.stdin)
except ValueError:
    envelope = {}
# The harness passes the run directory; fall back to the envelope cwd.
cwd = sys.argv[1] if len(sys.argv) > 1 else envelope.get("cwd") or os.getcwd()
missing = [name for name in DELIVERABLES if not os.path.isfile(os.path.join(cwd, name))]
if missing:
    sys.stderr.write(
        "The task is not finished: " + ", ".join(missing) + " "
        + ("is" if len(missing) == 1 else "are")
        + f" not in the working directory {cwd}. Continue with tool calls "
        "(write and run the files) instead of describing the next step.\n"
    )
    sys.exit(2)
sys.exit(0)
