#!/usr/bin/env python3
"""CLIO Stop hook: do not let the turn end before the deliverables exist.

Wire contract (clio-agent docs/HOOKS.md): the JSON envelope arrives on stdin;
exit 2 denies (re-drives the turn, bounded by loopLimit) and stderr is the
reason the model reads; exit 0 with empty stdout allows.

Why: on clio-agent 0.9.7.1 beam-1.0-fp8 wrote its plan as visible text
("... Let me write the test.") with no tool call; the loop takes a reply
without tool calls as the final answer and ended the turn with nothing made.
QICK try 5 (2026-11-27): beam made readout.png, then ended the turn with
"The hook seems stale. Let me verify." and no report. --answer-contains
checks the final message too.

usage: deliverables_stop_hook.py <run dir> [file ...] [--answer-contains TEXT]
Files default to the IceCube-7 deliverables.
"""
import json
import os
import sys

args = sys.argv[1:]
needle = None
if "--answer-contains" in args:
    i = args.index("--answer-contains")
    needle = args[i + 1]
    del args[i : i + 2]

try:
    envelope = json.load(sys.stdin)
except ValueError:
    envelope = {}
# The harness passes the run directory; fall back to the envelope cwd.
cwd = args[0] if args else envelope.get("cwd") or os.getcwd()
DELIVERABLES = tuple(args[1:]) or ("fig10.py", "fig10.png", "answer.md")
missing = [name for name in DELIVERABLES if not os.path.isfile(os.path.join(cwd, name))]


def answer_text(env: dict) -> str:
    """The visible text of the message that would end the turn."""
    assistant = (env.get("payload") or {}).get("assistant") or {}
    parts = assistant.get("parts") or []
    return "\n".join(
        p.get("text") or "" for p in parts if isinstance(p, dict) and p.get("type") == "text"
    )


reasons = []
if missing:
    reasons.append(
        ", ".join(missing) + " " + ("is" if len(missing) == 1 else "are")
        + f" not in the working directory {cwd}. Continue with tool calls "
        "(write and run the files) instead of describing the next step."
    )
if needle is not None and needle not in answer_text(envelope):
    reasons.append(
        "Your final message does not contain the report the task asks for (it must include "
        f"'{needle}'). Do not describe the next step: reply with the full report as your "
        "final message."
    )
if reasons:
    sys.stderr.write("The task is not finished: " + " ".join(reasons) + "\n")
    sys.exit(2)
sys.exit(0)
