#!/usr/bin/env python3
"""Collect final answers + accounting from all runs into results.json / answers/*.md."""
import json, glob, os
out = {}
os.makedirs("answers", exist_ok=True)
for d in sorted(glob.glob("runs/*")):
    tag = os.path.basename(d); r = {"tag": tag}
    try: r.update(json.load(open(f"{d}/meta.json")))
    except Exception: r["meta"] = "missing"
    text = ""
    if tag.startswith("coder_") and os.path.exists(f"{d}/events.jsonl"):
        for l in open(f"{d}/events.jsonl"):
            e = json.loads(l)
            if e["type"] == "agent_end": r["usage"] = e["usage"]
            if e["type"] == "turn_end": text = e.get("text", "")
    elif tag.startswith("genesis_") and os.path.exists(f"{d}/stream.jsonl"):
        skills, tools = [], 0
        for l in open(f"{d}/stream.jsonl"):
            try: e = json.loads(l)
            except Exception: continue
            if e.get("type") == "assistant":
                for c in e["message"].get("content", []):
                    if c.get("type") == "tool_use":
                        tools += 1
                        if c["name"] == "Skill": skills.append(c["input"].get("skill"))
            if e.get("type") == "result":
                text = e.get("result", ""); r["usage"] = e.get("usage"); r["num_turns"] = e.get("num_turns")
                r["is_error"] = e.get("is_error"); r["terminal_reason"] = e.get("terminal_reason")
        r["tool_calls"] = tools; r["skills_invoked"] = skills
    elif tag.startswith("agent_") and os.path.exists(f"{d}/out.json"):
        o = json.load(open(f"{d}/out.json")); text = o.get("answer", "") or json.dumps(o.get("error_info"))
        r["tool_calls"] = len(o.get("permissions_allowed", [])); r["tools"] = sorted(set(o.get("permissions_allowed", [])))
    r["has_plot"] = os.path.exists(f"{d}/readout.png")
    open(f"answers/{tag}.md", "w").write(text)
    out[tag] = r
json.dump(out, open("results.json", "w"), indent=1)
for k, v in out.items(): print(k, {x: v.get(x) for x in ("rc", "wall_s", "has_plot", "tool_calls", "skills_invoked")})
