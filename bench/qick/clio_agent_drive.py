"""Drive one CLIO Agent turn headlessly (like `clio-agent --query`), but
auto-ALLOW tool permissions and root the session in a given workspace.

usage: <clio-agent venv python> clio_agent_drive.py <workspace_dir> <prompt_file> <out.json>
"""
import json, sys, time
from rich.console import Console
from clio_agent.sdk.client import ClioClient
from clio_agent.ui.cli import ClioAgentCLI

ws_dir, prompt_file, out = sys.argv[1:4]
client = ClioClient("http://127.0.0.1:17800", timeout=120, stream_read_timeout=1800)
allowed = []

class AutoAllowCLI(ClioAgentCLI):
    def _resolve_permission(self, permission, *, interactive, status=None):
        allowed.append(permission.tool_call.tool_name)
        self.client.permissions.respond(permission.id, "allow")
        return None

cli = AutoAllowCLI(client, console=Console(stderr=True))
ws = client.workspaces.create(name="qick-" + str(int(time.time())), root_path=ws_dir)
cli._session_id = client.sessions.create(workspace_id=ws.id, title="qick").id
cli.MAX_PAUSE_ROUNDS = 400
t0 = time.time()
res = cli.ask_question(open(prompt_file).read(), interactive=False)
res["wall_s"] = round(time.time() - t0, 1)
res["permissions_allowed"] = allowed
res["workspace_id"] = ws.id
json.dump(res, open(out, "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("wall_s", "session_id")}), "tools:", len(allowed))
