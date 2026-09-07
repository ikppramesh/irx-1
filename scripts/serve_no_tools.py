#!/usr/bin/env python3
"""
Local reverse proxy in front of mlx_lm.server that strips tool/function-calling
fields from every request before forwarding. Use this when a client (e.g. Bionic)
always sends tool definitions and doesn't give you a way to turn that off per
model — IRx-1's fine-tuning never reinforced native tool-calling, so it's
unreliable when tools are present (see the model card's Limitations section).
This guarantees it never sees them, regardless of client settings.

Usage:
    # terminal 1 — the real model server
    bash scripts/serve.sh

    # terminal 2 — this proxy in front of it
    python3 scripts/serve_no_tools.py

Then point your client at http://127.0.0.1:8766 instead of 8765.
"""
import http.server
import json
import os
import urllib.error
import urllib.request

UPSTREAM = os.environ.get("IRX1_UPSTREAM", "http://127.0.0.1:8765")
PORT = int(os.environ.get("IRX1_PROXY_PORT", "8766"))

STRIP_KEYS = ("tools", "tool_choice", "functions", "function_call")


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def _forward(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""

        if body:
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    removed = [k for k in STRIP_KEYS if k in payload]
                    for k in removed:
                        del payload[k]
                    if removed:
                        print(f"[proxy] stripped {removed} from request to {self.path}")
                    body = json.dumps(payload).encode()
            except json.JSONDecodeError:
                pass  # not JSON — forward as-is

        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        req = urllib.request.Request(
            UPSTREAM + self.path,
            data=body if body else None,
            method=self.command,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req) as resp:
                self._reply(resp.status, dict(resp.getheaders()), resp.read())
        except urllib.error.HTTPError as e:
            self._reply(e.code, dict(e.headers), e.read())

    def _reply(self, status, headers, body):
        self.send_response(status)
        for k, v in headers.items():
            if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self._forward()

    def do_GET(self):
        self._forward()

    def log_message(self, format, *args):
        pass  # the [proxy] print above is enough signal


if __name__ == "__main__":
    print(f"Proxy on http://127.0.0.1:{PORT} -> {UPSTREAM} (stripping tool/function fields)")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), ProxyHandler).serve_forever()
