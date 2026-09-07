#!/usr/bin/env python3
"""
Local reverse proxy in front of mlx_lm.server that sanitizes every request
before forwarding, for clients (e.g. Bionic) whose built-in agentic/coding
scaffolding (file access, Python execution, workspace search) can send IRx-1
into a runaway repetition loop it has no training to escape from. IRx-1 is a
small fine-tuned chat model, not an agentic assistant — see the model card's
Limitations section.

Three things, in order:
1. Strip tools/tool_choice/functions/function_call — the native OpenAI
   function-calling fields.
2. Replace any "system" role message with a clean, known-good system prompt —
   many clients inject their tool/agent scaffolding as system-message text
   instead of (or in addition to) the structured "tools" field, which #1
   can't touch. Logs the length of what it replaced so you can confirm this
   was actually the cause.
3. Cap max_tokens at a safe ceiling — a backstop in case something still
   loops, so it can't run away indefinitely regardless of what the client
   requested.

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
import sys
import urllib.error
import urllib.request

sys.stdout.reconfigure(line_buffering=True)  # otherwise [proxy] logs sit in a buffer

UPSTREAM = os.environ.get("IRX1_UPSTREAM", "http://127.0.0.1:8765")
PORT = int(os.environ.get("IRX1_PROXY_PORT", "8766"))
MAX_TOKENS_CEILING = int(os.environ.get("IRX1_MAX_TOKENS_CEILING", "400"))

STRIP_KEYS = ("tools", "tool_choice", "functions", "function_call")

SAFE_SYSTEM_PROMPT = (
    "You are a helpful, direct offline assistant. Respond in plain text only. "
    "Do not attempt to call functions or tools, access files, run code, or "
    "reference a workspace — you don't have any of those; just answer the "
    "question directly. Do not show your reasoning, planning, or a "
    "step-by-step thinking process."
)


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

                    messages = payload.get("messages")
                    if isinstance(messages, list):
                        had_system = False
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("role") == "system":
                                had_system = True
                                original = msg.get("content", "")
                                if original != SAFE_SYSTEM_PROMPT:
                                    original_len = len(original) if isinstance(original, str) else 0
                                    print(f"[proxy] replaced system message "
                                          f"({original_len} chars -> clean prompt)")
                                msg["content"] = SAFE_SYSTEM_PROMPT
                        if not had_system:
                            print("[proxy] injected clean system prompt (client sent none)")
                            messages.insert(0, {"role": "system", "content": SAFE_SYSTEM_PROMPT})

                    requested = payload.get("max_tokens")
                    if not isinstance(requested, int) or requested > MAX_TOKENS_CEILING:
                        if requested is not None:
                            print(f"[proxy] capped max_tokens {requested} -> {MAX_TOKENS_CEILING}")
                        payload["max_tokens"] = MAX_TOKENS_CEILING

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
