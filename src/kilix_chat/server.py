"""A minimal OpenAI-compatible endpoint over one model, for coding agents.

It answers GET /health, GET /v1/models and POST /v1/chat/completions
(streamed or not). Requests are served one at a time: each diffusion reply
is a full llama-diffusion-cli run on the one GPU.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import uuid

from kilix_chat.diffusion import Cancelled


def completion(alias, reply, calls):
    message = {"role": "assistant", "content": reply.text if reply.text or not calls else None}
    if calls:
        message["tool_calls"] = calls
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
            "model": alias, "choices": [{"index": 0, "message": message,
                                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": usage(reply)}


def usage(reply):
    return {"prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens}


def chunks(alias, reply, calls):
    """The same completion as stream chunks: role, content or tool calls, finish."""
    base = {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion.chunk",
            "created": int(time.time()), "model": alias}
    def chunk(delta, finish=None, **extra):
        return dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}], **extra)
    yield chunk({"role": "assistant", "content": ""})
    if reply.text:
        yield chunk({"content": reply.text})
    if calls:
        yield chunk({"tool_calls": [dict(call, index=i) for i, call in enumerate(calls)]})
    yield chunk({}, "tool_calls" if calls else "stop", usage=usage(reply))


def make_handler(model, alias, max_tokens_cap, log):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if log:
                log(fmt % args)

        def send_json(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [
                    {"id": alias, "object": "model", "owned_by": "kilix-chat"}]})
            else:
                self.send_json(404, {"error": {"message": "Not found."}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": {"message": "Not found."}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                messages = body["messages"]
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
            except (ValueError, KeyError, TypeError) as error:
                self.send_json(400, {"error": {"message": f"Bad request ({error})."}})
                return
            requested = body.get("max_completion_tokens") or body.get("max_tokens") or max_tokens_cap
            budget = max(16, min(int(requested), max_tokens_cap))
            try:
                with lock:
                    reply, calls = model.reply(messages, tools=body.get("tools"), max_tokens=budget)
            except (RuntimeError, ValueError, OSError, Cancelled) as error:
                self.send_json(500, {"error": {"message": str(error)}})
                return
            if not body.get("stream"):
                self.send_json(200, completion(alias, reply, calls))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for part in chunks(alias, reply, calls):
                self.wfile.write(b"data: " + json.dumps(part).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return Handler


def serve(model, *, alias, host="127.0.0.1", port=8765, max_tokens_cap=512, log=None, ready=None):
    server = ThreadingHTTPServer((host, port), make_handler(model, alias, max_tokens_cap, log))
    server.daemon_threads = True
    if ready:
        ready(server)
    try:
        server.serve_forever()
    finally:
        server.server_close()
