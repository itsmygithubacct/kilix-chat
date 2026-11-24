"""A minimal endpoint over one model, for coding agents and chat clients.

It speaks two dialects on one loopback port:

- OpenAI: GET /health, GET /v1/models and POST /v1/chat/completions,
  streamed or not.
- Ollama: GET /api/tags, GET /api/version, POST /api/chat (NDJSON stream by
  default, as Ollama does) and POST /api/pull, which succeeds without
  downloading for the served model and fails for any other name.

Requests are served one at a time: each diffusion reply is a full
llama-diffusion-cli run on the one GPU.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import socket
import threading
import time
import uuid

from kilix_chat import __version__
from kilix_chat.diffusion import Cancelled

# A diffusion reply has no text until it is finished. While it denoises, an
# Ollama stream carries an empty message this often, so a client's read
# timeout (Kilix Avatar's is 120 s) does not end the reply.
KEEPALIVE_SECONDS = 5.0


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


def is_loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def served_names(alias):
    """Names an Ollama client may use for the served model."""
    return {alias} if ":" in alias else {alias, alias + ":latest"}


def unknown_model(name, alias):
    return f'model "{name}" not found; this kilix-chat serves only "{alias}" and does not download models'


def model_record(model, alias):
    """The served model as one /api/tags entry."""
    path = getattr(model.backend, "path", None)
    try:
        info = path.stat() if path else None
    except OSError:
        info = None
    size = info.st_size if info else 0
    stamp = info.st_mtime if info else time.time()
    identity = f"{path or model.spec}\0{size}\0{info.st_mtime_ns if info else 0}"
    return {"name": alias, "model": alias, "size": size,
            "modified_at": datetime.fromtimestamp(stamp, timezone.utc).isoformat().replace("+00:00", "Z"),
            "digest": hashlib.sha256(identity.encode()).hexdigest(),
            "details": {"parent_model": "", "format": "gguf" if path else "", "family": model.kind,
                        "families": [model.kind], "parameter_size": "", "quantization_level": ""}}


def ollama_message(reply, calls):
    message = {"role": "assistant", "content": reply.text if reply else ""}
    if calls:
        # Ollama gives arguments as an object, not a JSON string.
        message["tool_calls"] = [{"function": {"name": c["function"]["name"],
                                               "arguments": json.loads(c["function"]["arguments"])}}
                                 for c in calls]
    return message


def ollama_done(alias, reply, reason="stop"):
    nanoseconds = int((reply.seconds if reply else 0) * 1e9)
    return {"model": alias, "created_at": now(), "message": {"role": "assistant", "content": ""},
            "done": True, "done_reason": reason, "total_duration": nanoseconds, "load_duration": 0,
            "prompt_eval_count": reply.prompt_tokens if reply else 0, "prompt_eval_duration": 0,
            "eval_count": reply.completion_tokens if reply else 0, "eval_duration": nanoseconds}


def make_handler(model, alias, max_tokens_cap, log):
    lock = threading.Lock()

    def budget(requested):
        return max(16, min(int(requested), max_tokens_cap))

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

        def read_json(self):
            """The request body as a dict, or None after answering 400."""
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("the body must be a JSON object")
                return body
            except (ValueError, TypeError) as error:
                if self.path.startswith("/api/"):
                    self.send_json(400, {"error": f"invalid request body: {error}"})
                else:
                    self.send_json(400, {"error": {"message": f"Bad request ({error})."}})
                return None

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self.send_json(200, {"object": "list", "data": [
                    {"id": alias, "object": "model", "owned_by": "kilix-chat"}]})
            elif self.path == "/api/tags":
                self.send_json(200, {"models": [model_record(model, alias)]})
            elif self.path == "/api/version":
                self.send_json(200, {"version": __version__})
            elif self.path.startswith("/api/"):
                self.send_json(404, {"error": "not found"})
            else:
                self.send_json(404, {"error": {"message": "Not found."}})

        def do_POST(self):
            routes = {"/v1/chat/completions": self.openai_chat, "/api/chat": self.ollama_chat,
                      "/api/pull": self.ollama_pull}
            route = routes.get(self.path)
            if route is None:
                self.send_json(404, {"error": "not found"} if self.path.startswith("/api/")
                               else {"error": {"message": "Not found."}})
                return
            body = self.read_json()
            if body is not None:
                route(body)

        def openai_chat(self, body):
            messages = body.get("messages")
            if not isinstance(messages, list):
                self.send_json(400, {"error": {"message": "Bad request (messages must be a list)."}})
                return
            try:
                tokens = budget(body.get("max_completion_tokens") or body.get("max_tokens") or max_tokens_cap)
            except (TypeError, ValueError):
                self.send_json(400, {"error": {"message": "Bad request (max_tokens must be a number)."}})
                return
            try:
                with lock:
                    reply, calls = model.reply(messages, tools=body.get("tools"), max_tokens=tokens)
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

        def ollama_model(self, body):
            """The requested name if it is the served model, else None after answering."""
            name = body.get("model") or body.get("name")
            if not isinstance(name, str) or not name:
                self.send_json(400, {"error": "model is required"})
                return None
            if name not in served_names(alias):
                self.send_json(404, {"error": unknown_model(name, alias)})
                return None
            return name

        def start_ndjson(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def line(self, event):
            self.wfile.write(json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n")
            self.wfile.flush()

        def ollama_pull(self, body):
            if self.ollama_model(body) is None:
                return
            if body.get("stream", True) is False:
                self.send_json(200, {"status": "success"})
                return
            self.start_ndjson()
            self.line({"status": "success"})

        def ollama_chat(self, body):
            if self.ollama_model(body) is None:
                return
            messages = body.get("messages", [])
            options = body.get("options") or {}
            if not isinstance(messages, list) or not isinstance(options, dict):
                self.send_json(400, {"error": "messages must be a list and options an object"})
                return
            requested = options.get("num_predict")
            tokens = budget(requested if isinstance(requested, int) and requested > 0 else max_tokens_cap)
            stream = body.get("stream", True) is not False
            if not messages:
                # Ollama answers an empty chat by loading the model.
                if stream:
                    self.start_ndjson()
                    self.line(ollama_done(alias, None, "load"))
                else:
                    self.send_json(200, ollama_done(alias, None, "load"))
                return
            cancel = threading.Event()
            result = {}

            def work():
                try:
                    with lock:
                        if not cancel.is_set():
                            result["reply"] = model.reply(messages, tools=body.get("tools"),
                                                          max_tokens=tokens, cancel=cancel)
                except (RuntimeError, ValueError, OSError, Cancelled) as error:
                    result["error"] = str(error) or type(error).__name__
                except Exception as error:   # a bug still answers the client, never drops it
                    result["error"] = f"kilix-chat failed: {type(error).__name__}: {error}"
                    if log:
                        log(result["error"])

            worker = threading.Thread(target=work, daemon=True)
            worker.start()
            if not stream:
                worker.join()
                if "error" in result:
                    self.send_json(500, {"error": result["error"]})
                    return
                reply, calls = result["reply"]
                self.send_json(200, dict(ollama_done(alias, reply), message=ollama_message(reply, calls)))
                return
            self.start_ndjson()
            try:
                while True:
                    worker.join(KEEPALIVE_SECONDS)
                    if not worker.is_alive():
                        break
                    self.line({"model": alias, "created_at": now(),
                               "message": {"role": "assistant", "content": ""}, "done": False})
                if "error" in result:
                    self.line({"error": result["error"]})
                    return
                reply, calls = result["reply"]
                self.line({"model": alias, "created_at": now(), "message": ollama_message(reply, calls),
                           "done": False})
                self.line(ollama_done(alias, reply))
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                # The client left: stop the reply rather than finish it unseen.
                cancel.set()
                worker.join()

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True


class _Server6(_Server):
    address_family = socket.AF_INET6


def serve(model, *, alias, host="127.0.0.1", port=8765, max_tokens_cap=512, log=None, ready=None):
    if not is_loopback(host):
        raise ValueError(f"kilix-chat serves on loopback only (127.0.0.1, ::1 or localhost), not {host}.")
    kind = _Server6 if ":" in host else _Server
    server = kind((host, port), make_handler(model, alias, max_tokens_cap, log))
    if ready:
        ready(server)
    try:
        server.serve_forever()
    finally:
        server.server_close()
