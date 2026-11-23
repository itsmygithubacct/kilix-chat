"""Chat with a model served by a local Ollama daemon."""
from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from kilix_chat.diffusion import Cancelled, Reply


def base_url():
    host = os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    if urlsplit(host).port is None and not urlsplit(host).path.strip("/"):
        host = host.rstrip("/") + ":11434"
    return host.rstrip("/")


def _post(path, body, timeout):
    request = Request(base_url() + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urlopen(request, timeout=timeout)


def local_models(timeout=2):
    try:
        with urlopen(base_url() + "/api/tags", timeout=timeout) as response:
            return sorted({row["name"] for row in json.load(response).get("models", [])})
    except (OSError, URLError, ValueError, KeyError):
        return []


class OllamaModel:
    architecture = "ollama"

    def __init__(self, name):
        self.name = name

    def chat(self, messages, *, max_tokens=256, on_text=None, cancel=None):
        body = {"model": self.name, "messages": messages, "stream": True, "think": False,
                "options": {"num_predict": max_tokens}}
        started = time.monotonic()
        text, prompt_tokens, completion_tokens = "", 0, 0
        try:
            response = _post("/api/chat", body, timeout=600)
        except HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            if error.code == 400 and "think" in detail.lower():
                del body["think"]
                response = _post("/api/chat", body, timeout=600)
            else:
                raise RuntimeError(f"Ollama returned HTTP {error.code}. {detail[:300]}") from error
        except (OSError, URLError) as error:
            raise RuntimeError(f"Could not reach Ollama at {base_url()} ({error}).") from error
        with response:
            for line in response:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                part = json.loads(line)
                if part.get("error"):
                    raise RuntimeError(part["error"])
                text += (part.get("message") or {}).get("content") or ""
                if on_text:
                    on_text(text)
                if part.get("done"):
                    prompt_tokens = part.get("prompt_eval_count") or 0
                    completion_tokens = part.get("eval_count") or 0
                    break
        return Reply(text.strip(), time.monotonic() - started, prompt_tokens, completion_tokens)
