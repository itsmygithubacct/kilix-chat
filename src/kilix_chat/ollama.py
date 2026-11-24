"""Chat with a model served by a local Ollama daemon, or put a question to
one of its decision models."""
from __future__ import annotations

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from kilix_chat.diffusion import Cancelled, Reply
from kilix_chat.prompt import text_of

# Without a tokenizer, a token is taken to be four bytes of UTF-8, and every
# message adds a few tokens of chat template around its text.
BYTES_PER_TOKEN = 4
MESSAGE_TOKENS = 4
DEFAULT_CONTEXT = 4096


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


def is_decision(row):
    """Whether an /api/tags row is a decision model: one that answers
    questions about text and cannot write any."""
    capabilities = row.get("capabilities") or []
    return "decision" in capabilities and "completion" not in capabilities


def models_by_kind(timeout=2):
    """(names of the models that chat, names of the decision models)."""
    try:
        with urlopen(base_url() + "/api/tags", timeout=timeout) as response:
            rows = json.load(response).get("models", [])
        chat = sorted({row["name"] for row in rows if not is_decision(row)})
        return chat, sorted({row["name"] for row in rows if is_decision(row)})
    except (OSError, URLError, ValueError, KeyError, AttributeError):
        return [], []


def local_models(timeout=2):
    return models_by_kind(timeout)[0]


def decide(model, state, question, options, timeout=600):
    """Put one multiple-choice question to a decision model.

    `state` is the text the question is about, and `options` maps each
    choice's key to the text shown beside it, which may be empty. Returns
    ([(key, probability), ...] most likely first, prompt tokens). The model
    reads the state and every option again for each question, so the cost
    grows with their length.
    """
    body = {"model": model, "state": state,
            "questions": {"q": {"type": "choice", "instructions": question, "criteria": options}}}
    try:
        with _post("/v1/systemone", body, timeout) as response:
            answer = json.load(response)
        odds = answer["answers"]["q"]["probabilities"]
        ranked = sorted(((key, float(value)) for key, value in odds.items()), key=lambda item: -item[1])
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"Ollama returned HTTP {error.code} for a decision question to {model}. "
                           f"{detail[:300]}") from error
    except (OSError, URLError) as error:
        raise RuntimeError(f"Could not reach Ollama at {base_url()} ({error}).") from error
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError(f"Ollama's answer to a decision question was not understood ({error!r}).") from error
    return ranked, (answer.get("usage") or {}).get("input_tokens") or 0


def _get(path, timeout):
    with urlopen(base_url() + path, timeout=timeout) as response:
        return json.load(response)


def estimate(messages):
    """Estimated prompt tokens for `messages`."""
    total = 0
    for message in messages:
        size = len(text_of(message.get("content")).encode("utf-8"))
        total += -(-size // BYTES_PER_TOKEN) + MESSAGE_TOKENS
    return total


class OllamaModel:
    architecture = "ollama"

    def __init__(self, name):
        self.name = name

    def context_length(self, timeout=3):
        """(tokens, where the number comes from) for this model's context window."""
        try:
            for row in _get("/api/ps", timeout).get("models", []):
                if self.name in (row.get("name"), row.get("model")) and isinstance(row.get("context_length"), int):
                    return row["context_length"], "the window Ollama has it loaded with"
        except (OSError, URLError, ValueError, AttributeError):
            pass
        try:
            with _post("/api/show", {"model": self.name}, timeout) as response:
                shown = json.load(response)
        except (OSError, URLError, ValueError):
            return DEFAULT_CONTEXT, "Ollama did not answer, so this is a guess"
        for line in (shown.get("parameters") or "").splitlines():
            key, _, value = line.partition(" ")
            if key == "num_ctx" and value.strip().isdigit():
                return int(value.strip()), "num_ctx in the model's parameters"
        info = shown.get("model_info") or {}
        value = info.get(f"{info.get('general.architecture')}.context_length")
        if isinstance(value, int) and value > 0:
            return value, "the model's trained length; Ollama may load it with a smaller num_ctx"
        return DEFAULT_CONTEXT, "Ollama did not say, so this is a guess"

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
