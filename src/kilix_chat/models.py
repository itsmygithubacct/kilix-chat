"""One interface over diffusion GGUFs, Ollama models and d3-talk."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kilix_chat import d3talk, diffusion, ollama, prompt

OLLAMA_PREFIX = "ollama:"
D3TALK_PREFIX = "d3-talk:"
D3TALK_FAST_PREFIX = "d3-talk-fast:"    # one partial reply instead of two
# d3-talk:NAME uses the default unit (mixed). d3-talk-words:NAME, d3-talk-tokens:NAME,
# d3-talk-letters:NAME and d3-talk-mixed:NAME name the unit.
D3TALK_UNIT_PREFIXES = {f"d3-talk-{unit}:": unit for unit in d3talk.UNITS}


def catalog():
    """(spec, label) for every model kilix-chat can open on this machine."""
    rows = [(path, f"{Path(path).name}  [{kind}]") for path, kind in diffusion.local_models()]
    chat, decision = ollama.models_by_kind()
    rows += [(OLLAMA_PREFIX + name, f"{name}  [ollama]") for name in chat]
    # A decision model cannot chat by itself, so it is offered through d3-talk.
    rows += [(D3TALK_PREFIX + name, f"{name}  [d3-talk]") for name in decision]
    return rows


def load(spec, *, device=None):
    """Open the model `spec` names: a diffusion GGUF path, ollama:NAME, or
    d3-talk:NAME for an Ollama decision model.

    Every part of kilix-chat opens models here, so this is the seam for
    anything that must know which model is about to use the GPU. A diffusion
    model's weights reach the GPU only for the length of one reply, in
    DiffusionModel.generate.
    """
    return Model(spec, device=device)


def oldest_turn(history):
    """Index of the turn to drop when a conversation is too long, or None.

    System messages stay, and so does the newest turn: without it there is
    nothing to reply to.
    """
    first = next((i for i, m in enumerate(history) if m.get("role") not in ("system", "developer")), None)
    return None if first is None or first == len(history) - 1 else first


@dataclass
class Budget:
    """How a conversation fits the model's context."""
    prompt_tokens: int      # the conversation as it would be sent, after dropping
    full_tokens: int        # the whole conversation, before dropping
    context: int            # the model's context length, in tokens
    reply_tokens: int       # the reply length reserved after the prompt
    turns: int              # turns (messages other than system ones) in the conversation
    dropped: int            # the oldest turns that would be left out to fit
    exact: bool             # counted by the model's tokenizer, not estimated
    note: str = ""

    @property
    def fits(self):
        return self.prompt_tokens + self.reply_tokens <= self.context


class Model:
    def __init__(self, spec, *, device=None):
        self.spec = spec
        if spec.startswith(OLLAMA_PREFIX):
            self.backend = ollama.OllamaModel(spec[len(OLLAMA_PREFIX):])
        elif spec.startswith(D3TALK_PREFIX):
            self.backend = d3talk.D3TalkModel(spec[len(D3TALK_PREFIX):])
        elif spec.startswith(D3TALK_FAST_PREFIX):
            self.backend = d3talk.D3TalkModel(spec[len(D3TALK_FAST_PREFIX):], fast=True)
        elif spec.split(":", 1)[0] + ":" in D3TALK_UNIT_PREFIXES:
            prefix = spec.split(":", 1)[0] + ":"
            self.backend = d3talk.D3TalkModel(spec[len(prefix):], unit=D3TALK_UNIT_PREFIXES[prefix])
        else:
            self.backend = diffusion.DiffusionModel(spec, device=device)
        self.name = self.backend.name
        self.kind = self.backend.architecture

    @property
    def diffusion(self):
        return isinstance(self.backend, diffusion.DiffusionModel)

    def reply(self, messages, *, tools=None, max_tokens=256, steps=None, on_partial=None, cancel=None):
        """Return (Reply, tool_calls). on_partial(text, step, steps) shows progress."""
        if not self.diffusion:
            if tools:
                raise ValueError("Tool calls are only offered for diffusion models.")
            text = (lambda t: on_partial(t, None, None)) if on_partial else None
            return self.backend.chat(messages, max_tokens=max_tokens, on_text=text, cancel=cancel), []
        frame = (lambda step, total, t: on_partial(t, step, total)) if on_partial else None
        history = list(messages)
        while True:
            system, user = prompt.flatten(history, tools)
            try:
                result = self.backend.generate(system, user, max_tokens=max_tokens, steps=steps,
                                               on_frame=frame, cancel=cancel)
                break
            except ValueError as error:
                # Drop the oldest non-system turn until the conversation fits.
                first = oldest_turn(history)
                if "holds" not in str(error) or first is None:
                    raise
                del history[first]
        content, calls = prompt.tool_calls(result.text, tools)
        result.text = content
        return result, calls

    def budget(self, messages, *, tools=None, max_tokens=256):
        """Budget for sending `messages` with a reply of `max_tokens`.

        It follows the same rule as reply: the oldest turns go, one at a time,
        until the prompt and the reply fit. Ollama trims an overlong prompt the
        same way, so its numbers are estimates of what Ollama will do.
        """
        if self.diffusion:
            context, exact, note = self.backend.context, self.backend.exact_counts(), ""
            def count(history):
                system, user = prompt.flatten(history, tools)
                return self.backend.prompt_tokens(system, user)
        else:
            context, note = self.backend.context_length()
            exact = False
            count = ollama.estimate
        history = list(messages)
        turns = sum(1 for m in history if m.get("role") not in ("system", "developer"))
        full = tokens = count(history)
        dropped = 0
        while tokens + max_tokens > context:
            first = oldest_turn(history)
            if first is None:
                break
            del history[first]
            dropped += 1
            tokens = count(history)
        return Budget(tokens, full, context, max_tokens, turns, dropped, exact, note)
