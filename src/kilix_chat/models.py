"""One interface over diffusion GGUFs and Ollama models."""
from __future__ import annotations

from pathlib import Path

from kilix_chat import diffusion, ollama, prompt

OLLAMA_PREFIX = "ollama:"


def catalog():
    """(spec, label) for every model kilix-chat can open on this machine."""
    rows = [(path, f"{Path(path).name}  [{kind}]") for path, kind in diffusion.local_models()]
    rows += [(OLLAMA_PREFIX + name, f"{name}  [ollama]") for name in ollama.local_models()]
    return rows


class Model:
    def __init__(self, spec, *, device=None):
        self.spec = spec
        if spec.startswith(OLLAMA_PREFIX):
            self.backend = ollama.OllamaModel(spec[len(OLLAMA_PREFIX):])
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
                first = next((i for i, m in enumerate(history) if m.get("role") not in ("system", "developer")), None)
                if "holds" not in str(error) or first is None or first == len(history) - 1:
                    raise
                del history[first]
        content, calls = prompt.tool_calls(result.text, tools)
        result.text = content
        return result, calls
