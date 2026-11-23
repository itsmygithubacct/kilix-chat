"""Run text diffusion GGUFs (LLaDA, LLaDA-MoE, Dream, RND1) through llama-diffusion-cli.

llama-server cannot load these models, so every reply is one
llama-diffusion-cli run. The reply, and in visual mode every intermediate
denoising frame, is read from the log file llama.cpp writes. Its stderr also
carries GPU driver messages, which must not reach the reply.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time

from kilix_chat import gguf

# Sampling algorithm per GGUF architecture, from the llama.cpp diffusion
# examples: confidence for LLaDA, entropy for Dream and RND1. Every model uses
# the timestep schedule, which puts no block constraint on the length.
ARCHITECTURES = {"llada": 4, "llada-moe": 4, "dream": 1, "rnd1": 1}
# The chat template adds a few dozen tokens around the system and user text.
TEMPLATE_TOKENS = 64
CLEAR = "\x1b[2J\x1b[H"
END_MARKERS = ("<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>", "<|eos|>")
_STEP = re.compile(r"diffusion step: (\d+)/(\d+)")


def model_folders():
    configured = os.environ.get("KILIX_CHAT_MODELS")
    if configured:
        return [Path(p).expanduser() for p in configured.split(os.pathsep) if p]
    data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    return [data / "kilix-chat/models", data / "puzzllm/diffusion"]


def local_models():
    """Diffusion GGUFs in the model folders, by file name."""
    found = {}
    for folder in model_folders():
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.gguf")):
            if not path.is_file():
                continue
            try:
                kind = gguf.architecture(path)
            except (OSError, ValueError):
                continue
            if kind in ARCHITECTURES:
                found.setdefault(str(path.absolute()), kind)
    return sorted(found.items(), key=lambda item: (Path(item[0]).name.casefold(), item[0]))


def tool():
    binary = (os.environ.get("KILIX_CHAT_DIFFUSION_CLI") or os.environ.get("PUZZLLM_DIFFUSION_CLI")
              or shutil.which("llama-diffusion-cli"))
    if not binary:
        candidates = [p for folder in model_folders() for p in folder.glob("llama-*/llama-diffusion-cli")]
        binary = str(sorted(candidates)[-1]) if candidates else None
    if not binary or not os.access(binary, os.X_OK):
        raise RuntimeError("Install llama-diffusion-cli or set KILIX_CHAT_DIFFUSION_CLI.")
    return Path(binary).resolve()


def final_reply(log, visual):
    """The denoised text, or None when generation did not finish."""
    marker = log.rfind("\ntotal time: ")
    if marker < 0:
        return None
    if visual:
        clear = log.rfind(CLEAR)
        if clear < marker:
            return None
        text = log[clear + len(CLEAR):]
    else:
        end = log.find("\n", marker + 1)
        if end < 0:
            return None
        text = log[end + 1:]
    text = text[1:] if text.startswith("\n") else text
    text = text[:-1] if text.endswith("\n") else text
    return clean(text)


def clean(text):
    for marker in END_MARKERS:
        cut = text.find(marker)
        if cut >= 0:
            text = text[:cut]
    return text.strip()


def latest_frame(log):
    """(step, steps, partial text) from the newest complete visual frame."""
    frames = log.split(CLEAR)
    for frame in reversed(frames[1:]):
        # The progress line starts with a carriage return, which the log file
        # writes as a line break.
        header, sep, body = frame.lstrip("\r\n").partition("\n")
        match = _STEP.search(header)
        if match and sep and body.endswith("\n"):
            # The timing line follows the last frame before the final clear.
            return int(match[1]), int(match[2]), body[:-1].split("\ntotal time: ")[0].rstrip("\n")
    return None


@dataclass
class Reply:
    text: str
    seconds: float
    prompt_tokens: int
    completion_tokens: int


class Cancelled(Exception):
    pass


class DiffusionModel:
    def __init__(self, path, *, device=None, timeout=1800):
        self.path = Path(path).expanduser().resolve(strict=True)
        if self.path.suffix.lower() != ".gguf":
            raise ValueError("Choose a GGUF model file.")
        info = gguf.metadata(self.path)
        self.architecture = info.get("general.architecture")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"{self.path.name} is a {self.architecture} model. llama-diffusion-cli runs "
                             + ", ".join(sorted(ARCHITECTURES)) + " models.")
        self.name = info.get("general.name") or self.path.stem
        self.context = gguf.context_length(info) or 4096
        self.binary = tool()
        self.device = device or os.environ.get("KILIX_CHAT_DEVICE") or os.environ.get("PUZZLLM_GPU")
        self.timeout = timeout
        self.env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(filter(None, (
            str(self.binary.parent), os.environ.get("LD_LIBRARY_PATH"))))}

    def exact_counts(self):
        """True when a sibling llama-tokenize counts tokens exactly."""
        counter = self.binary.with_name("llama-tokenize")
        return counter.is_file() and os.access(counter, os.X_OK)

    def prompt_tokens(self, system, prompt):
        """Tokens the prompt takes in the sequence, chat template included."""
        with tempfile.TemporaryDirectory(prefix="kilix-chat-") as folder:
            return self.count(system + "\n" + prompt, Path(folder)) + TEMPLATE_TOKENS

    def count(self, text, scratch):
        """Count with the sibling llama-tokenize, or bound it by the UTF-8 length."""
        counter = self.binary.with_name("llama-tokenize")
        if self.exact_counts():
            source = scratch / "count.txt"
            source.write_text(text, encoding="utf-8")
            result = subprocess.run([str(counter), "-m", str(self.path), "-f", str(source), "--show-count",
                                     "--log-disable"], capture_output=True, text=True, env=self.env, timeout=120)
            for line in reversed(result.stdout.splitlines()):
                if line.startswith("Total number of tokens:"):
                    return int(line.rsplit(":", 1)[1])
        return len(text.encode("utf-8"))

    def generate(self, system, prompt, *, max_tokens=256, steps=None, on_frame=None, cancel=None):
        """Denoise one reply. on_frame(step, steps, text) sees each visual frame."""
        steps = steps or max(8, max_tokens // 2)
        with tempfile.TemporaryDirectory(prefix="kilix-chat-") as folder:
            scratch = Path(folder)
            (scratch / "prompt.txt").write_text(prompt, encoding="utf-8")
            prompt_tokens = self.count(system + "\n" + prompt, scratch) + TEMPLATE_TOKENS
            if prompt_tokens + max_tokens > self.context:
                raise ValueError(f"The conversation needs {prompt_tokens + max_tokens} tokens; "
                                 f"{self.path.name} holds {self.context}.")
            length = prompt_tokens + max_tokens
            log = scratch / "diffusion.log"
            args = [str(self.binary), "-m", str(self.path), "-f", str(scratch / "prompt.txt"),
                    "-c", str(length), "-b", str(length), "-ub", str(length),
                    "--diffusion-steps", str(steps), "--diffusion-eps", "0.001",
                    "--diffusion-algorithm", str(ARCHITECTURES[self.architecture]),
                    "--temp", "0", "--seed", "0", "-ngl", "99",
                    "--log-file", str(log), "--no-log-prefix", "--no-log-timestamps", "--log-colors", "off"]
            if system:
                args[5:5] = ["-sys", system]
            if self.device:
                args += ["--device", self.device]
            visual = on_frame is not None
            if visual:
                args.append("--diffusion-visual")
            started = time.monotonic()
            with (scratch / "stderr").open("w+", encoding="utf-8", errors="replace") as errors:
                process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=errors, env=self.env, start_new_session=True)
                try:
                    seen = None
                    while process.poll() is None:
                        if cancel is not None and cancel.is_set():
                            raise Cancelled()
                        if time.monotonic() - started > self.timeout:
                            raise RuntimeError(f"llama-diffusion-cli gave no reply within {self.timeout} seconds.")
                        if visual and log.exists():
                            frame = latest_frame(log.read_text(encoding="utf-8", errors="replace"))
                            if frame and frame != seen:
                                seen = frame
                                on_frame(*frame)
                        time.sleep(0.1)
                finally:
                    if process.poll() is None:
                        _stop(process)
                errors.seek(0)
                stderr = errors.read()
            output = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            if visual and process.returncode == 0:
                # A short run can finish between polls: show its last frame.
                frame = latest_frame(output)
                if frame and frame != seen:
                    on_frame(*frame)
            text = final_reply(output, visual)
            if process.returncode != 0 or text is None:
                if "failed to allocate" in output or "unable to allocate" in output:
                    raise RuntimeError(f"There is not enough free GPU memory for {self.path.name}. "
                                       "Another model may be loaded (see `ollama ps`).")
                lines = [line for line in (output or stderr).strip().splitlines() if line.strip()]
                raise RuntimeError(f"llama-diffusion-cli exited {process.returncode}. "
                                   + (lines[-1][:300] if lines else ""))
            return Reply(text, time.monotonic() - started, prompt_tokens, max_tokens)


def _stop(process):
    for sig, wait in ((signal.SIGTERM, 2), (signal.SIGKILL, None)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
