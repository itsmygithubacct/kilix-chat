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

# Which masked tokens llama-diffusion-cli fills first, per GGUF architecture.
# Every one uses 4, the most probable tokens first. Dream and RND1 are meant
# for entropy order (1), but llama.cpp ranks entropy the wrong way round: its
# confidence is the entropy itself, sorted highest first, so it fills the
# least certain tokens first and Dream-Coder writes broken code. Every model
# uses the timestep schedule, which puts no block constraint on the length.
ARCHITECTURES = {"llada": 4, "llada-moe": 4, "dream": 4, "rnd1": 4}
MIN_STEPS = 8


@dataclass(frozen=True)
class Preset:
    """Default decoding for a family of weights. /steps still overrides it."""
    name: str
    divisor: int                 # steps per reply = reply length // divisor
    algorithm: int | None = None # None: the architecture's

    def steps(self, max_tokens):
        return max(MIN_STEPS, max_tokens // self.divisor)


DEFAULT_PRESET = Preset("default", 2)
# d3LLM (hao-ai-lab, arXiv 2601.07568) distils LLaDA and Dream to commit many
# tokens per step. On a 6 GB GPU, at 256-token replies over 10 prompts, its
# Q4_K_M GGUF gave 9 coherent replies at 32 steps (17-25 s), 5 at 64 steps
# and 10 at 128. LLaDA-1.5 gave 7 at its default of 128 steps (59-66 s).
# See README.
D3LLM_PRESET = Preset("d3LLM", 8)


def preset_for(path, info):
    """d3LLM by file name or any general.* text; else the default.

    The d3LLM_LLaDA GGUF keeps LLaDA's general.name and general.basename, and
    names d3LLM only in general.source.url and its dataset entries.
    """
    names = [Path(path).name] + [v for k, v in info.items() if k.startswith("general.")]
    if any(isinstance(n, str) and "d3llm" in n.casefold() for n in names):
        return D3LLM_PRESET
    return DEFAULT_PRESET


# llama-diffusion-cli at --temp 0 leaves out its temperature sampler and so
# samples at temperature 1. A low temperature comes close to always taking the
# most probable token, which the models are tuned for.
TEMPERATURE = "0.1"
# The chat template adds a few dozen tokens around the system and user text.
TEMPLATE_TOKENS = 64
CLEAR = "\x1b[2J\x1b[H"
END_MARKERS = ("<|im_end|>", "<|eot_id|>", "<|endoftext|>", "<|end_of_text|>", "<|eos|>")
_STEP = re.compile(r"diffusion step: (\d+)/(\d+)")
# How the window draws a token that is still masked.
MASK = "░"
_SPACES = re.compile(r" +")


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


def show_masks(frame, glyph=MASK):
    """Draw the masked tokens of a visual frame as `glyph`.

    llama-diffusion-cli prints a masked token as one space, so a frame early
    in a reply is blank. A run of spaces before a word keeps one space, the
    word's own; every other space in the run is a mask. Indentation is drawn
    as masks until the reply is finished, which then shows as written.
    """
    def run(match):
        spaces = len(match.group(0))
        word_follows = match.end() < len(frame) and frame[match.end()] != "\n"
        return glyph * (spaces - 1) + " " if word_follows else glyph * spaces
    return _SPACES.sub(run, frame)


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
        self.preset = preset_for(self.path, info)
        self.context = gguf.context_length(info) or 4096
        self.binary = tool()
        self.device = device or os.environ.get("KILIX_CHAT_DEVICE") or os.environ.get("PUZZLLM_GPU")
        self.timeout = timeout
        self.env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(filter(None, (
            str(self.binary.parent), os.environ.get("LD_LIBRARY_PATH"))))}

    @property
    def algorithm(self):
        return self.preset.algorithm or ARCHITECTURES[self.architecture]

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
        steps = steps or self.preset.steps(max_tokens)
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
                    "--diffusion-algorithm", str(self.algorithm),
                    "--temp", TEMPERATURE, "--seed", "0", "-ngl", "99",
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
