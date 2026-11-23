#!/usr/bin/env python3
"""Planted-defect controls: every edit below must turn the suite red.

Each defect is one exact edit to a copy of the checkout. It must apply
exactly once, so a stale entry cannot pass silently. A defect the suite does
not notice is a hole in the tests, and this script fails.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
S = "src/kilix_chat/"

DEFECTS = [
    ("array metadata is skipped by element count only",
     S + "gguf.py", "_read(source, struct.calcsize(_SCALARS[inner]) * count)", "_read(source, struct.calcsize(_SCALARS[inner]))"),
    ("chat models are listed as diffusion models",
     S + "diffusion.py", "            if kind in ARCHITECTURES:\n", "            if kind:\n"),
    ("the prompt leaves no room for the chat template",
     S + "diffusion.py", "self.count(system + \"\\n\" + prompt, scratch) + TEMPLATE_TOKENS", "self.count(system + \"\\n\" + prompt, scratch)"),
    ("replies keep llama.cpp's log prefix",
     S + "diffusion.py", '"--no-log-prefix", ', ""),
    ("replies are read from stderr with the driver messages",
     S + "diffusion.py", 'output = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""',
     "output = stderr"),
    ("end-of-turn markers stay in the reply",
     S + "diffusion.py", "        if cut >= 0:\n            text = text[:cut]\n", ""),
    ("the GPU choice is ignored",
     S + "diffusion.py", '        if self.device:\n                args += ["--device", self.device]\n', ""),
    ("the frame parser misses frames that start with a line break",
     S + "diffusion.py", 'frame.lstrip("\\r\\n").partition("\\n")', 'frame.partition("\\n")'),
    ("a too-long conversation is sent whole",
     S + "models.py", "                del history[first]\n", "                raise\n"),
    ("the system prompt is dropped with the history",
     S + "prompt.py", 'if message.get("role") in ("system", "developer") and not turns:',
     'if message.get("role") in ("system", "developer") and turns:'),
    ("tool results lose their content",
     S + "prompt.py", 'return f"Tool result ({name}):\\n{content}"', 'return f"Tool result ({name}):"'),
    ("calls to tools that were not offered are accepted",
     S + "prompt.py", 'if call and call["function"]["name"] in names:\n            calls.append(call)',
     "if call:\n            calls.append(call)"),
    ("the server ignores its reply length cap",
     S + "server.py", "budget = max(16, min(int(requested), max_tokens_cap))", "budget = max(16, int(requested))"),
    ("streamed tool calls end as an ordinary stop",
     S + "server.py", 'yield chunk({}, "tool_calls" if calls else "stop", usage=usage(reply))',
     'yield chunk({}, "stop", usage=usage(reply))'),
    ("Ollama replies ignore the length setting",
     S + "ollama.py", '"options": {"num_predict": max_tokens}', '"options": {}'),
    ("a failed reply leaves the unanswered message",
     S + "tui.py", "                self.partial = None\n                self.messages.pop()\n                self.status = f\"No reply: {error}\"",
     "                self.partial = None\n                self.status = f\"No reply: {error}\""),
    ("F3 does not change the reply length",
     S + "tui.py", "chat.length = LENGTHS[(LENGTHS.index(chat.length) + 1) % len(LENGTHS)]", "pass"),
]


def suite(root):
    return subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"], cwd=root,
                          env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin", "HOME": str(root)},
                          capture_output=True, text=True, timeout=600).returncode


def main():
    holes = 0
    with tempfile.TemporaryDirectory(prefix="kilix-chat-defects-") as scratch:
        for i, (name, path, old, new) in enumerate(DEFECTS, 1):
            root = Path(scratch) / str(i)
            shutil.copytree(HERE, root, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            target = root / path
            text = target.read_text()
            if text.count(old) != 1:
                print(f"BROKEN defect {i} ({name}): the edit applies {text.count(old)} times")
                holes += 1
                continue
            target.write_text(text.replace(old, new))
            red = suite(root) != 0
            print(f"{'red  ' if red else 'HOLE '} defect {i} ({name})")
            holes += not red
            shutil.rmtree(root)
    print(f"defects: {len(DEFECTS)} planted, {holes} not caught")
    return 1 if holes else 0


if __name__ == "__main__":
    sys.exit(main())
