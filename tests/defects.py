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
    ("a full GPU is reported as an opaque load failure",
     S + "diffusion.py", 'if "failed to allocate" in output or "unable to allocate" in output:',
     'if False:'),
    ("F3 does not change the reply length",
     S + "tui.py", "chat.length = next_length(chat.length)", "pass"),
    ("F3 fails on a reply length set with /length",
     S + "tui.py", "return next((n for n in LENGTHS if n > length), LENGTHS[0])",
     "return LENGTHS[(LENGTHS.index(length) + 1) % len(LENGTHS)]"),
    ("unknown commands are sent to the model",
     S + "tui.py", "if commands.is_command(text):", "if commands.is_command(text) and commands.lookup(text[1:].split()[0]):"),
    ("// does not escape a leading slash",
     S + "tui.py", '        if text.startswith("//"):\n            text = text[1:]\n', ""),
    ("Ctrl-U does not clear the entry",
     S + "tui.py", '    elif key == "\\x15":\n        chat.entry = ""\n', ""),
    ("/exit does not quit",
     S + "commands.py", '("exit", "q")', '("q",)'),
    ("/model SPEC opens the picker instead of the model",
     S + "commands.py", "    if arg:\n        chat.switch(arg)\n    else:\n        chat.open_picker()",
     "    chat.open_picker()"),
    ("commands that change the chat run during a reply",
     S + "commands.py", "if command.idle and chat.busy:", "if False:"),
    ("/new does not start a new chat",
     S + "commands.py", '_clear, ("new",)', "_clear, ()"),
    ("/system TEXT is not used",
     S + "commands.py", "        chat.system = arg\n", "        pass\n"),
    ("/length accepts any number",
     S + "commands.py", "    if not low <= value <= high:", "    if False:"),
    ("/steps does not reach llama-diffusion-cli",
     S + "tui.py", "max_tokens=self.length, steps=self.steps,", "max_tokens=self.length,"),
    ("/retry keeps the old reply",
     S + "commands.py", "    chat.messages.pop()\n    chat.start_reply()", "    chat.start_reply()"),
    ("/device is ignored by the open model",
     S + "commands.py", "        chat.model.backend.device = chat.device", "        pass"),
    ("/save leaves out the model's replies",
     S + "tui.py", "        for message in self.messages:\n", "        for message in self.messages[::2]:\n"),
    ("Tab completes an ambiguous command to the first match",
     S + "commands.py", "        if len(names) == 1:\n            return f\"/{names[0]} \"", "        if names:\n            return f\"/{names[0]} \""),
    ("the hint line is not shown while a command is typed",
     S + "tui.py", "        status = commands.hint(chat.entry)", "        pass"),
    ("the memory is not saved to disk",
     S + "memory.py", "    items = notes() + [note]\n    _write(items)\n", "    items = notes() + [note]\n"),
    ("/memory clear keeps the notes",
     S + "memory.py", "        _write([])", "        pass"),
    ("the memory is left out of the system prompt",
     S + "tui.py", "(self.system.strip(), memory.system_text(memory.notes()))", "(self.system.strip(),)"),
    ("/context counts the chat without its system prompt",
     S + "tui.py", "self.model.budget(self.conversation(), max_tokens=self.length)",
     "self.model.budget(list(self.messages), max_tokens=self.length)"),
    ("/context reserves no room for the reply",
     S + "models.py", "        while tokens + max_tokens > context:", "        while tokens > context:"),
    ("/context miscounts the dropped turns",
     S + "models.py", "            dropped += 1\n", "            dropped = 1\n"),
    ("the window of a loaded Ollama model is ignored",
     S + "ollama.py", 'if self.name in (row.get("name"), row.get("model")) and', "if False and"),
    ("num_ctx in an Ollama model's parameters is ignored",
     S + "ollama.py", 'if key == "num_ctx" and value.strip().isdigit():', "if False:"),
]


def suite(root):
    return subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"], cwd=root,
                          env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin", "HOME": str(root),
                               "TMPDIR": tempfile.gettempdir()},
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
