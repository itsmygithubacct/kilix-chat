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
     S + "server.py", "return max(16, min(int(requested), max_tokens_cap))", "return max(16, int(requested))"),
    ("streamed tool calls end as an ordinary stop",
     S + "server.py", 'yield chunk({}, "tool_calls" if calls else "stop", usage=usage(reply))',
     'yield chunk({}, "stop", usage=usage(reply))'),
    ("a slow Ollama stream sends nothing until the reply is done",
     S + "server.py", '''                    self.line({"model": alias, "created_at": now(),
                               "message": {"role": "assistant", "content": ""}, "done": False})''', "                    pass"),
    ("the Ollama done line has no done_reason",
     S + "server.py", '"done": True, "done_reason": reason,', '"done": True,'),
    ("Ollama requests for any model name are answered",
     S + "server.py", "            if name not in served_names(alias):", "            if False:"),
    ("NAME:latest is not the served model",
     S + "server.py", '{alias} if ":" in alias else {alias, alias + ":latest"}', "{alias}"),
    ("Ollama's num_predict is ignored",
     S + "server.py", "tokens = budget(requested if isinstance(requested, int) and requested > 0 else max_tokens_cap)",
     "tokens = budget(max_tokens_cap)"),
    ("an Ollama chat without stream is not streamed",
     S + "server.py", 'stream = body.get("stream", True) is not False', 'stream = bool(body.get("stream"))'),
    ("a failed Ollama stream ends without an error line",
     S + "server.py", '                    self.line({"error": result["error"]})\n', ""),
    ("an unexpected failure drops the Ollama client",
     S + "server.py", "                except Exception as error:   # a bug still answers the client, never drops it\n",
     "                except ZeroDivisionError as error:\n"),
    ("a client that leaves does not stop the reply",
     S + "server.py", "                cancel.set()\n                worker.join()", "                worker.join()"),
    ("Ollama tool call arguments are a JSON string",
     S + "server.py", '"arguments": json.loads(c["function"]["arguments"])}}', '"arguments": c["function"]["arguments"]}}'),
    ("/api/tags lists the file's model name, not the alias",
     S + "server.py", 'return {"name": alias, "model": alias, "size": size,', 'return {"name": model.name, "model": alias, "size": size,'),
    ("serve binds a non-loopback address",
     S + "server.py", "    if not is_loopback(host):\n        raise", "    if False:\n        raise"),
    ("the serve command accepts a non-loopback host",
     S + "cli.py", "            if not is_loopback(args.host):", "            if False:"),
    ("decision models are listed as chat models",
     S + "ollama.py", '    return "decision" in capabilities and "completion" not in capabilities', "    return False"),
    ("a decision question reports its least likely choice first",
     S + "ollama.py", "for key, value in odds.items()), key=lambda item: -item[1])",
     "for key, value in odds.items()), key=lambda item: item[1])"),
    ("d3-talk:NAME is opened as a GGUF path",
     S + "models.py", "        elif spec.startswith(D3TALK_PREFIX):\n"
     "            self.backend = d3talk.D3TalkModel(spec[len(D3TALK_PREFIX):])\n", ""),
    ("a d3-talk sentence may open on a topic word",
     S + "d3talk.py", "            topic = []\n", "            pass\n"),
    ("d3-talk offers the same word twice in a row",
     S + "d3talk.py", "        banned = set(words[-1:])\n", "        banned = set()\n"),
    ("a finished d3-talk reply teaches nothing",
     S + "d3talk.py", "            if not self.lexicon.learn(steps):\n", "            if not True:\n"),
    ("d3-talk saves only its own learning over another writer's",
     S + "d3talk.py", "        used = self.saved_usage()\n", "        used = self.used\n"),
    ("a failed d3-talk save loses the reply",
     S + "d3talk.py", "        except OSError:\n            return False\n", "        except ValueError:\n            return False\n"),
    ("one bad usage row drops the rest of the file",
     S + "d3talk.py", "                    except (IndexError, ValueError):\n                        continue\n",
     "                    except (IndexError, ValueError):\n                        break\n"),
    ("rows used zero times are counted",
     S + "d3talk.py", "if len(parts) == 4 and times > 0:", "if len(parts) == 4:"),
    ("a d3-talk download cut short escapes as an unexpected error",
     S + "d3talk.py", "except (OSError, URLError, HTTPException) as error:", "except (OSError, URLError) as error:"),
    ("a failed d3-talk download leaves its part file",
     S + "d3talk.py", "            if part:\n                Path(part).unlink(missing_ok=True)\n", ""),
    ("d3-talk answers only the last of two user messages",
     S + "d3talk.py", 'said = f"{said}\\n{text}" if said else text', "said = text"),
    ("learned word pairs are not offered",
     S + "d3talk.py", "                    score[word] = score.get(word, 0) + 0.5 * times / (mine + 20)\n",
     "                    pass\n"),
    ("a d3-talk reply cannot be stopped",
     S + "d3talk.py", "        if self.cancel is not None and self.cancel.is_set():\n            raise Cancelled()\n", ""),
    ("d3-talk forgets the earlier turns",
     S + "d3talk.py", '        if past:\n            text += "Earlier in the conversation:\\n" + past + "\\n\\n"\n', ""),
    ("d3-talk ignores the system prompt and the memory",
     S + "d3talk.py", 'text = f"Notes for the assistant: {notes[:NOTES_CHARS]}\\n\\n" if notes else ""', 'text = ""'),
    ("d3-talk ignores the reply length",
     S + "d3talk.py", "text = talker.reply(message, turns[-3:], notes, max_words=max_tokens)",
     "text = talker.reply(message, turns[-3:], notes)"),
    ("d3-talk writes 'a apple'",
     S + "d3talk.py", '            word = "an" if words[place + 1][0] in "aeiou" else "a"\n', "            pass\n"),
    ("d3-talk offers web junk and two-letter non-words",
     S + "d3talk.py", "                if word in JUNK or not word.isalpha() or (len(word) <= 2 and word not in SHORT):\n"
     "                    continue\n", ""),
    ("d3-talk --show prints only the finished reply",
     S + "d3talk.py", '        if args.show and not text.startswith("("):\n            print(text, flush=True)\n'
     "        elif live:\n", "        if live:\n"),
    ("the d3-talk sweep reads every word again after choosing the best lines",
     S + "d3talk.py", '            candidates = " ".join(kept).split()\n', "            pass\n"),
    ("a d3-talk sentence is written on after its best candidate has ended",
     S + "d3talk.py", "            if not beams or kept[0][-1][0] in STOPS:\n", "            if not beams:\n"),
    ("d3-talk-fast:NAME keeps two partial replies",
     S + "d3talk.py", "Settings(beam=1) if fast else Settings()", "Settings()"),
    ("d3-talk --fast keeps two partial replies",
     S + "d3talk.py", "Settings(1 if args.fast else args.beam,", "Settings(args.beam,"),
    ("d3-talk --no-learn still counts the words",
     S + "d3talk.py", "                        not args.no_learn)", "                        True)"),
    ("an Ollama tool result loses its tool name",
     S + "prompt.py", 'message.get("name") or message.get("tool_name") or', 'message.get("name") or'),
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
    ("Dream fills its least certain tokens first",
     S + "diffusion.py", '"dream": 4, "rnd1": 4}', '"dream": 1, "rnd1": 1}'),
    ("diffusion replies sample at llama.cpp's temperature 1",
     S + "diffusion.py", 'TEMPERATURE = "0.1"', 'TEMPERATURE = "0"'),
    ("a tall denoising reply shows only its padded end",
     S + "tui.py", "            end = max(0, min(len(rows), start + body) - chat.scroll)\n", ""),
    ("d3LLM is recognised by its file name only",
     S + "diffusion.py", 'names = [Path(path).name] + [v for k, v in info.items() if k.startswith("general.")]',
     "names = [Path(path).name]"),
    ("d3LLM is recognised by any metadata text, not only general.*",
     S + "diffusion.py", 'for k, v in info.items() if k.startswith("general.")]', "for k, v in info.items()]"),
    ("d3LLM is recognised only in lower case",
     S + "diffusion.py", '"d3llm" in n.casefold()', '"d3llm" in n'),
    ("the d3LLM preset is not used",
     S + "diffusion.py", "        steps = steps or self.preset.steps(max_tokens)", "        steps = steps or max(8, max_tokens // 2)"),
    ("the d3LLM preset takes the default step count",
     S + "diffusion.py", 'D3LLM_PRESET = Preset("d3LLM", 8)', 'D3LLM_PRESET = Preset("d3LLM", 2)'),
    ("a short reply gets fewer than the minimum steps",
     S + "diffusion.py", "        return max(MIN_STEPS, max_tokens // self.divisor)", "        return max_tokens // self.divisor"),
    ("masked tokens show as blank space",
     S + "tui.py", "        if step is not None:\n            text = show_masks(text)\n", ""),
    ("the space before a word is drawn as a mask",
     S + "diffusion.py", 'return glyph * (spaces - 1) + " " if word_follows else glyph * spaces', "return glyph * spaces"),
    ("masks before a line break are left blank",
     S + "diffusion.py", 'word_follows = match.end() < len(frame) and frame[match.end()] != "\\n"',
     "word_follows = match.end() < len(frame)"),
    ("/visual off still shows the denoising",
     S + "tui.py", "on_partial=self._progress if self.visual else None,", "on_partial=self._progress,"),
    ("F4 does not toggle the denoising display",
     S + "tui.py", '        commands.lookup("visual").run(chat, "")\n', ""),
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
    """The suite's exit status; a run that hangs counts as red."""
    try:
        return subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"], cwd=root,
                              env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin", "HOME": str(root),
                                   "TMPDIR": tempfile.gettempdir()},
                              capture_output=True, text=True, timeout=300).returncode
    except subprocess.TimeoutExpired:
        print("      (the suite timed out)")
        return 124


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
