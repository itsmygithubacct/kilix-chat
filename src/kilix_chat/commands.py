"""Slash commands for the chat window.

A line that starts with `/` runs a command and never reaches the model. Two
slashes send the line as a message with one slash removed, so a message can
still start with `/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
from typing import Callable

from kilix_chat import memory

MIN_LENGTH, MAX_LENGTH = 8, 16384
MAX_STEPS = 4096


@dataclass(frozen=True)
class Command:
    name: str
    usage: str
    help: str
    run: Callable
    aliases: tuple = ()
    idle: bool = False           # waits for the reply in progress
    choices: tuple = field(default=())   # words Tab completes after the name


def is_command(line):
    return line.startswith("/") and not line.startswith("//")


def _number(arg, low, high, what):
    try:
        value = int(arg)
    except ValueError:
        raise ValueError(f"{what} must be a whole number from {low} to {high}.") from None
    if not low <= value <= high:
        raise ValueError(f"{what} must be from {low} to {high}.")
    return value


def _quit(chat, arg):
    chat.cancel.set()
    return False


def _help(chat, arg):
    width = max(len(f"/{c.name} {c.usage}".rstrip()) for c in COMMANDS) + 2
    lines = []
    for command in COMMANDS:
        line = f"/{command.name} {command.usage}".rstrip().ljust(width) + command.help
        if command.aliases:
            line += " (also " + ", ".join("/" + a for a in command.aliases) + ")"
        lines.append(line)
    lines += ["", "Tab completes a command. Start a message with // to send it with one leading /.",
              "Keys: Enter send · Tab complete · Ctrl-U clear the line · F2 model · F3 reply length · F4 show denoising · "
              "Ctrl-N new chat · PgUp/PgDn scroll · Esc stop · F10 or Ctrl-Q quit"]
    chat.show("Commands", lines)


def _model(chat, arg):
    if arg:
        chat.switch(arg)
    else:
        chat.open_picker()


def _context(chat, arg):
    if not chat.model:
        raise ValueError("Choose a model first (/model or F2).")
    chat.background(chat.context_report, "Counting tokens…")


def _memory(chat, arg):
    word, _, rest = arg.partition(" ")
    word = word.lower()
    if not word:
        items = memory.notes()
        if items:
            lines = [f"{i}. {item}" for i, item in enumerate(items, 1)]
        else:
            lines = ["Nothing yet. /memory add TEXT saves a note."]
        lines += ["", f"Kept in {memory.path()} and added to the system prompt of every chat."]
        chat.show("Memory", lines)
    elif word == "add":
        items = memory.add(rest)
        chat.status = f"Remembered note {len(items)}. It is in the system prompt from the next reply."
    elif word == "forget":
        removed = memory.forget(_number(rest.strip(), 1, 10 ** 6, "The note number"))
        chat.status = f"Forgot: {removed}"
    elif word == "clear":
        count = memory.clear()
        chat.status = f"Forgot {count} note{'s' if count != 1 else ''}."
    else:
        raise ValueError("Use /memory, /memory add TEXT, /memory forget N or /memory clear.")


def _remember(chat, arg):
    _memory(chat, "add " + arg)


def _clear(chat, arg):
    chat.new_chat()


def _system(chat, arg):
    if not arg:
        lines = (chat.system or "(none)").splitlines()
        notes = memory.system_text(memory.notes())
        if notes:
            lines += ["", "Added from the memory:"] + notes.splitlines()
        chat.show("System prompt", lines)
    elif arg.lower() == "clear":
        chat.system = ""
        chat.status = "The system prompt is cleared."
    else:
        chat.system = arg
        chat.status = "The system prompt is set. It applies from the next reply."


def _length(chat, arg):
    if not arg:
        chat.status = f"Replies are up to {chat.length} tokens. /length N changes it; F3 cycles."
        return
    chat.length = _number(arg, MIN_LENGTH, MAX_LENGTH, "The reply length")
    chat.status = f"Replies are up to {chat.length} tokens."


def _steps(chat, arg):
    if not arg:
        shown = chat.steps or f"automatic (half the reply length, {max(8, chat.length // 2)} now)"
        chat.status = f"Diffusion steps: {shown}. /steps N or /steps auto changes it."
        return
    chat.steps = None if arg.lower() == "auto" else _number(arg, 1, MAX_STEPS, "The step count")
    chat.status = f"Diffusion steps: {chat.steps or 'automatic'}."
    if chat.model and not chat.model.diffusion:
        chat.status += " Ollama models do not use steps."


def _save(chat, arg):
    if not chat.messages:
        raise ValueError("Nothing to save yet.")
    if arg:
        target = Path(os.path.expanduser(arg))
    else:
        target = memory.data_dir() / "chats" / datetime.now().strftime("chat-%Y%m%d-%H%M%S.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(chat.markdown(), encoding="utf-8")
    chat.status = f"Saved the chat to {target}."


def _visual(chat, arg):
    word = arg.lower()
    if word not in ("", "on", "off"):
        raise ValueError("Use /visual on or /visual off.")
    chat.visual = (not chat.visual) if not word else word == "on"
    if chat.visual:
        chat.status = "Replies show as they arrive; a diffusion reply shows each denoising step, ░ for masked tokens."
    else:
        chat.status = "Replies show only when finished."
    if chat.busy:
        chat.status += " From the next reply."


def _retry(chat, arg):
    if not chat.messages or chat.messages[-1]["role"] != "assistant":
        raise ValueError("There is no reply to retry.")
    chat.messages.pop()
    chat.start_reply()


def _undo(chat, arg):
    if not chat.messages:
        raise ValueError("There is nothing to undo.")
    if chat.messages[-1]["role"] == "assistant":
        chat.messages.pop()
    text = chat.messages.pop()["content"] if chat.messages and chat.messages[-1]["role"] == "user" else ""
    chat.entry = text
    chat.status = "Took back the last exchange. Your message is back in the entry."


def _device(chat, arg):
    if not arg:
        chat.status = f"llama.cpp device: {chat.device or 'llama.cpp chooses'}."
        return
    chat.device = None if arg.lower() == "auto" else arg
    if chat.model and chat.model.diffusion:
        chat.model.backend.device = chat.device
    chat.status = f"llama.cpp device: {chat.device or 'llama.cpp chooses'}, from the next reply."
    if chat.model and not chat.model.diffusion:
        chat.status += " Ollama picks its own device."


COMMANDS = (
    Command("help", "", "list the commands and keys", _help, ("?",)),
    Command("model", "[GGUF|ollama:NAME]", "choose a model, or switch to the one named", _model, idle=True),
    Command("context", "", "how much of the model's context this chat uses", _context, idle=True),
    Command("memory", "[add TEXT|forget N|clear]", "show or change the notes kept across chats", _memory,
            choices=("add", "forget", "clear")),
    Command("remember", "TEXT", "the same as /memory add TEXT", _remember),
    Command("system", "[TEXT|clear]", "show or set the system prompt", _system, choices=("clear",)),
    Command("length", "[N]", f"set the reply length in tokens ({MIN_LENGTH}-{MAX_LENGTH})", _length),
    Command("steps", "[N|auto]", "set the diffusion steps per reply", _steps, choices=("auto",)),
    Command("device", "[NAME|auto]", "set the llama.cpp device, for example Vulkan1", _device, choices=("auto",)),
    Command("visual", "[on|off]", "show replies as they denoise or stream (on; F4 toggles)", _visual,
            ("diffusion",), choices=("on", "off")),
    Command("retry", "", "ask for the last reply again", _retry, idle=True),
    Command("undo", "", "take back the last exchange", _undo, idle=True),
    Command("save", "[FILE]", "write the chat as Markdown", _save),
    Command("clear", "", "start a new chat", _clear, ("new",), idle=True),
    Command("quit", "", "leave kilix-chat", _quit, ("exit", "q")),
)


def lookup(name):
    name = name.lower()
    return next((c for c in COMMANDS if name == c.name or name in c.aliases), None)


def matches(prefix):
    """Commands whose name or an alias starts with `prefix` (no slash)."""
    prefix = prefix.lower()
    return [c for c in COMMANDS if c.name.startswith(prefix) or any(a.startswith(prefix) for a in c.aliases)]


def _common(words):
    first, last = min(words), max(words)
    size = 0
    while size < min(len(first), len(last)) and first[size] == last[size]:
        size += 1
    return first[:size]


def complete(entry):
    """The entry with its command name, or its first argument, completed."""
    if not is_command(entry):
        return entry
    name, space, rest = entry[1:].partition(" ")
    if not space:
        names = [n for c in COMMANDS for n in (c.name, *c.aliases) if n.startswith(name.lower())]
        if not names:
            return entry
        if len(names) == 1:
            return f"/{names[0]} "
        return "/" + max(name, _common(names), key=len)
    command = lookup(name)
    if not command or " " in rest:
        return entry
    words = [w for w in command.choices if w.startswith(rest.lower())]
    if len(words) == 1:
        return f"/{name} {words[0]} "
    return f"/{name} {max(rest, _common(words), key=len)}" if words else entry


def hint(entry):
    """The status line while a command is typed."""
    name, space, _ = entry[1:].partition(" ")
    found = [] if space else matches(name)
    command = lookup(name) or (found[0] if len(found) == 1 else None)
    if command:
        return f"/{command.name} {command.usage}".rstrip() + f" — {command.help}"
    if not found:
        return f"Unknown command /{name}. /help lists the commands."
    return "  ".join(f"/{c.name}" for c in found) + "   (Tab completes)"


def run(chat, line):
    """Run one command line. Return False to quit."""
    name, _, arg = line[1:].strip().partition(" ")
    command = lookup(name)
    if command is None:
        chat.status = f"Unknown command /{name}. /help lists the commands." if name else \
            "Type a command after the /, or /help."
        return True
    if command.idle and chat.busy:
        chat.status = f"/{command.name} waits for the reply. Esc stops it."
        return True
    chat.entry = ""
    try:
        return command.run(chat, arg.strip()) is not False
    except (OSError, ValueError) as error:
        chat.status = str(error)
        return True
