"""Notes kept across chats and added to every chat's system prompt.

The notes live in one Markdown list, `$XDG_DATA_HOME/kilix-chat/memory.md`
(by default under ~/.local/share), one note per `- ` line, so they can also
be edited by hand.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

HEADER = "Notes the user asked you to remember from earlier chats:"


def data_dir():
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share") / "kilix-chat"


def path():
    return data_dir() / "memory.md"


def notes():
    """The saved notes, oldest first. A missing file holds none."""
    try:
        text = path().read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    found = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- ") and line[2:].strip():
            found.append(line[2:].strip())
    return found


def _write(items):
    target = path()
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"- {item}\n" for item in items)
    # Write a sibling and rename it over the old file, so a crash never
    # leaves half a memory behind.
    handle, temporary = tempfile.mkstemp(prefix=".memory-", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(body)
        os.replace(temporary, target)
    except BaseException:
        os.unlink(temporary)
        raise


def add(text):
    """Save one note. Line breaks become spaces: one note is one line."""
    note = " ".join(text.split())
    if not note:
        raise ValueError("Say what to remember: /memory add TEXT")
    items = notes() + [note]
    _write(items)
    return items


def forget(number):
    """Remove note `number`, counted from 1. Return the removed note."""
    items = notes()
    if not 1 <= number <= len(items):
        raise ValueError(f"There is no note {number}; the memory holds {len(items)}.")
    removed = items.pop(number - 1)
    _write(items)
    return removed


def clear():
    """Remove every note. Return how many there were."""
    count = len(notes())
    if path().exists():
        _write([])
    return count


def system_text(items):
    """The part of the system prompt that carries the notes."""
    if not items:
        return ""
    return HEADER + "\n" + "\n".join(f"- {item}" for item in items)
