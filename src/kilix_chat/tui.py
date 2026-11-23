"""Terminal chat: one conversation with one model at a time.

Diffusion replies are shown as they denoise: the whole reply appears at once
and sharpens step by step, rather than growing word by word.
"""
from __future__ import annotations

import curses
import threading
import textwrap

from kilix_chat.diffusion import Cancelled
from kilix_chat.models import Model, catalog

LENGTHS = (64, 128, 256, 512)
HELP = "Enter send · F2 model · F3 reply length · Ctrl-N new chat · Esc stop · F10 quit"
ENTRY = "Enter a GGUF path or ollama:NAME..."


def wrap(text, width):
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width, replace_whitespace=False, drop_whitespace=True) or [""])
    return lines


def transcript_lines(messages, partial, width, model_name):
    """(text, style) rows for the conversation, newest last."""
    rows = []
    for message in messages:
        mine = message["role"] == "user"
        rows.append(("You" if mine else model_name, "you" if mine else "them"))
        rows.extend((line, "text") for line in wrap(message["content"], max(10, width - 2)))
        rows.append(("", "text"))
    if partial is not None:
        text, label = partial
        rows.append((f"{model_name} {label}".rstrip(), "them"))
        rows.extend((line, "partial") for line in wrap(text or "…", max(10, width - 2)))
    return rows


class Chat:
    def __init__(self, spec=None, device=None, max_tokens=256):
        self.device = device
        self.model = None
        self.messages = []
        self.partial = None
        self.status = ""
        self.entry = ""
        self.scroll = 0
        self.length = max_tokens if max_tokens in LENGTHS else 256
        self.worker = None
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.picking = False
        self.choices = []
        self.cursor = 0
        self.typing_path = False
        if spec:
            self.open(spec)

    def open(self, spec):
        try:
            self.model = Model(spec, device=self.device)
        except (OSError, RuntimeError, ValueError) as error:
            self.status = f"Could not open the model: {error}"
            return False
        self.messages = []
        self.status = f"Chatting with {self.model.name} ({self.model.kind})."
        return True

    @property
    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def send(self):
        text = self.entry.strip()
        if not text or self.busy:
            return
        if not self.model:
            self.status = "Choose a model first (F2)."
            return
        self.entry = ""
        self.scroll = 0
        self.messages.append({"role": "user", "content": text})
        self.cancel.clear()
        self.partial = ("", "is thinking…")
        self.worker = threading.Thread(target=self._reply, args=(list(self.messages),), daemon=True)
        self.worker.start()

    def _progress(self, text, step, steps):
        with self.lock:
            self.partial = (text, f"(step {step}/{steps})" if step is not None else "")

    def _reply(self, history):
        try:
            reply, _ = self.model.reply(history, max_tokens=self.length, on_partial=self._progress, cancel=self.cancel)
        except Cancelled:
            with self.lock:
                self.partial = None
                self.messages.pop()
                self.status = "Stopped. Your message was taken back."
            return
        except (OSError, RuntimeError, ValueError) as error:
            with self.lock:
                self.partial = None
                self.messages.pop()
                self.status = f"No reply: {error}"
            return
        with self.lock:
            self.partial = None
            self.messages.append({"role": "assistant", "content": reply.text})
            self.status = f"Replied in {reply.seconds:.1f} s."

    def open_picker(self):
        self.choices = catalog()
        self.cursor = 0
        self.picking = True
        self.typing_path = False

    def pick(self):
        if self.cursor == len(self.choices):
            self.typing_path = True
            self.entry = ""
            return
        if self.choices and self.open(self.choices[self.cursor][0]):
            self.picking = False


def draw(screen, chat):
    screen.erase()
    height, width = screen.getmaxyx()
    if height < 6 or width < 30:
        screen.addnstr(0, 0, "Enlarge the window.", max(1, width - 1))
        return
    title = " kilix-chat"
    if chat.model:
        title += f" · {chat.model.name} [{chat.model.kind}]"
    title += f" · replies up to {chat.length} tokens"
    screen.addnstr(0, 0, title.ljust(width), width - 1, curses.A_REVERSE)
    body = height - 3
    if chat.picking:
        screen.addnstr(1, 1, "Choose a model (Enter opens it, Esc goes back):", width - 2, curses.A_BOLD)
        rows = [label for _, label in chat.choices] + [ENTRY]
        if not chat.choices:
            screen.addnstr(2, 1, "No models found. Put diffusion GGUFs in ~/.local/share/kilix-chat/models.",
                           width - 2, curses.A_DIM)
        top = max(0, chat.cursor - (body - 4))
        for i, row in enumerate(rows[top:top + body - 3]):
            index = top + i
            style = curses.A_REVERSE if index == chat.cursor else curses.A_NORMAL
            screen.addnstr(3 + i, 2, row, width - 4, style)
    else:
        with chat.lock:
            rows = transcript_lines(chat.messages, chat.partial, width - 2,
                                    chat.model.name if chat.model else "Model")
        end = max(0, len(rows) - chat.scroll)
        visible = rows[max(0, end - body):end]
        styles = {"you": curses.A_BOLD | curses.color_pair(1), "them": curses.A_BOLD | curses.color_pair(2),
                  "text": curses.A_NORMAL, "partial": curses.A_DIM}
        for i, (text, style) in enumerate(visible):
            screen.addnstr(1 + i, 1, text, width - 2, styles[style])
    status = chat.status if not chat.busy else "Generating… Esc stops."
    screen.addnstr(height - 2, 0, (status or HELP)[:width - 1], width - 1, curses.A_DIM)
    prompt = "path> " if chat.typing_path else "> "
    tail = chat.entry[-(width - len(prompt) - 2):]
    screen.addnstr(height - 1, 0, prompt + tail, width - 1)
    screen.move(height - 1, min(width - 1, len(prompt) + len(tail)))


def handle(chat, key):
    """Apply one key. Return False to quit."""
    if key in (curses.KEY_F10, "\x11"):
        chat.cancel.set()
        return False
    if key == "\x1b":
        if chat.busy:
            chat.cancel.set()
        elif chat.typing_path:
            chat.typing_path = False
        elif chat.picking and chat.model:
            chat.picking = False
        return True
    if key == curses.KEY_F2 and not chat.busy:
        chat.open_picker()
        return True
    if key == curses.KEY_F3:
        chat.length = LENGTHS[(LENGTHS.index(chat.length) + 1) % len(LENGTHS)]
        return True
    if key == "\x0e" and not chat.busy:
        chat.messages = []
        chat.status = "New chat."
        return True
    if chat.picking and not chat.typing_path:
        if key == curses.KEY_UP:
            chat.cursor = max(0, chat.cursor - 1)
        elif key == curses.KEY_DOWN:
            chat.cursor = min(len(chat.choices), chat.cursor + 1)
        elif key in ("\n", "\r", curses.KEY_ENTER):
            chat.pick()
        return True
    if key in (curses.KEY_PPAGE, curses.KEY_NPAGE):
        chat.scroll = max(0, chat.scroll + (10 if key == curses.KEY_PPAGE else -10))
    elif key in ("\n", "\r", curses.KEY_ENTER):
        if chat.typing_path:
            if chat.entry.strip() and chat.open(chat.entry.strip()):
                chat.picking = chat.typing_path = False
            chat.entry = ""
        else:
            chat.send()
    elif key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
        chat.entry = chat.entry[:-1]
    elif isinstance(key, str) and key.isprintable():
        chat.entry += key
    return True


def run(spec=None, device=None, max_tokens=256):
    chat = Chat(spec, device, max_tokens)
    if not chat.model:
        status = chat.status
        chat.open_picker()
        chat.status = status

    def main(screen):
        curses.curs_set(1)
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        screen.timeout(100)
        screen.keypad(True)
        while True:
            draw(screen, chat)
            screen.refresh()
            try:
                key = screen.get_wch()
            except curses.error:
                continue
            if key == curses.KEY_RESIZE:
                continue
            if not handle(chat, key):
                break

    curses.wrapper(main)
    if chat.worker:
        chat.worker.join(timeout=5)
    return 0
