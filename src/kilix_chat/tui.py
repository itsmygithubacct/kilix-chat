"""Terminal chat: one conversation with one model at a time.

Diffusion replies are shown as they denoise: the whole reply appears at once
and sharpens step by step, rather than growing word by word.
"""
from __future__ import annotations

import curses
from datetime import datetime
import threading
import textwrap

from kilix_chat import commands, memory
from kilix_chat.diffusion import Cancelled, show_masks
from kilix_chat.models import catalog, load

LENGTHS = (64, 128, 256, 512)
HELP = ("Enter send · /help commands · F2 model · F3 reply length · F4 show denoising · Ctrl-N new chat · "
        "Esc stop · F10 quit")
ENTRY = "Enter a GGUF path or ollama:NAME..."


def wrap(text, width):
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width, replace_whitespace=False, drop_whitespace=True) or [""])
    return lines


def next_length(length):
    """F3: the next standard reply length above `length`, wrapping around."""
    return next((n for n in LENGTHS if n > length), LENGTHS[0])


def transcript_lines(messages, partial, width, model_name, info=None):
    """(text, style) rows for the conversation, newest last.

    `info` is (title, lines) from a command. It shows below the
    conversation and is never sent to the model."""
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
    if info is not None:
        title, lines = info
        rows.append((title, "info_title"))
        rows.extend((row, "info") for line in lines for row in wrap(line, max(10, width - 2)))
        rows.append(("", "text"))
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
        self.length = max_tokens if commands.MIN_LENGTH <= max_tokens <= commands.MAX_LENGTH else 256
        self.steps = None
        self.system = ""
        self.info = None
        self.activity = ""
        self.visual = True
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
            self.model = load(spec, device=self.device)
        except (OSError, RuntimeError, ValueError) as error:
            self.status = f"Could not open the model: {error}"
            return False
        self.messages = []
        self.info = None
        self.status = f"Chatting with {self.model.name} ({self.model.kind})."
        return True

    def switch(self, spec):
        """/model SPEC: open it, or keep the current model if it will not open."""
        if self.open(spec):
            self.picking = self.typing_path = False

    def new_chat(self):
        self.messages = []
        self.info = None
        self.scroll = 0
        self.status = "New chat."

    def show(self, title, lines):
        """Show a command's output below the conversation."""
        self.info = (title, list(lines))
        self.scroll = 0

    def system_prompt(self):
        """The chat's own system prompt followed by the saved memory."""
        return "\n\n".join(part for part in (self.system.strip(), memory.system_text(memory.notes())) if part)

    def conversation(self):
        """What the model is sent: the system prompt, then the turns."""
        system = self.system_prompt()
        return ([{"role": "system", "content": system}] if system else []) + list(self.messages)

    @property
    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def background(self, job, activity):
        """Run job() off the window's thread; the status shows `activity`."""
        self.cancel.clear()
        self.activity = activity
        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def submit(self):
        """Enter: run a command, or send a message. Return False to quit."""
        text = self.entry.strip()
        if commands.is_command(text):
            return commands.run(self, text)
        self.send()
        return True

    def send(self):
        text = self.entry.strip()
        if not text or self.busy:
            return
        if not self.model:
            self.status = "Choose a model first (F2 or /model)."
            return
        if text.startswith("//"):
            text = text[1:]
        self.entry = ""
        self.messages.append({"role": "user", "content": text})
        self.start_reply()

    def start_reply(self):
        """Ask for a reply to the conversation as it stands."""
        self.scroll = 0
        self.info = None
        self.partial = ("", "is thinking…")
        history = self.conversation()
        self.background(lambda: self._reply(history), "Generating… Esc stops.")

    def _progress(self, text, step, steps):
        if step is not None:
            text = show_masks(text)
        with self.lock:
            self.partial = (text, f"(step {step}/{steps})" if step is not None else "")

    def _reply(self, history):
        try:
            reply, _ = self.model.reply(history, max_tokens=self.length, steps=self.steps,
                                        on_partial=self._progress if self.visual else None,
                                        cancel=self.cancel)
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

    def context_report(self):
        """/context: how the conversation fits the model's context."""
        try:
            budget = self.model.budget(self.conversation(), max_tokens=self.length)
        except (OSError, RuntimeError, ValueError) as error:
            with self.lock:
                self.status = f"Could not count the tokens: {error}"
            return
        how = "counted by llama-tokenize" if budget.exact else "estimated"
        free = budget.context - budget.prompt_tokens - budget.reply_tokens
        lines = [f"Context   {budget.context:,} tokens" + (f" ({budget.note})" if budget.note else ""),
                 f"Prompt    {budget.prompt_tokens:,} tokens, {how}; system prompt and memory included",
                 f"Reply     {budget.reply_tokens:,} tokens reserved (/length)",
                 f"Free      {free:,} tokens" if free >= 0 else f"Over      {-free:,} tokens"]
        if budget.dropped:
            lines.append(f"Turns     {budget.turns} in this chat; the oldest {budget.dropped} would be dropped "
                         f"(the whole chat needs {budget.full_tokens:,} tokens)")
        else:
            lines.append(f"Turns     {budget.turns} in this chat; none would be dropped")
        if not budget.fits:
            lines.append("Even the newest turn does not fit: shorten it, or the reply with /length.")
        with self.lock:
            self.show("Context", lines)

    def markdown(self):
        """/save: the chat as a Markdown transcript."""
        name = self.model.name if self.model else "Model"
        parts = [f"# Chat with {name}", "",
                 f"Saved by kilix-chat on {datetime.now():%Y-%m-%d %H:%M}"
                 + (f" · {self.model.kind}" if self.model else "") + f" · replies up to {self.length} tokens", ""]
        if self.system_prompt():
            parts += ["## System prompt", "", self.system_prompt(), ""]
        for message in self.messages:
            parts += ["## You" if message["role"] == "user" else f"## {name}", "", message["content"], ""]
        return "\n".join(parts)

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
    if chat.steps:
        title += f" in {chat.steps} steps"
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
                                    chat.model.name if chat.model else "Model", chat.info)
        end = max(0, len(rows) - chat.scroll)
        if chat.partial is not None and chat.partial[1].startswith("(step"):
            # A denoising reply changes everywhere at once, and its end is
            # padding that is settled last. Keep its start in view.
            start = max(i for i, (_, style) in enumerate(rows) if style == "them")
            end = max(0, min(len(rows), start + body) - chat.scroll)
        visible = rows[max(0, end - body):end]
        styles = {"you": curses.A_BOLD | curses.color_pair(1), "them": curses.A_BOLD | curses.color_pair(2),
                  "text": curses.A_NORMAL, "partial": curses.A_DIM,
                  "info_title": curses.A_BOLD | curses.color_pair(3), "info": curses.color_pair(3)}
        for i, (text, style) in enumerate(visible):
            screen.addnstr(1 + i, 1, text, width - 2, styles[style])
    status = chat.status if not chat.busy else chat.activity
    if commands.is_command(chat.entry) and not chat.typing_path:
        status = commands.hint(chat.entry)
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
        elif chat.info is not None and not chat.picking:
            chat.info = None
        elif chat.typing_path:
            chat.typing_path = False
        elif chat.picking and chat.model:
            chat.picking = False
        return True
    if key == curses.KEY_F2 and not chat.busy:
        chat.open_picker()
        return True
    if key == curses.KEY_F4:
        commands.lookup("visual").run(chat, "")
        return True
    if key == curses.KEY_F3:
        chat.length = next_length(chat.length)
        return True
    if key == "\x0e" and not chat.busy:
        chat.new_chat()
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
            return chat.submit()
    elif key == "\t":
        chat.entry = commands.complete(chat.entry)
    elif key == "\x15":
        chat.entry = ""
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
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
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
