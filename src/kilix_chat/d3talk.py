"""d3-talk: let a decision model speak.

Ollama's decision models (d3:2b and its larger siblings) cannot write text.
They only answer questions about text: yes or no, one of these choices, or a
score. d3-talk turns that into a reply by always handing the model a list of
choices, most-used first.

  think    The most-used words are swept in frequency order. The model first
           picks, from lines of ten, the lines that hold a word a good
           answer would name, and then picks those words from the best
           lines. The survivors are the topic words.
  compose  The reply grows one word at a time. Each step offers three groups
           of choices: the words that most often follow the last word, the
           most-used grammar words, and the topic words. The group winners
           meet in a final. A sentence has to open on a grammar word, or it
           reads like a headline.
  compare  Two partial replies are kept alive at once. After every word the
           model ranks them as whole sentences, which it does far better
           than it picks single words, and a sentence is done as soon as the
           best one has ended. A second sentence is kept only if the model
           prefers the reply with it.
  learn    Every word of the finished reply is counted with the word before
           it. Pairs the model uses often are offered earlier next time.

The unit of choice is a setting. Words are the default. With letters, each
step picks the next letter of the word being written, or that the word ends
there; with tokens, the next piece of the word as d3's own tokenizer cuts it.
Both offer only what continues some word of the lexicon or of the user's
message, so a name or a number from the message can be spelled. The mixed
unit offers whole words, tokens and letters together at every step. An
unfinished word is shown to the model with a hyphen after it ("It is Par-").

Word and word-pair frequencies come from Peter Norvig's lists
(norvig.com/ngrams). They are downloaded once, on first use, into
`$XDG_DATA_HOME/kilix-chat/d3-talk/`, where the learned usage is kept too.
"""
from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
import functools
import itertools
from http.client import HTTPException
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from kilix_chat import memory, ollama
from kilix_chat.diffusion import Cancelled, Reply
from kilix_chat.prompt import text_of

SOURCES = {"count_1w.txt": "https://norvig.com/ngrams/count_1w.txt",
           "count_2w.txt": "https://norvig.com/ngrams/count_2w.txt"}
# d3's tokenizer (that of Qwen3.5, its base model), needed only for the token
# unit. Its merges alone define every token.
TOKENIZER = {"d3-merges.txt": "https://huggingface.co/vllm-sr/d3-nano/resolve/main/merges.txt"}
# Spellings and capitals come from the system word list when there is one.
# KILIX_CHAT_WORD_LIST names another.
WORD_LIST = "/usr/share/dict/words"

START = "<S>"              # stands for the start of a sentence
STOPS = ".?!"
PUNCT = ".,?!"
UNIGRAMS_READ = 60000      # lines of count_1w.txt to read
GROUP = 14                 # choices per question while composing
SWEEP_LINE = 10            # words on a line in the topic sweep's first pass
SWEEP_GROUP = 100          # lines, then words, per question in the topic sweep
SWEEP_WORDS = 320          # words that go on from the best lines to the second pass
SWEEP_KEEP = 12            # words that survive each question of the second pass
HISTORY_CHARS = 1200       # earlier turns repeated in every question
UNITS = ("words", "tokens", "letters", "mixed")
PARTIAL = "-"              # shown after a word that is not finished yet
LONGEST = 24               # most letters in a word that is spelled
FALLBACK_PIECE = 3         # letters per piece when d3's tokenizer cannot be had
TOPIC_WEIGHT = 0.05        # how much likelier the best topic word's letters and tokens are offered
WORD_WEIGHT = 0.1          # how much likelier the words the word unit would offer are spelled
TOKEN_CHOICES = 3 * GROUP - 2  # most tokens offered at one step (a word can open on thousands), as many as words
NOTES_CHARS = 400          # system prompt and memory repeated in every question

# First words of a sentence, most-used first. The corpus is web text and its
# sentence starts are poor, so these seed the list until usage reorders it.
STARTERS = ("the i it you this there that yes no we they a in if my he she what when to for because most "
            "some many people one").split()

# Grammar words: the small closed set that holds sentences together. They are
# offered in order of use, as their own group of choices.
GRAMMAR = frozenset(
    "the of and to a in for is on that by this with i you it not or be are from at as your all have an was "
    "we will can about if my has but our one do no they he up may what which their out so there when who "
    "how why would could should been were had more some any these those than then them his her she me us "
    "into also just only very most many much its am does did because while where here yes too over after "
    "before each such like".split())

# The only one- and two-letter words that are words.
SHORT = frozenset("a i of to in is on by it or be at as an we us if my do no he up so am me go oh ok hi".split())

# Web text is full of these. They are not words anyone says.
JUNK = frozenset("www com http https html htm org net edu gov etc inc ltd usa php faq url rss pdf jpg gif "
                 "sex porn xxx nude".split())

# Contractions are missing from the corpus. Each borrows the followers of the
# word it ends in, and is offered after the listed words.
CONTRACTIONS = {
    "don't": ("not", "i you we they"),
    "can't": ("not", "i you we they he she it"),
    "won't": ("not", "i you we they he she it"),
    "doesn't": ("not", "he she it that this"),
    "isn't": ("not", "he she it that this"),
    "didn't": ("not", "i you we they he she it"),
    "i'm": ("am", START),
    "it's": ("is", START),
    "that's": ("is", START),
    "you're": ("are", START),
}

VOICE = 'The assistant calls itself "I" and calls the user "you".'
TOPIC_QUESTION = ("Which of these words is most likely one of the specific things, facts or details named in "
                  "that answer?")
LINE_QUESTION = ("Which of these lines holds a word that is most likely one of the specific things, facts or "
                 "details named in that answer?")
STEP_QUESTION = "Which option continues the reply with correct English grammar and good sense?"
# How each unit is described to the model while it composes.
UNIT_TEXT = {"tokens": "written a few letters at a time",
             "letters": "written one letter at a time",
             "mixed": "written a word or a few letters at a time"}
BEGINNING_QUESTION = ("Which option is the best beginning of a full spoken sentence, with a subject and a verb, "
                      "that answers the user correctly? It must not read like a title or heading.")
REPLY_QUESTION = ("Which reply is the best: full, correct, natural sentences that a person would say out loud, "
                  "not a title or heading?")


def data_dir():
    return memory.data_dir() / "d3-talk"


def write_atomically(target, text):
    """Write `text` to a sibling with a name of its own, then rename it over
    `target`, so a reader never sees half a file and two writers never share one."""
    handle, part = tempfile.mkstemp(prefix=f".{target.name}-", suffix=".part", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
        os.replace(part, target)
    except BaseException:
        Path(part).unlink(missing_ok=True)
        raise


def fetch(directory, sources=None):
    """Download the frequency lists (or other `sources`) that `directory` does not hold yet."""
    for name, url in (sources or SOURCES).items():
        target = directory / name
        if target.exists():
            continue
        part = None
        try:
            directory.mkdir(parents=True, exist_ok=True)
            handle, part = tempfile.mkstemp(prefix=f".{name}-", suffix=".part", dir=directory)
            with urlopen(Request(url, headers={"User-Agent": "curl"}), timeout=120) as response, \
                    os.fdopen(handle, "wb") as out:
                shutil.copyfileobj(response, out)
            os.replace(part, target)
        except (OSError, URLError, HTTPException) as error:
            # A download cut short raises IncompleteRead, which is no OSError.
            if part:
                Path(part).unlink(missing_ok=True)
            if sources:
                raise RuntimeError(f"d3-talk could not download {url} ({error}). "
                                   f"Put {name} in {directory}.") from error
            raise RuntimeError(f"d3-talk needs word frequencies and could not download {url} ({error}). "
                               f"Put count_1w.txt and count_2w.txt in {directory}.") from error


def byte_letters():
    """The byte-level BPE alphabet: each byte is one printable character."""
    shown = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    letters, extra = {}, 0
    for byte in range(256):
        if byte in shown:
            letters[byte] = chr(byte)
        else:
            letters[byte] = chr(256 + extra)
            extra += 1
    return letters


# The tokenizer's own split of text into chunks, for the text a word can hold
# (its \p{L} and \p{N} classes become their ASCII-and-Unicode \w equivalents).
CHUNKS = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?[^\W\d_]+|\d| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


class Tokenizer:
    """d3's byte-level BPE, read from its merges: how it cuts a word into tokens."""
    name = "d3's tokenizer"

    def __init__(self, path):
        self.rank = {}
        with open(path, encoding="utf-8") as rows:
            for row in rows:
                if row.startswith("#version"):
                    continue
                first, _, second = row.rstrip("\n").partition(" ")
                if first and second:
                    self.rank.setdefault((first, second), len(self.rank))
        if not self.rank:
            raise ValueError(f"{path} holds no merges")
        self.letter = byte_letters()
        self.byte = {letter: byte for byte, letter in self.letter.items()}
        self.whole = {first + second for first, second in self.rank}     # every token longer than a byte

    def merge(self, chunk):
        parts, rank = [self.letter[byte] for byte in chunk.encode("utf-8")], self.rank
        if "".join(parts) in self.whole:
            return [chunk]      # most words are one token (checked against the reference tokenizer)
        while len(parts) > 1:
            best = place = None
            for here in range(len(parts) - 1):
                order = rank.get((parts[here], parts[here + 1]))
                if order is not None and (best is None or order < best):
                    best, place = order, here
            if best is None:
                break
            parts[place:place + 2] = [parts[place] + parts[place + 1]]
        return [bytes(self.byte[letter] for letter in part).decode("utf-8", "replace") for part in parts]

    def pieces(self, text):
        """`text` as the tokenizer cuts it: ' Paris' -> [' Paris']."""
        return [piece for chunk in CHUNKS.findall(text) for piece in self.merge(chunk)]


class Pieces:
    """The stand-in when d3's tokenizer cannot be had: pieces of a few letters."""
    name = f"pieces of {FALLBACK_PIECE} letters"

    def pieces(self, text):
        word = text.lstrip(" ")
        return [word[start:start + FALLBACK_PIECE] for start in range(0, len(word), FALLBACK_PIECE)]


def word_pieces(tokenizer, word, shown):
    """The tokens of `word` (lower case) after a space, as lower-case pieces.

    `shown` is how the word is written ("Paris"), which is what the model
    reads. A word the tokenizer does not cut along its letters is one piece."""
    pieces = tokenizer.pieces(" " + shown)
    if pieces:
        pieces[0] = pieces[0].lstrip(" ")
    pieces = [piece for piece in pieces if piece]
    if sum(map(len, pieces)) != len(word):
        return (word,)
    out, start = [], 0
    for piece in pieces:
        out.append(word[start:start + len(piece)])
        start += len(piece)
    return tuple(out)


class Lexicon:
    """Word frequencies, word-pair frequencies, and what the model itself has used."""

    def __init__(self, directory):
        self.directory = Path(directory)
        fetch(self.directory)
        lower, capital = set(), {}
        try:
            with open(os.environ.get("KILIX_CHAT_WORD_LIST") or WORD_LIST,
                      encoding="utf-8", errors="replace") as words:
                for line in words:
                    word = line.strip()
                    if not word.isalpha():
                        continue
                    if word.islower():
                        lower.add(word)
                    elif word.istitle():
                        capital.setdefault(word.lower(), word)
        except OSError:
            pass    # no word list: every alphabetic corpus word counts, and none is capitalised
        checked = bool(lower)
        # "paris" is only ever "Paris"; "may" stays lowercase.
        self.proper = {word: shown for word, shown in capital.items() if word not in lower}

        self.count = {}     # in order of use
        with open(self.directory / "count_1w.txt", encoding="utf-8") as rows:
            for number, row in enumerate(rows):
                if number >= UNIGRAMS_READ:
                    break
                word, _, count = row.partition("\t")
                if word in JUNK or not word.isalpha() or (len(word) <= 2 and word not in SHORT):
                    continue
                if not checked or word in lower or word in self.proper:
                    self.count[word] = int(count)
        self.words = list(self.count)
        self.total = sum(self.count.values()) or 1
        self.grammar = [word for word in self.words if word in GRAMMAR]

        self.after = {}     # which words follow which, and how often
        with open(self.directory / "count_2w.txt", encoding="utf-8") as rows:
            for row in rows:
                pair, _, count = row.partition("\t")
                first, _, second = pair.lower().partition(" ")
                if first in self.count and second in self.count:
                    table = self.after.setdefault(first, {})
                    table[second] = table.get(second, 0) + int(count)
        # Sentence starts: the seeded list, shaped like a frequency table.
        self.after[START] = {word: len(STARTERS) - place for place, word in enumerate(STARTERS)}
        self.after_total = {word: sum(table.values()) for word, table in self.after.items()}

        self.offer_after = {}
        for contraction, (_, befores) in CONTRACTIONS.items():
            for before in befores.split():
                self.offer_after.setdefault(before, []).append(contraction)

        # What the model has used: (word before, word) -> [times, sum of confidence].
        self._use(self.saved_usage())
        self._tokenizer = None
        self._pieces = {}

    def tokenizer(self):
        """d3's tokenizer, downloaded on first use; three-letter pieces if it cannot be had.

        Returns (tokenizer, why the stand-in is used or None)."""
        if self._tokenizer is None:
            try:
                fetch(self.directory, TOKENIZER)
                self._tokenizer = (Tokenizer(self.directory / next(iter(TOKENIZER))), None)
            except (OSError, RuntimeError, ValueError) as error:
                self._tokenizer = (Pieces(), str(error))
        return self._tokenizer

    def pieces(self, word, shown=None):
        """The tokens `word` is written with, as lower-case pieces: 'paris' -> ('paris',)."""
        if word not in self._pieces:
            self._pieces[word] = word_pieces(self.tokenizer()[0], word,
                                             shown or self.proper.get(word, word))
        return self._pieces[word]

    @functools.cached_property
    def lower_stems(self):
        """Every beginning of a word that is written in lower case."""
        return {word[:end] for word in itertools.chain(self.count, CONTRACTIONS) if word not in self.proper
                for end in range(1, len(word))}

    def usage_path(self):
        return self.directory / "usage.tsv"

    def saved_usage(self):
        """The usage as saved: {(word before, word): [times, sum of confidence]}.

        A row that does not parse is skipped, not the rest of the file, which
        the next save would otherwise drop."""
        saved = {}
        try:
            with open(self.usage_path(), encoding="utf-8") as rows:
                for row in rows:
                    parts = row.rstrip("\n").split("\t")
                    try:
                        times, confidence = int(parts[2]), float(parts[3])
                    except (IndexError, ValueError):
                        continue
                    if len(parts) == 4 and times > 0:
                        entry = saved.setdefault((parts[0], parts[1]), [0, 0.0])
                        entry[0] += times
                        entry[1] += confidence
        except OSError:
            pass
        return saved

    def _use(self, used):
        self.used, self.used_after, self.used_word = used, {}, {}
        for (before, word), (times, _) in used.items():
            self.used_after[before] = self.used_after.get(before, 0) + times
            self.used_word[word] = self.used_word.get(word, 0) + times

    def learn(self, steps):
        """Count a finished reply: [(word before, word, confidence), ...].

        The counts are added to the file as it is now, so replies learned by
        another kilix-chat or d3-talk since this one read it are kept. Return
        whether they were saved; a failed save never costs the reply."""
        used = self.saved_usage()
        for before, word, confidence in steps:
            if not (self.known(word) or word in PUNCT) or not (self.known(before) or before == START):
                continue    # a spelled word, or one from the message: never offered as a word later
            entry = used.setdefault((before, word), [0, 0.0])
            entry[0] += 1
            entry[1] += confidence
        self._use(used)
        try:
            write_atomically(self.usage_path(), "".join(
                f"{before}\t{word}\t{times}\t{confidence:.2f}\n"
                for (before, word), (times, confidence) in sorted(used.items())))
        except OSError:
            return False
        return True

    def followers(self, before, limit):
        """Words likely to follow `before`, most likely first: how often the
        pair occurs in the corpus, plus how often the model has used it."""
        key = CONTRACTIONS[before][0] if before in CONTRACTIONS else before
        total = self.after_total.get(key) or 1
        score = {word: count / total for word, count in self.after.get(key, {}).items()}
        mine = self.used_after.get(before, 0)
        if mine:
            for (first, word), (times, _) in self.used.items():
                if first == before and word not in PUNCT:
                    # Learned use counts for half, and the +20 keeps a few
                    # lucky uses from outranking the whole corpus.
                    score[word] = score.get(word, 0) + 0.5 * times / (mine + 20)
        ranked = sorted(score, key=lambda word: -score[word])[:limit]
        for contraction in self.offer_after.get(before, []):
            if contraction not in ranked:
                ranked.insert(min(len(ranked), 4), contraction)
        return ranked

    def common(self, limit):
        """Grammar words, most used first, counting the model's own use of them."""
        mine = sum(self.used_word.get(word, 0) for word in self.grammar) + 20
        return sorted(self.grammar, key=lambda word: -(self.count[word] / self.total
                                                       + 0.5 * self.used_word.get(word, 0) / mine))[:limit]

    def known(self, word):
        return word in self.count or word in CONTRACTIONS


@functools.lru_cache(maxsize=2)
def _lexicon(directory):
    return Lexicon(directory)


def lexicon():
    """The lexicon for this user's data directory, read once per process."""
    return _lexicon(str(data_dir()))


def render(words, proper=None):
    """['it', 'is', 'paris', '.'] -> 'It is Paris.'"""
    out = []
    for place, word in enumerate(words):
        if word in PUNCT:
            if out:
                out[-1] += word
            continue
        # "a apple" -> "an apple", "an book" -> "a book"
        if word in ("a", "an") and place + 1 < len(words) and words[place + 1] not in PUNCT:
            word = "an" if words[place + 1][0] in "aeiou" else "a"
        if word == "i" or word.startswith("i'"):
            word = "I" + word[1:]
        word = (proper or {}).get(word, word)
        if not out or out[-1][-1] in STOPS:
            word = word[0].upper() + word[1:]
        out.append(word)
    return " ".join(out)


def partial(word):
    """Whether `word` is a word not finished yet ("par-")."""
    return len(word) > 1 and word.endswith(PARTIAL)


def written(steps):
    """The words of a reply from its steps, [(word or unfinished word, odds), ...].

    A step after an unfinished word replaces it: "p-", "pa-", "paris"."""
    words = []
    for word, _ in steps:
        if words and partial(words[-1]):
            words[-1] = word
        else:
            words.append(word)
    return words


def word_steps(steps):
    """[(word, confidence), ...] for the finished words of `steps`. A spelled
    word's confidence is the mean of the choices that wrote it."""
    out, spelled = [], []
    for word, odds in steps:
        spelled.append(odds)
        if not partial(word):
            out.append((word, sum(spelled) / len(spelled)))
            spelled = []
    return out


def sentence_words(words):
    """The words of the sentence being written: those after the last full stop."""
    for place in range(len(words) - 1, -1, -1):
        if words[place] in STOPS:
            return words[place + 1:]
    return words


def word_before(words):
    """The word whose followers apply next, or START at a sentence start."""
    if not words or words[-1] in STOPS:
        return START
    return next((word for word in reversed(words) if word not in PUNCT), START)


def split(messages):
    """(notes, earlier turns, the message to answer) from a conversation.

    The notes are the system messages. The earlier turns are (user, assistant)
    pairs, oldest first."""
    notes, turns, said = [], [], None
    for message in messages:
        role, text = message.get("role"), text_of(message.get("content")).strip()
        if role in ("system", "developer"):
            notes.append(text)
        elif role == "user":
            # Two user messages in a row are one thing to answer.
            said = f"{said}\n{text}" if said else text
        elif role == "assistant" and said is not None:
            turns.append((said, text))
            said = None
    if not said:
        raise ValueError("d3-talk needs a message from the user to answer.")
    return "\n".join(note for note in notes if note), turns, said


@dataclass
class Settings:
    beam: int = 2            # partial replies kept alive at once
    sentences: int = 2       # most sentences in a reply
    max_words: int = 20      # most words in a sentence
    vocab: int = 4000        # most-used words swept for topic words; 0 skips the sweep
    topics: int = 24         # topic words kept from the sweep
    judge: str | None = None  # a larger decision model that picks the final reply
    learn: bool = True       # count the words of every finished reply
    unit: str = "words"      # what one choice adds: words, tokens, letters or mixed
    any_letter: bool = False  # also offer letters that continue no known word


class Inventory:
    """Words with weights: shared `base` weights, plus `extra` weight for
    some, less the `banned` words."""

    def __init__(self, base, extra, banned):
        self.base, self.extra, self.banned = base, extra, banned

    def get(self, word):
        if word in self.banned or len(word) > LONGEST or (word not in self.base and word not in self.extra):
            return None
        return self.base.get(word, 0.0) + self.extra.get(word, 0.0)

    def __contains__(self, word):
        return self.get(word) is not None

    def __getitem__(self, word):
        weight = self.get(word)
        if weight is None:
            raise KeyError(word)
        return weight

    def __iter__(self):
        return (word for word in itertools.chain(self.base, (word for word in self.extra if word not in self.base))
                if word in self)


class Talker:
    """Writes one reply by asking `model` multiple-choice questions."""

    def __init__(self, model, lexicon, settings=None, *, on_text=None, on_detail=None, cancel=None):
        self.model = model
        self.lexicon = lexicon
        self.settings = settings or Settings()
        self.on_text = on_text          # called with the reply as it stands
        self.on_detail = on_detail      # called with a line about the choices
        self.cancel = cancel
        self.questions = 0
        self.tokens = 0
        self.words = []                 # the finished reply, word by word
        self.steps = 0                  # choices made while composing, of any unit
        self.proper = lexicon.proper    # how words are written, with the message's own names
        self.extra = {}                 # the message's words that the lexicon lacks, as written there
        self._inventory = {}
        self._capitals = None
        self._sorted = None             # every word offered so far, in order, for finding a stem's words
        self._bounds = {}
        self._base = None
        self._openings = {}

    def choose(self, state, question, options, model=None):
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()
        ranked, tokens = ollama.decide(model or self.model, state, question, options)
        self.questions += 1
        self.tokens += tokens
        return ranked

    def show(self, words):
        if words and partial(words[-1]):
            # An unfinished word that only names can finish is shown with their capital: "Par-" in Paris.
            stem = words[-1][:-1]
            shown = self.capitals().get(stem)
            if shown:
                words = words[:-1] + [shown + PARTIAL]
        return render(words, self.proper)

    def capitals(self):
        """{stem: stem as written} for the stems that only capitalised words begin with."""
        if self._capitals is None:
            known = self.lexicon.lower_stems
            lower = {word[:end] for word in self.extra if word not in self.proper for end in range(1, len(word))}
            self._capitals = {}
            for word, shown in self.proper.items():
                for end in range(1, len(word)):
                    stem = word[:end]
                    if stem not in known and stem not in lower and shown[:end].lower() == stem:
                        self._capitals[stem] = shown[:end]
        return self._capitals

    def detail(self, line):
        if self.on_detail:
            self.on_detail(line)

    def conversation(self, message, turns, notes):
        """What every question says about the chat. It is read again for
        each question, so it is kept short."""
        lines = []
        for said, replied in turns:
            lines += [f'The user said: "{said}"', f'The assistant replied: "{replied}"']
        past = "\n".join(lines)[-HISTORY_CHARS:]
        text = f"Notes for the assistant: {notes[:NOTES_CHARS]}\n\n" if notes else ""
        if past:
            text += "Earlier in the conversation:\n" + past + "\n\n"
        return text + f'The user said: "{message}"'

    # think: find the topic words

    def think(self, message, context):
        lexicon, settings = self.lexicon, self.settings
        # The user's own content words always count as topic words.
        asked = []
        for word in re.findall(r"[a-z']+", message.lower()):
            if lexicon.known(word) and word not in GRAMMAR and len(word) > 2 and word not in asked:
                asked.append(word)
        if settings.vocab <= 0:
            return asked
        state = (context + "\nA knowledgeable assistant answers directly in one sentence, giving the actual "
                 "answer and its specific details.")
        candidates = [word for word in lexicon.words[:settings.vocab] if len(word) > 2 and word not in GRAMMAR]

        def progress(picked, earlier):
            if self.on_text:    # the newest picks first: the line may be cut short
                self.on_text("(choosing topic words) " + " ".join((picked + earlier[::-1])[:24]))

        # Pass one: whole lines of words, in frequency order. A line costs a
        # quarter of what its words cost one by one, and most lines hold nothing.
        lines = [" ".join(candidates[start:start + SWEEP_LINE]) for start in range(0, len(candidates), SWEEP_LINE)]
        share = math.ceil(SWEEP_WORDS / SWEEP_LINE / math.ceil(len(lines) / SWEEP_GROUP)) if lines else 0
        if len(lines) > share:
            kept = []
            for start in range(0, len(lines), SWEEP_GROUP):
                ranked = self.choose(state, LINE_QUESTION, dict.fromkeys(lines[start:start + SWEEP_GROUP], ""))
                picked = [line for line, _ in ranked[:share]]
                progress(" ".join(picked).split(), " ".join(kept).split())
                kept += picked
            candidates = " ".join(kept).split()
        # Pass two: the words of the best lines, a group at a time, then a final.
        survivors = []
        for start in range(0, len(candidates), SWEEP_GROUP):
            ranked = self.choose(state, TOPIC_QUESTION, dict.fromkeys(candidates[start:start + SWEEP_GROUP], ""))
            picked = [word for word, _ in ranked[:SWEEP_KEEP]]
            progress(picked, survivors)
            survivors += picked
        found = survivors[:48]
        if len(found) > 1:
            found = [word for word, _ in self.choose(state, TOPIC_QUESTION, dict.fromkeys(found, ""))]
        found = found[:settings.topics]
        self.detail("topic words: " + " ".join(found))
        return found + [word for word in asked if word not in found]

    # compose: offer the next word

    def step_state(self, context, words, topics):
        unit = self.settings.unit
        if unit != "words":
            return (context + "\nThe assistant is answering in complete, grammatical English sentences, "
                    + UNIT_TEXT[unit] + f'. A word ending in "{PARTIAL}" is not finished yet. ' + VOICE
                    + f"\nKey words the answer may need: {', '.join(topics[:10])}."
                    + f'\nReply so far: "{self.show(words)}"')
        return (context + "\nThe assistant is answering in complete, grammatical English sentences, written "
                "one word at a time. " + VOICE
                + f"\nKey words the answer may need: {', '.join(topics[:10])}."
                + f'\nReply so far: "{self.show(words)}"')

    def step_options(self, words, group):
        """Each choice is shown as the reply with that word added. A choice
        made inside a word takes the unfinished word's place."""
        if words and partial(words[-1]):
            words = words[:-1]
        tail, lead = words, ""
        if len(words) > 12:
            tail, lead = words[-9:], "... "
            while tail and tail[0] in PUNCT:
                tail = tail[1:]
        return {word: lead + self.show(tail + [word]) + ("" if word in STOPS or partial(word) else " ...")
                for word in group}

    def groups(self, words, topics):
        """Up to three lists of choices, each in order of likely use."""
        lexicon = self.lexicon
        sentence = sentence_words(words)
        before = word_before(words)
        # No word twice in a row, no repeated pair, and no word a third time
        # in one sentence ("meat and bone and fish and ...").
        banned = set(words[-1:])
        banned |= {second for first, second in zip(words, words[1:]) if first == before}
        banned |= {word for word in sentence if sentence.count(word) >= 2}
        follow = [word for word in lexicon.followers(before, GROUP + 4) if word not in banned][:GROUP]
        common = [word for word in lexicon.common(40) if word not in banned and word not in follow][:GROUP - 2]
        topic = [word for word in topics if word not in words and word not in follow and word not in common][:GROUP]
        if len(sentence) >= 3:
            common.append(".")
        if before == START:
            # A sentence that opens on a topic word reads like a headline
            # ("Introduction to ocean."), so it has to open on grammar.
            topic = []
        return [group for group in (follow, common, topic) if group]

    # compose by tokens and letters

    def inventory(self, done, topics):
        """The words the next one can be, each with a weight: every known
        word, the message's own words and the topic words, or at a sentence
        start only the words a sentence may open on."""
        key = (tuple(done), tuple(topics))
        if key not in self._inventory:
            if len(self._inventory) >= 8:       # a few partial replies' worth
                self._inventory.pop(next(iter(self._inventory)))
            self._inventory[key] = self._weigh(done, topics)
        return self._inventory[key]

    def _weigh(self, done, topics):
        lexicon = self.lexicon
        sentence = sentence_words(done)
        before = word_before(done)
        banned = set(done[-1:])
        banned |= {second for first, second in zip(done, done[1:]) if first == before}
        banned |= {word for word in sentence if sentence.count(word) >= 2}
        key = CONTRACTIONS[before][0] if before in CONTRACTIONS else before
        total = lexicon.after_total.get(key) or 1
        if before == START:
            opening = set(STARTERS) | set(lexicon.grammar) | set(lexicon.offer_after.get(START, []))
            base, extra = {word: lexicon.count.get(word, 0) / lexicon.total for word in opening}, {}
        else:
            base = self.base()
            # The best topic words count most, as many as the word unit offers in its topic group.
            extra = {word: TOPIC_WEIGHT * (GROUP - place) / GROUP for place, word in enumerate(topics[:GROUP])}
            for word in self.extra:
                extra[word] = TOPIC_WEIGHT
        for word, count in lexicon.after.get(key, {}).items():
            if word in base or word in extra:
                extra[word] = extra.get(word, 0.0) + count / total
        for contraction in lexicon.offer_after.get(before, []):
            if contraction in base:
                extra[contraction] = extra.get(contraction, 0.0) + 1 / len(STARTERS)
        # The words the word unit would offer here come first, so their letters
        # and tokens do: no smaller unit offers less than words do.
        for word in itertools.chain.from_iterable(self.groups(done, topics)):
            if word in base or word in extra:
                extra[word] = extra.get(word, 0.0) + WORD_WEIGHT
        return Inventory(base, extra, banned)

    def base(self):
        """Every known word, weighted by how often it is used."""
        if self._base is None:
            lexicon = self.lexicon
            self._base = {word: count / lexicon.total for word, count in lexicon.count.items()
                          if len(word) <= LONGEST}
            self._base.update((word, 0.0) for word in CONTRACTIONS if word not in self._base)
        return self._base

    def grown(self, word, length, unit):
        """`word` cut one letter or one token past its first `length`
        letters, or None if its tokens do not break there."""
        if unit == "letters":
            return word[:length + 1]
        ends = self.bounds(word)
        if length not in ends:
            return None
        return word[:ends[ends.index(length) + 1]]

    def continuations(self, done, stem, topics, unit, limit=None):
        """What can follow the unfinished word `stem` ("" at a word's start)
        by one letter or one token, likeliest first: [outcome, ...], the
        first `limit` of them.

        An outcome is the word itself when that unit ends the only word it
        can lead to, and otherwise the unfinished word ("par-")."""
        weights = self.inventory(done, topics)
        if stem or weights.base is not self._base:
            nexts = {}      # what the unit leads to -> [weight, how many words it can end as]
            for word in self.starting(stem, weights):
                grown = self.grown(word, len(stem), unit) if len(word) > len(stem) else None
                if grown:
                    entry = nexts.setdefault(grown, [0.0, 0])
                    entry[0] += weights[word]
                    entry[1] += 1
        else:
            # A word's start, from the whole lexicon: its sums are made once, and
            # only what this step adds or bans is counted again.
            table, ranked = self.openings(unit)
            nexts = {}
            for word, weight in itertools.chain(((word, -self._base[word]) for word in weights.banned
                                                 if word in self._base),
                                                weights.extra.items()):
                if word in weights.banned and weight >= 0 or len(word) > LONGEST:
                    continue
                grown = self.grown(word, 0, unit)
                entry = nexts.get(grown) or nexts.setdefault(grown, list(table.get(grown, (0.0, 0))))
                entry[0] += weight
                if weight < 0:
                    entry[1] -= 1
                elif word not in self._base:
                    entry[1] += 1
            # Only the changed sums can climb past the others, so the best
            # `limit` are among the changed and as many unchanged ones as there are changes.
            reach = len(ranked) if limit is None else limit + len(nexts)
            for grown in ranked[:reach]:
                if grown not in nexts:
                    nexts[grown] = table[grown]
            nexts = {grown: entry for grown, entry in nexts.items() if entry[1] > 0}
        if self.settings.any_letter and unit == "letters" and word_before(done) != START:
            for letter in "abcdefghijklmnopqrstuvwxyz0123456789":
                nexts.setdefault(stem + letter, [0.0, 2])
        best = sorted(nexts.items(), key=lambda item: (-item[1][0], item[0]))[:limit]
        return [grown if count == 1 and grown in weights else grown + PARTIAL for grown, (_, count) in best]

    def openings(self, unit):
        """({first letter or token: [summed weight, words]} over every known
        word, those letters or tokens heaviest first)."""
        if unit not in self._openings:
            table = {}
            for word, weight in self.base().items():
                entry = table.setdefault(self.grown(word, 0, unit), [0.0, 0])
                entry[0] += weight
                entry[1] += 1
            self._openings[unit] = (table, sorted(table, key=lambda grown: (-table[grown][0], grown)))
        return self._openings[unit]

    def starting(self, stem, weights):
        """The words of `weights` that begin with `stem`."""
        if not stem:
            return list(weights)
        words = set(weights.base) | set(weights.extra)
        if self._sorted is None or not self._sorted[1].issuperset(words):
            known = (self._sorted[1] if self._sorted else set()) | words
            self._sorted = (sorted(known), known)
        ordered = self._sorted[0]
        start = bisect.bisect_left(ordered, stem)
        end = bisect.bisect_left(ordered, stem + "\U0010ffff")
        return [word for word in ordered[start:end] if word in weights]

    def bounds(self, word):
        """Where the tokens of `word` begin, and its length: 'photosynthesis' -> (0, 6, 14)."""
        ends = self._bounds.get(word)
        if ends is None:
            ends = [0]
            for piece in self.word_pieces(word)[:-1]:
                ends.append(ends[-1] + len(piece))
            ends = self._bounds[word] = tuple(ends) + (len(word),)
        return ends

    def word_pieces(self, word):
        return self.lexicon.pieces(word, self.proper.get(word, word))

    def unit_groups(self, words, topics):
        """Up to a few lists of choices for the next letter, token or word."""
        unit = self.settings.unit
        stem = words[-1][:-1] if words and partial(words[-1]) else ""
        done = words[:-1] if stem else words
        weights = self.inventory(done, topics)
        # the word ends here (any spelling may, once letters that spell nothing are offered)
        ending = [stem] if stem and (stem in weights or self.settings.any_letter) else []
        stop = ["."] if not stem and len(sentence_words(done)) >= 3 else []
        if unit in ("letters", "tokens"):
            nexts = [option for option in self.continuations(done, stem, topics, unit,
                                                             TOKEN_CHOICES + 1 if unit == "tokens" else None)
                     if option not in ending]
            options = ending + (nexts[:TOKEN_CHOICES] if unit == "tokens" else nexts) + stop
            return [options[start:start + GROUP] for start in range(0, len(options), GROUP)]
        # mixed: whole words, then tokens, then letters
        if stem:
            whole = sorted(self.starting(stem, weights), key=lambda word: -weights[word])
            groups = [(ending + [word for word in whole if word not in ending])[:GROUP]]
        else:
            groups = self.groups(done, topics)
        offered = {option for group in groups for option in group}
        for kind in ("tokens", "letters"):
            group = [option for option in self.continuations(done, stem, topics, kind, GROUP + len(offered))
                     if option not in offered][:GROUP]
            offered.update(group)
            groups.append(group)
        return [group for group in groups if group]

    def expand(self, context, beam, topics):
        """The best few next words for one partial reply: [(word, odds), ...]."""
        if self.settings.unit != "words":
            return self.expand_units(context, beam, topics)
        state = self.step_state(context, beam, topics)
        finalists = []
        for group in self.groups(beam, topics):
            for word, _ in self.choose(state, STEP_QUESTION, self.step_options(beam, group))[:2]:
                if word not in finalists:
                    finalists.append(word)
        ranked = self.choose(state, STEP_QUESTION, self.step_options(beam, finalists))
        self.detail(f"  {self.show(beam) or '(start)'} -> "
                    + "  ".join(f"{word} {odds:.0%}" for word, odds in ranked[:5]))
        return ranked[:self.settings.beam]

    def expand_units(self, context, words, topics):
        """The best few next letters, tokens or words for one partial reply:
        [(word or unfinished word, odds), ...]."""
        state = self.step_state(context, words, topics)
        groups = self.unit_groups(words, topics)
        if not groups:
            return []
        if len(groups) == 1:        # one question is the final
            ranked = self.choose(state, STEP_QUESTION, self.step_options(words, groups[0]))
        else:
            finalists = []
            for group in groups:
                for word, _ in self.choose(state, STEP_QUESTION, self.step_options(words, group))[:2]:
                    if word not in finalists:
                        finalists.append(word)
            ranked = self.choose(state, STEP_QUESTION, self.step_options(words, finalists))
        self.detail(f"  {self.show(words) or '(start)'} -> "
                    + "  ".join(f"{word} {odds:.0%}" for word, odds in ranked[:5]))
        return ranked[:self.settings.beam]

    def close(self, beam):
        """`beam` with its sentence ended: an unfinished word is finished if
        it is a word, and dropped if not."""
        if partial(beam[-1][0]):
            stem = beam[-1][0][:-1]
            words = written(beam)
            if stem in self.inventory(words[:-1], ()) or stem in self.extra:
                beam = beam + [(stem, 0.0)]
            else:
                while beam and partial(beam[-1][0]):
                    beam = beam[:-1]
        return beam + [(".", 0.0)]

    # compare: rank whole replies

    def rank(self, context, candidates, question, model=None):
        """Candidate replies, best first: [(words, odds), ...]."""
        if len(candidates) == 1:
            return [(candidates[0], 1.0)]
        options = {str(number): self.show(words) + ("" if words[-1] in STOPS or partial(words[-1]) else " ...")
                   for number, words in enumerate(candidates)}
        state = context + "\nThe assistant answers in complete, grammatical English sentences. " + VOICE
        return [(candidates[int(key)], odds) for key, odds in self.choose(state, question, options, model)]

    def sentence(self, context, so_far, topics, limit):
        """Candidates for `so_far` plus one more sentence of up to `limit`
        words. Both are lists of (word, odds) steps; with tokens or letters,
        unfinished words are steps too."""
        width, words_only = self.settings.beam, self.settings.unit == "words"
        beams, finished = [so_far], []
        for _ in range(limit if words_only else limit * (LONGEST + 1)):
            grown = []
            for beam in beams:
                nexts = self.expand(context, written(beam), topics)
                grown += [beam + [step] for step in nexts]
            if not grown:
                break
            self.steps += 1
            ranked = self.rank(context, [written(beam) for beam in grown], BEGINNING_QUESTION)
            place = {tuple(words): number for number, (words, _) in enumerate(ranked)}
            grown.sort(key=lambda beam: place[tuple(written(beam))])
            # Only the best few go on; of those, some may have just ended.
            kept = grown[:width - len(finished)]
            beams = []
            for beam in kept:
                words = written(beam)
                if beam[-1][0] in STOPS:
                    finished.append(beam)
                elif not words_only and (len(sentence_words(words)) >= limit and not partial(words[-1])
                                         or partial(words[-1]) and len(words[-1]) > LONGEST):
                    finished.append(self.close(beam))     # out of words or letters
                else:
                    beams.append(beam)
            if self.on_text:
                self.on_text(self.show(written(kept[0])))
            # The best candidate has ended: writing on from the others rarely beats it.
            if not beams or kept[0][-1][0] in STOPS:
                break
        # Out of words: close what there is.
        return finished or [self.close(beam) for beam in beams]

    def wants_more(self, context, words):
        """Whether the model says the reply needs another sentence."""
        odds = dict(self.choose(
            context + f'\nThe assistant has replied so far: "{self.show(words)}"',
            "Is that reply finished, or does it need another sentence?",
            {"finished": "the reply already answers the user; stop here",
             "more": "the reply is incomplete; another sentence is needed"}))
        self.detail(f"  another sentence? {odds['more']:.0%}")
        return odds["more"] > 0.6

    def message_words(self, message):
        """The words of `message` that the lexicon lacks, such as names and
        numbers, as {word: how it is written there}. They can be spelled."""
        out = {}
        for shown in re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", message):
            word = shown.lower()
            if not self.lexicon.known(word) and word not in out and len(word) <= LONGEST:
                out[word] = shown
        return out

    def prepare(self, message, topics):
        """The topic words, with the message's own words the lexicon lacks
        when they can be spelled; and how those are written."""
        if self.settings.unit == "words":
            return topics
        self.extra = self.message_words(message)
        self.proper = {**self.lexicon.proper, **{word: shown for word, shown in self.extra.items() if shown != word}}
        self._capitals = None
        self._inventory = {}
        if self.settings.unit in ("tokens", "mixed"):
            why = self.lexicon.tokenizer()[1]
            if why:
                self.detail(f"  tokens are {Pieces.name}: {why}")
        # They come first: a name or a number the user wrote is what an answer most often repeats.
        return list(self.extra) + [word for word in topics if word not in self.extra]

    def reply(self, message, turns=(), notes="", max_words=None):
        """The reply to `message`, as text."""
        settings = self.settings
        if settings.unit not in UNITS:
            raise ValueError(f"d3-talk's unit must be one of {', '.join(UNITS)}, not {settings.unit!r}.")
        context = self.conversation(message, turns, notes)
        topics = self.prepare(message, self.think(message, context))
        left = max_words if max_words else settings.sentences * settings.max_words
        chosen = []
        for number in range(settings.sentences):
            so_far = written(chosen)
            finished = self.sentence(context, chosen, topics, max(1, min(settings.max_words, left)))
            candidates = [written(beam) for beam in finished]
            if chosen:      # the reply without the new sentence competes too
                candidates.append(so_far)
            ranked = self.rank(context, candidates, REPLY_QUESTION, settings.judge)
            for words, odds in ranked:
                self.detail(f"  candidate {odds:.0%}: {self.show(words)}")
            best = ranked[0][0]
            if best == so_far:
                break       # the extra sentence did not improve the reply
            chosen = next(beam for beam in finished if written(beam) == best)
            left -= len(best) - len(so_far)
            if number + 1 == settings.sentences or left < 4 or not self.wants_more(context, best):
                break
        self.words = written(chosen)
        if settings.learn:
            steps, before = [], START
            for word, odds in word_steps(chosen):
                steps.append((before, word, odds))
                before = START if word in STOPS else before if word in PUNCT else word
            if not self.lexicon.learn(steps):
                self.detail(f"  could not save what was learned to {self.lexicon.usage_path()}")
        return self.show(self.words)


class D3TalkModel(ollama.OllamaModel):
    """A decision model that kilix-chat talks with like any other model."""
    architecture = "d3-talk"

    def __init__(self, name, *, fast=False, unit="words"):
        super().__init__(name)
        self.settings = Settings(beam=1) if fast else Settings()
        self.settings.unit = unit

    def chat(self, messages, *, max_tokens=256, on_text=None, cancel=None):
        started = time.monotonic()
        notes, turns, message = split(messages)
        talker = Talker(self.name, lexicon(), self.settings, on_text=on_text, cancel=cancel)
        text = talker.reply(message, turns[-3:], notes, max_words=max_tokens)
        return Reply(text, time.monotonic() - started, talker.tokens, len(talker.words))


# the d3-talk command

DIM, RESET = "\033[2m", "\033[0m"


def usage_report(lexicon, limit=40):
    """Lines listing the word pairs the model has used most."""
    if not lexicon.used:
        return ["Nothing learned yet."]
    lines = ["word before\tword\tuses\taverage confidence"]
    for (before, word), (times, confidence) in sorted(lexicon.used.items(), key=lambda item: -item[1][0])[:limit]:
        lines.append(f"{before}\t{word}\t{times}\t{confidence / times:.0%}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="d3-talk", description="Let an Ollama decision model speak, one chosen word at a time.",
        epilog="With no message it reads messages from the terminal, one per line, as one conversation.")
    parser.add_argument("message", nargs="*", help="what to say to the model")
    parser.add_argument("-m", "--model", default="d3:2b", help="decision model to speak with (default: d3:2b)")
    parser.add_argument("-b", "--beam", type=int, default=2, metavar="N",
                        help="partial replies kept alive at once (default: 2)")
    parser.add_argument("--fast", action="store_true",
                        help="keep one partial reply: a third less waiting, and vaguer answers")
    parser.add_argument("-s", "--sentences", type=int, default=2, metavar="N",
                        help="most sentences in a reply (default: 2)")
    parser.add_argument("--max-words", type=int, default=20, metavar="N",
                        help="most words in a sentence (default: 20)")
    parser.add_argument("--vocab", type=int, default=4000, metavar="N",
                        help="most-used words swept for topic words; 0 skips the sweep (default: 4000)")
    parser.add_argument("--topics", type=int, default=24, metavar="N",
                        help="topic words kept from the sweep (default: 24)")
    parser.add_argument("--judge", metavar="MODEL", help="a larger decision model that picks the final reply")
    parser.add_argument("-u", "--unit", choices=UNITS, default="words",
                        help="what one choice adds: a word, a token of d3's tokenizer, a letter, or any of the "
                             "three (default: words)")
    parser.add_argument("--any-letter", action="store_true",
                        help="with --unit letters or mixed, also offer letters that continue no known word")
    parser.add_argument("--no-learn", action="store_true", help="do not count the words used")
    parser.add_argument("--usage", action="store_true", help="list the word pairs used most, then exit")
    parser.add_argument("--show", action="store_true",
                        help="print the reply as it stands after every word (or letter or token), each on "
                             "its own line")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the topic words, the choices at every step and the time taken")
    args = parser.parse_args(argv)
    if min(args.beam, args.sentences, args.max_words) < 1:
        parser.error("--beam, --sentences and --max-words must be at least 1")
    settings = Settings(1 if args.fast else args.beam, args.sentences, args.max_words, args.vocab, args.topics, args.judge,
                        not args.no_learn, args.unit, args.any_letter)
    live = sys.stderr.isatty() and not args.show
    drawn = False

    def status(text):
        """One dim line on the terminal, redrawn in place."""
        nonlocal drawn
        width = shutil.get_terminal_size((80, 24)).columns - 1
        sys.stderr.write(f"\r\033[K{DIM}{text[:width]}{RESET}")
        sys.stderr.flush()
        drawn = True

    def clear():
        nonlocal drawn
        if drawn:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
            drawn = False

    def on_text(text):
        if args.show and not text.startswith("("):
            print(text, flush=True)
        elif live:
            status(text)

    def on_detail(line):
        clear()
        print(line, file=sys.stderr)

    try:
        words = lexicon()
        if args.usage:
            print("\n".join(usage_report(words)))
            return 0
        turns = []

        def answer(message):
            talker = Talker(args.model, words, settings, on_text=on_text,
                            on_detail=on_detail if args.verbose else None)
            started = time.monotonic()
            text = talker.reply(message, turns[-3:])
            clear()
            if args.verbose:
                steps = f" in {talker.steps} steps of {args.unit}" if args.unit != "words" else ""
                print(f"{len(talker.words)} words{steps} from {talker.questions} questions "
                      f"({talker.tokens} prompt tokens) in {time.monotonic() - started:.1f} s", file=sys.stderr)
            turns.append((message, text))
            return text

        if args.message:
            print(answer(" ".join(args.message)))
            return 0
        prompt = "you> " if sys.stdin.isatty() else ""
        if prompt:
            import readline  # noqa: F401 - line editing for input()
        while True:
            try:
                message = input(prompt).strip()
            except EOFError:
                return 0
            if message:
                print(answer(message))
    except (OSError, RuntimeError, ValueError) as error:
        clear()
        print(f"d3-talk: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        clear()
        return 130
