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

Word and word-pair frequencies come from Peter Norvig's lists
(norvig.com/ngrams). They are downloaded once, on first use, into
`$XDG_DATA_HOME/kilix-chat/d3-talk/`, where the learned usage is kept too.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import functools
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


def fetch(directory):
    """Download the frequency lists that `directory` does not hold yet."""
    for name, url in SOURCES.items():
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
            raise RuntimeError(f"d3-talk needs word frequencies and could not download {url} ({error}). "
                               f"Put count_1w.txt and count_2w.txt in {directory}.") from error


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

    def choose(self, state, question, options, model=None):
        if self.cancel is not None and self.cancel.is_set():
            raise Cancelled()
        ranked, tokens = ollama.decide(model or self.model, state, question, options)
        self.questions += 1
        self.tokens += tokens
        return ranked

    def show(self, words):
        return render(words, self.lexicon.proper)

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
        return (context + "\nThe assistant is answering in complete, grammatical English sentences, written "
                "one word at a time. " + VOICE
                + f"\nKey words the answer may need: {', '.join(topics[:10])}."
                + f'\nReply so far: "{self.show(words)}"')

    def step_options(self, words, group):
        """Each choice is shown as the reply with that word added."""
        tail, lead = words, ""
        if len(words) > 12:
            tail, lead = words[-9:], "... "
            while tail and tail[0] in PUNCT:
                tail = tail[1:]
        return {word: lead + self.show(tail + [word]) + ("" if word in STOPS else " ...") for word in group}

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

    def expand(self, context, beam, topics):
        """The best few next words for one partial reply: [(word, odds), ...]."""
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

    # compare: rank whole replies

    def rank(self, context, candidates, question, model=None):
        """Candidate replies, best first: [(words, odds), ...]."""
        if len(candidates) == 1:
            return [(candidates[0], 1.0)]
        options = {str(number): self.show(words) + ("" if words[-1] in STOPS else " ...")
                   for number, words in enumerate(candidates)}
        state = context + "\nThe assistant answers in complete, grammatical English sentences. " + VOICE
        return [(candidates[int(key)], odds) for key, odds in self.choose(state, question, options, model)]

    def sentence(self, context, written, topics, limit):
        """Candidates for `written` plus one more sentence of up to `limit`
        words. Both are lists of (word, odds) steps."""
        width = self.settings.beam
        beams, finished = [written], []
        for _ in range(limit):
            grown = []
            for beam in beams:
                nexts = self.expand(context, [word for word, _ in beam], topics)
                grown += [beam + [step] for step in nexts]
            ranked = self.rank(context, [[word for word, _ in beam] for beam in grown], BEGINNING_QUESTION)
            place = {tuple(words): number for number, (words, _) in enumerate(ranked)}
            grown.sort(key=lambda beam: place[tuple(word for word, _ in beam)])
            # Only the best few go on; of those, some may have just ended.
            kept = grown[:width - len(finished)]
            beams = []
            for beam in kept:
                (finished if beam[-1][0] in STOPS else beams).append(beam)
            if self.on_text:
                self.on_text(self.show([word for word, _ in kept[0]]))
            # The best candidate has ended: writing on from the others rarely beats it.
            if not beams or kept[0][-1][0] in STOPS:
                break
        # Out of words: close what there is.
        return finished or [beam + [(".", 0.0)] for beam in beams]

    def wants_more(self, context, words):
        """Whether the model says the reply needs another sentence."""
        odds = dict(self.choose(
            context + f'\nThe assistant has replied so far: "{self.show(words)}"',
            "Is that reply finished, or does it need another sentence?",
            {"finished": "the reply already answers the user; stop here",
             "more": "the reply is incomplete; another sentence is needed"}))
        self.detail(f"  another sentence? {odds['more']:.0%}")
        return odds["more"] > 0.6

    def reply(self, message, turns=(), notes="", max_words=None):
        """The reply to `message`, as text."""
        settings = self.settings
        context = self.conversation(message, turns, notes)
        topics = self.think(message, context)
        left = max_words if max_words else settings.sentences * settings.max_words
        written = []
        for number in range(settings.sentences):
            so_far = [word for word, _ in written]
            finished = self.sentence(context, written, topics, max(1, min(settings.max_words, left)))
            candidates = [[word for word, _ in beam] for beam in finished]
            if written:     # the reply without the new sentence competes too
                candidates.append(so_far)
            ranked = self.rank(context, candidates, REPLY_QUESTION, settings.judge)
            for words, odds in ranked:
                self.detail(f"  candidate {odds:.0%}: {self.show(words)}")
            best = ranked[0][0]
            if best == so_far:
                break       # the extra sentence did not improve the reply
            written = next(beam for beam in finished if [word for word, _ in beam] == best)
            left -= len(best) - len(so_far)
            if number + 1 == settings.sentences or left < 4 or not self.wants_more(context, best):
                break
        self.words = [word for word, _ in written]
        if settings.learn:
            steps, before = [], START
            for word, odds in written:
                steps.append((before, word, odds))
                before = START if word in STOPS else before if word in PUNCT else word
            if not self.lexicon.learn(steps):
                self.detail(f"  could not save what was learned to {self.lexicon.usage_path()}")
        return self.show(self.words)


class D3TalkModel(ollama.OllamaModel):
    """A decision model that kilix-chat talks with like any other model."""
    architecture = "d3-talk"

    def __init__(self, name, *, fast=False):
        super().__init__(name)
        self.settings = Settings(beam=1) if fast else Settings()

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
    parser.add_argument("--no-learn", action="store_true", help="do not count the words used")
    parser.add_argument("--usage", action="store_true", help="list the word pairs used most, then exit")
    parser.add_argument("--show", action="store_true",
                        help="print the reply as it stands after every word, each on its own line")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the topic words, the choices at every step and the time taken")
    args = parser.parse_args(argv)
    if min(args.beam, args.sentences, args.max_words) < 1:
        parser.error("--beam, --sentences and --max-words must be at least 1")
    settings = Settings(1 if args.fast else args.beam, args.sentences, args.max_words, args.vocab, args.topics, args.judge,
                        not args.no_learn)
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
                print(f"{len(talker.words)} words from {talker.questions} questions in "
                      f"{time.monotonic() - started:.1f} s", file=sys.stderr)
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
