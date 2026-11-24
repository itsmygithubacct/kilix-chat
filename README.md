# kilix-chat

Chat with a local model in the terminal. kilix-chat runs text diffusion
models (LLaDA, LLaDA-MoE, Dream, DiffuCoder, RND1) through llama.cpp's
`llama-diffusion-cli`, and ordinary models through a local Ollama daemon.
A diffusion reply appears all at once and sharpens step by step as it
denoises, rather than growing word by word: the window draws every token
that is still masked as `░`, and words settle into place among them.
With d3-talk it also chats with Ollama's decision models, such as `d3:2b`,
which can only choose among options: the reply is built from the model's
choices, one word at a time.

```sh
./kilix-chat                                   # pick a model, then chat
./kilix-chat --model ~/models/LLaDA-1.5.Q4_K_M.gguf --device Vulkan1
./kilix-chat --model ollama:qwen3.5:4b
./kilix-chat --model d3-talk:d3:2b             # a decision model, through d3-talk
./kilix-chat models                            # what it can open here
./kilix-chat --model MODEL.gguf ask "Is two even?"
./kilix-chat --model MODEL.gguf serve --port 8765 --alias local
```

It needs Python 3.11 or newer and nothing else from PyPI.

## Chat window

| Key | Action |
|---|---|
| Enter | Send the message, or run the command |
| Tab | Complete a command name or its first word |
| Ctrl-U | Clear the entry line |
| F2 | Choose a model |
| F3 | Cycle the reply length: 64, 128, 256 or 512 tokens |
| F4 | Show or hide replies as they denoise or stream (shown by default) |
| Ctrl-N | Start a new chat |
| PgUp / PgDn | Scroll the conversation |
| Esc | Stop the reply in progress (the unanswered message is taken back), or close a command's output |
| F10 or Ctrl-Q | Quit |

A line that starts with `/` is a command and is never sent to the model.
While you type one, the status line shows the matching commands or the
command's usage. An unknown command says so and stays in the entry to fix.
Start a message with `//` to send it with a single leading `/`.

| Command | Action |
|---|---|
| `/help` (`/?`) | List the commands and keys |
| `/model [SPEC]` | Open the model picker, or switch to a GGUF path, `ollama:NAME` or `d3-talk:NAME` |
| `/context` | Show the prompt's tokens against the model's context, the reply budget, and how many old turns would be dropped |
| `/memory` | Show the saved notes |
| `/memory add TEXT`, `/remember TEXT` | Save a note |
| `/memory forget N` | Remove note N |
| `/memory clear` | Remove every note |
| `/system [TEXT]` | Show or set the system prompt (`/system clear` removes it) |
| `/length [N]` | Show or set the reply length, 8 to 16384 tokens |
| `/steps [N]` | Show or set the diffusion steps per reply (`/steps auto`: the model's preset, half the reply length, or an eighth for d3LLM) |
| `/visual [on\|off]` (`/diffusion`) | Show replies as they denoise or stream, or only when finished; with no word it toggles, like F4 |
| `/device [NAME]` | Show or set the llama.cpp device, for example `Vulkan1` (`/device auto`: llama.cpp chooses) |
| `/retry` | Ask for the last reply again |
| `/undo` | Take back the last exchange and put your message back in the entry |
| `/save [FILE]` | Write the chat as Markdown, by default to `~/.local/share/kilix-chat/chats/` |
| `/clear` (`/new`) | Start a new chat |
| `/quit` (`/exit`, `/q`) | Leave kilix-chat |

`/model`, `/context`, `/retry`, `/undo` and `/clear` wait until the reply
in progress finishes or is stopped. The other commands work during a reply,
and a new length, step count, device, display or system prompt applies from
the next reply.

### Watching a reply denoise

llama-diffusion-cli prints a masked token as one space, so kilix-chat draws
each run of spaces as masks, keeping one space before a word for the word
itself. Indentation therefore looks masked until the reply is finished, and
the finished reply is shown exactly as the model wrote it. The sequence is as
long as the reply length, and most of it ends up as end-of-text padding,
which LLaDA unmasks first, so the masks shrink for most of the steps before
the words arrive. A shorter `/length` gets to the words sooner. While a reply denoises,
the window keeps its start in view, because its end is padding that is
settled last. PgUp and PgDn still scroll.

### Memory

The memory is a list of notes kept across chats. It lives in
`$XDG_DATA_HOME/kilix-chat/memory.md` (by default
`~/.local/share/kilix-chat/memory.md`), one `- ` line per note, so it can also
be edited by hand. Every chat in the window sends the notes to the model at
the end of its system prompt. `kilix-chat ask` and `kilix-chat serve` do not
use the memory.

### Context

`/context` counts the conversation as it would be sent: the system prompt,
the memory and every turn. For a diffusion model the count comes from the
sibling `llama-tokenize`, and the context length from the GGUF. The reply
length is reserved after the prompt, because a diffusion run denoises the
prompt and the reply as one sequence. When the two do not fit, the oldest
turns are dropped, one at a time, and `/context` says how many.

For an Ollama model the tokens are estimated at four bytes each. The context
is the window Ollama has the model loaded with, else `num_ctx` from the
model's parameters, else the model's trained length. Ollama drops the oldest
turns of an overlong prompt by the same rule.

## Diffusion models

llama-server cannot load diffusion models, so every reply is one
`llama-diffusion-cli` run. The sequence holds the conversation plus the reply
length, denoised in half as many steps at temperature 0.1 with seed 0. Each
step fills the masked tokens the model is most sure of
(`--diffusion-algorithm 4`).

Two llama.cpp behaviours set those choices. Up to at least upstream master
of October 2026:

- `--temp 0` does not take the most probable token. It leaves out the
  temperature sampler and so samples at temperature 1. A temperature of 0.1
  comes close to always taking the most probable token.
- Entropy order (`--diffusion-algorithm 1`), which Dream and RND1 are meant
  to use, is ranked the wrong way round. Its confidence is the entropy
  itself, sorted highest first, so it fills the least certain tokens first.

With entropy order at `--temp 0`, Dream-Coder-v0-Instruct-7B wrote code that
used undefined names. With confidence order at 0.1, the same prompts gave
correct code. A sibling `llama-tokenize` sizes the prompt
exactly. Without it, the UTF-8 byte count serves as an upper bound. A
conversation that outgrows the model's context drops its oldest turns.

`llama-diffusion-cli` applies the model's chat template to exactly one
system message and one user message. kilix-chat writes earlier turns, and a
coding agent's tool calls and results, into that user message as a labelled
transcript. The reply is read from the log file that llama.cpp writes, with
prefixes off, because GPU drivers can add messages to stderr.

Models are found in the folders on `KILIX_CHAT_MODELS` (separated by `:`),
or by default `~/.local/share/kilix-chat/models` and
`~/.local/share/puzzllm/diffusion`. Only GGUFs with a supported diffusion
architecture are listed. The tool is `KILIX_CHAT_DIFFUSION_CLI`,
`PUZZLLM_DIFFUSION_CLI`, `llama-diffusion-cli` on PATH, or the newest
`llama-*/llama-diffusion-cli` in a model folder. `--device`,
`KILIX_CHAT_DEVICE` or `PUZZLLM_GPU` selects the llama.cpp device.

Models and tools are never downloaded by kilix-chat. Q4_K_M files that fit a
6 GB GPU include LLaDA-MoE-7B-A1B-Instruct (4.5 GB), LLaDA-1.5 (4.9 GB),
Dream-v0-Instruct-7B and Dream-Coder-v0-Instruct-7B (4.7 GB each). Build the
tool from llama.cpp with `cmake -DGGML_VULKAN=ON` (or CUDA) and
`cmake --build build --target llama-diffusion-cli llama-tokenize`.

### d3LLM

[d3LLM](https://github.com/hao-ai-lab/d3LLM) (arXiv 2601.07568) distils
LLaDA and Dream so that each denoising step commits many tokens. Its own
PyTorch decoder needs the BF16 weights, about 16 GB. The
`mradermacher/d3LLM_LLaDA-GGUF` Q4_K_M file (4.9 GB) runs in
llama-diffusion-cli on a 6 GB GPU.

kilix-chat recognises d3LLM by its file name or any `general.*` metadata
text, such as the `general.source.url` that the GGUF carries; it keeps
LLaDA's `general.name`. It then defaults to an eighth as many steps as
reply tokens, at least 8, with the most certain tokens filled first.
`/steps` still overrides this.

Measured 2026-11-23 on a Quadro RTX 3000 (6 GB, Vulkan) with llama.cpp
b10174: 256-token replies to 10 prompts (explanation, Python, tips,
arithmetic, an email, a summary, a recipe, a haiku), temperature 0.1.
"Coherent" means the reply answers the prompt without loops, garbled words
or a cut-off ending, judged by reading each reply.

| Model | Steps | Seconds per reply | Coherent replies |
|---|---|---|---|
| LLaDA-1.5, default | 128 | 59–66 | 7 of 10 |
| d3LLM | 256 | 132–195 | 5 of 5 (first five prompts only) |
| d3LLM | 128 | 60–68 | 10 of 10 |
| d3LLM | 64 | 32–35 | 5 of 10 |
| d3LLM, default | 32 | 17–25 | 9 of 10 |

At 32 steps, d3LLM was about three times as fast as LLaDA-1.5 at its
default, with fewer bad replies. Its one failure was a word repeated
dozens of times in the email. At 64 steps, replies were more often cut
short or garbled than at 32, on both prompt sets. llama.cpp's 32-token
block schedule (`--diffusion-block-length 32`) did no better than the
timestep schedule at 40 or 80 steps.

## d3-talk: decision models

Ollama's decision models (`d3:2b`, `d3:4b`, `d3:9b`, `d3:27b`; upstream
`vllm-sr/d3-nano` and its siblings) do not write text. Each request puts a
question to the model about some text, and the answer is a probability for
every option offered. They are unrelated to d3LLM above, which is a
diffusion model. Ollama refuses to chat with one, so the model picker lists
each as `NAME  [d3-talk]`, and `d3-talk:NAME` opens it.

d3-talk builds a reply out of multiple-choice questions, always offering the
most-used words first:

1. **Think.** The 4,000 most-used English words are swept in order of use,
   100 to a question, and the model picks the words a good answer would
   name. Five survive each question, the survivors compete, and the best 24
   are the topic words, together with the content words of your message.
2. **Compose.** The reply grows one word at a time. Each step asks about
   three groups of up to 14 choices: the words that most often follow the
   last word, the most-used grammar words, and the topic words. Each choice
   is shown to the model as the reply so far with that word added. The best
   two of every group meet in a final. A sentence has to open on a grammar
   word; when topic words could open one, replies read like headlines.
3. **Compare.** Three partial replies are kept alive. After every word the
   model ranks them as whole sentences, which it does far better than it
   picks single words. After a full stop it is asked whether the reply needs
   another sentence, and a second sentence stays only if the model prefers
   the reply with it.
4. **Learn.** Every word of the finished reply is counted with the word
   before it, and pairs the model uses often are offered earlier next time.

In the chat window the reply is shown as it stands after every word, after a
line of topic words as they are found. The reply length counts words. The
system prompt and the memory are sent with every question, cut to 400
characters, and so are the last three exchanges.

`kilix-chat --model d3-talk:d3:2b serve` serves it like any other model,
so OpenAI and Ollama clients can chat with a decision model. The reported
prompt tokens add up every question's prompt: about 41,000 for the reply
"It is Paris." A request that offers tools is refused, as it is for an
Ollama model.

The same engine runs on its own as `d3-talk`:

```sh
./d3-talk "What is the capital of France?"     # one reply
./d3-talk                                      # one message per line
./d3-talk --show "Why is the sky blue?"        # the reply after every word, a line each
./d3-talk -v "How are you today?"              # the topic words and the choices at every step
./d3-talk --usage                              # the word pairs it has used most
```

`-m MODEL` names another decision model, `-b N` the number of partial
replies kept (1 is fastest and much worse), `-s N` the most sentences,
`--max-words N` the most words in a sentence, `--vocab N` how many words the
sweep covers (0 skips it), `--judge MODEL` a larger decision model for the
final choice, and `--no-learn` leaves the usage counts alone.

Word and word-pair frequencies come from Peter Norvig's
[n-gram lists](https://norvig.com/ngrams/) (`count_1w.txt` and
`count_2w.txt`, 10 MB together). They are downloaded on first use into
`$XDG_DATA_HOME/kilix-chat/d3-talk/`, where the learned usage is kept as
`usage.tsv`. To work offline, put the two files there yourself. Spellings
and capitals are checked against `/usr/share/dict/words` when it exists
(`KILIX_CHAT_WORD_LIST` names another list); without one, every alphabetic
word in the lists counts and names are not capitalised.

What to expect, measured 2026-11-23 with `d3:2b` (Q8_0) on a Quadro RTX
3000 (6 GB), which holds the whole model:

- A question costs about 30 ms plus 0.27 ms per prompt token, and the model
  reads the conversation and every option again for each question. Sending
  questions in parallel, or several in one request, was no faster. A reply
  takes about 90 to 250 questions and 10 to 30 seconds, of which the sweep
  is 45 questions and about 7 seconds.
- Short factual questions come out well ("It is Paris.", "I feel well
  today.", "It's four total."). Open questions give vaguer sentences, and a
  sentence can end unfinished. The wording changes from run to run.
- Against ten reference replies the model picked the reference's next word
  30% of the time from one list of 40 choices, and 40% of the time (65%
  within its top three) from three small groups with a final. Asking it for
  the next word's first letter (19%) or whether a grammar word or a content
  word comes next (63%) did not help, so neither is used.
- If the model is loaded while the GPU is full, Ollama keeps it on the CPU,
  where a question takes seconds. `ollama ps` shows the split, and
  `ollama stop d3:2b` makes the next reply load it again.

## Serving a coding agent or a chat client

`kilix-chat serve` listens on loopback only (127.0.0.1, ::1 or localhost) and
answers in two dialects on one port, one request at a time.

OpenAI, for coding agents: `GET /health`, `GET /v1/models` and
`POST /v1/chat/completions`, streamed or not. Offered tools are described in
the system prompt. The model calls one by replying
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, which becomes an
OpenAI `tool_calls` entry. Calls to tools that were not offered are ignored.

Ollama, for chat clients such as Kilix Avatar
(`--ollama http://127.0.0.1:PORT --model ALIAS`):

- `GET /api/tags` lists the one served model under its alias, with the GGUF's
  size, a digest of its path, size and modification time, and its
  architecture as the family.
- `POST /api/chat` streams NDJSON by default, as Ollama does, or answers once
  with `"stream": false`. The reply arrives whole, followed by a `done` line
  with `done_reason: "stop"` and Ollama's counts and durations. While a
  diffusion reply denoises, the stream carries an empty message every 5
  seconds so a client's read timeout does not end it. A client that
  disconnects stops the reply. A failure mid-stream is an `{"error": ...}`
  line. `options.num_predict` sets the reply length, up to `--max-tokens`.
  Tool calls come back as Ollama `tool_calls` with object arguments.
- `POST /api/pull` of the served name, or `NAME:latest`, succeeds without
  downloading anything.
- `GET /api/version` gives kilix-chat's version.
- Any other model name is HTTP 404 with an Ollama-style `{"error": ...}`
  that names the served model.

```sh
./kilix-chat --model ~/.local/share/puzzllm/diffusion/d3LLM_LLaDA.Q4_K_M.gguf \
    --device Vulkan1 --max-tokens 256 serve --port 11435 --alias d3llm
# then, for example: ./build/kilix-avatar --chat --ollama http://127.0.0.1:11435 --model d3llm
```

`--max-tokens` caps every reply (default 256), whatever a client asks for.
Diffusion models are not trained for tool use, and every turn denoises the
whole conversation again, so expect slow and unreliable agent turns.

## Loading models

Every part of kilix-chat opens a model through `kilix_chat.models.load`. A
diffusion model's weights reach the GPU only while one reply runs, in
`DiffusionModel.generate`. These two functions are where a GPU manager can
hook in. kilix-chat does not unload other models itself, so a diffusion
reply fails with "There is not enough free GPU memory" while, for example,
Ollama keeps a large model loaded.

## Tests

```sh
make test       # offline: stand-ins for llama-diffusion-cli and a decision model
make defects    # every planted defect must turn the suite red
```
