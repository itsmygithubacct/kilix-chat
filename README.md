# kilix-chat

Chat with a local model in the terminal. kilix-chat runs text diffusion
models (LLaDA, LLaDA-MoE, Dream, DiffuCoder, RND1) through llama.cpp's
`llama-diffusion-cli`, and ordinary models through a local Ollama daemon.
A diffusion reply appears all at once and sharpens step by step as it
denoises, rather than growing word by word: the window draws every token
that is still masked as `░`, and words settle into place among them.

```sh
./kilix-chat                                   # pick a model, then chat
./kilix-chat --model ~/models/LLaDA-1.5.Q4_K_M.gguf --device Vulkan1
./kilix-chat --model ollama:qwen3.5:4b
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
| `/model [SPEC]` | Open the model picker, or switch to a GGUF path or `ollama:NAME` |
| `/context` | Show the prompt's tokens against the model's context, the reply budget, and how many old turns would be dropped |
| `/memory` | Show the saved notes |
| `/memory add TEXT`, `/remember TEXT` | Save a note |
| `/memory forget N` | Remove note N |
| `/memory clear` | Remove every note |
| `/system [TEXT]` | Show or set the system prompt (`/system clear` removes it) |
| `/length [N]` | Show or set the reply length, 8 to 16384 tokens |
| `/steps [N]` | Show or set the diffusion steps per reply (`/steps auto`: half the reply length) |
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
make test       # offline, against a stand-in llama-diffusion-cli
make defects    # every planted defect must turn the suite red
```
