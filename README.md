# kilix-chat

Chat with a local model in the terminal. kilix-chat runs text diffusion
models (LLaDA, LLaDA-MoE, Dream, DiffuCoder, RND1) through llama.cpp's
`llama-diffusion-cli`, and ordinary models through a local Ollama daemon.
A diffusion reply appears all at once and sharpens step by step as it
denoises, rather than growing word by word.

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
| Enter | Send the message |
| F2 | Choose a model |
| F3 | Cycle the reply length: 64, 128, 256 or 512 tokens |
| Ctrl-N | Start a new chat |
| PgUp / PgDn | Scroll the conversation |
| Esc | Stop the reply in progress (the unanswered message is taken back) |
| F10 or Ctrl-Q | Quit |

## Diffusion models

llama-server cannot load diffusion models, so every reply is one
`llama-diffusion-cli` run. The sequence holds the conversation plus the reply
length, denoised in half as many steps at temperature 0 with seed 0. The
sampling algorithm comes from the GGUF architecture: confidence for LLaDA and
entropy for Dream and RND1. A sibling `llama-tokenize` sizes the prompt
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

## Serving a coding agent

`kilix-chat serve` answers `GET /health`, `GET /v1/models` and
`POST /v1/chat/completions`, streamed or not, one request at a time. Offered
tools are described in the system prompt. The model calls one by replying
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`, which becomes an
OpenAI `tool_calls` entry. Calls to tools that were not offered are ignored.
`--max-tokens` caps every reply (default 256), whatever a client asks for.
Diffusion models are not trained for tool use, and every turn denoises the
whole conversation again, so expect slow and unreliable agent turns.

## Tests

```sh
make test       # offline, against a stand-in llama-diffusion-cli
make defects    # every planted defect must turn the suite red
```
