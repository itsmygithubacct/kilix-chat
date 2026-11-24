"""kilix-chat: chat with a local model, or serve one to a coding agent."""
from __future__ import annotations

import argparse
import sys

from kilix_chat import __version__
from kilix_chat.models import Model, catalog


def main(argv=None):
    parser = argparse.ArgumentParser(prog="kilix-chat", description=__doc__)
    parser.add_argument("--version", action="version", version=f"kilix-chat {__version__}")
    parser.add_argument("--model", help="a diffusion GGUF path, or ollama:NAME")
    parser.add_argument("--device", help="llama.cpp device for diffusion models, for example Vulkan1")
    parser.add_argument("--max-tokens", type=int, default=256, help="reply length in tokens (default 256)")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("models", help="list the models kilix-chat can open")
    ask = sub.add_parser("ask", help="print one reply to one message")
    ask.add_argument("message")
    ask.add_argument("--system", default="")
    serve = sub.add_parser("serve", help="serve the model to OpenAI and Ollama clients on loopback")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--alias", help="model ID reported to clients (default: the file name)")
    args = parser.parse_args(argv)

    if args.command == "models":
        for spec, label in catalog():
            print(f"{label}\t{spec}")
        return 0
    if args.command in ("ask", "serve"):
        if not args.model:
            parser.error(f"{args.command} needs --model")
        if args.command == "serve":
            from kilix_chat.server import is_loopback
            if not is_loopback(args.host):
                parser.error(f"serve listens on loopback only (127.0.0.1, ::1 or localhost), not {args.host}")
        try:
            model = Model(args.model, device=args.device)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"kilix-chat: {error}", file=sys.stderr)
            return 1
        if args.command == "ask":
            messages = ([{"role": "system", "content": args.system}] if args.system else [])
            messages.append({"role": "user", "content": args.message})
            try:
                reply, _ = model.reply(messages, max_tokens=args.max_tokens)
            except (OSError, RuntimeError, ValueError) as error:
                print(f"kilix-chat: {error}", file=sys.stderr)
                return 1
            print(reply.text)
            return 0
        from kilix_chat.server import serve as run_server
        alias = args.alias or model.name
        def ready(server):
            print(f"kilix-chat serving {alias} on http://{args.host}:{server.server_address[1]}/v1", flush=True)
        try:
            run_server(model, alias=alias, host=args.host, port=args.port,
                       max_tokens_cap=args.max_tokens, log=lambda line: print(line, file=sys.stderr), ready=ready)
        except KeyboardInterrupt:
            pass
        except OSError as error:
            print(f"kilix-chat: could not serve on {args.host}:{args.port} ({error})", file=sys.stderr)
            return 1
        return 0
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("the chat window needs a terminal; use `kilix-chat ask` in scripts")
    from kilix_chat.tui import run
    return run(args.model, args.device, args.max_tokens)
