#!/usr/bin/env python3
"""kilix-chat offline: GGUF metadata, prompts, tool calls, the diffusion
runner against a stand-in llama-diffusion-cli, the server, Ollama and the
chat window's key handling."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from kilix_chat import diffusion, gguf, models, ollama, prompt, server, tui  # noqa: E402


def gguf_bytes(arch, name="Tiny", context=None, extra=()):
    """Metadata-only GGUF with arrays (strings and floats) ahead of the scalars."""
    def text(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data
    pairs = [text("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 2) + text("a") + text("bc"),
             text("tokenizer.ggml.scores") + struct.pack("<IIQ", 9, 6, 3) + struct.pack("<3f", 0, 1, 2),
             text("general.alignment") + struct.pack("<II", 4, 32),
             text("general.architecture") + struct.pack("<I", 8) + text(arch),
             text("general.name") + struct.pack("<I", 8) + text(name)]
    if context:
        pairs.append(text(f"{arch}.context_length") + struct.pack("<II", 4, context))
    pairs.extend(extra)
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs)) + b"".join(pairs)


FAKE_CLI = r'''#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
prompt = open(args[args.index('-f') + 1]).read()
system = args[args.index('-sys') + 1] if '-sys' in args else ''
with open(os.environ['CHAT_CALLS'], 'a') as out:
    print(json.dumps([args, system, prompt, os.environ.get('LD_LIBRARY_PATH', '')]), file=out)
prefix = '' if '--no-log-prefix' in args else '0.05.630.887 I '
log = open(args[args.index('--log-file') + 1], 'w') if '--log-file' in args else None
def entry(text):
    for stream in filter(None, (sys.stderr, log)):
        stream.write(prefix + text)
        stream.flush()
if os.environ.get('CHAT_FAIL'):
    entry('error: failed to load model\n')
    sys.exit(1)
reply = os.environ.get('CHAT_REPLY') or ('Echo: ' + prompt.splitlines()[-1])
steps = int(args[args.index('--diffusion-steps') + 1])
if '--diffusion-visual' in args:
    for step in range(steps):
        entry('\x1b[2J\x1b[H')
        entry('\rdiffusion step: %d/%d [==] 50%%' % (step, steps))
        entry('\n')
        shown = reply[: len(reply) * step // steps]
        entry(shown + ' ' * (len(reply) - len(shown)) + '\n')
    entry('\ntotal time: 5.00ms, time per step: 0.16ms\n')
    entry('\x1b[2J\x1b[H')
    entry('\n' + reply + '<|im_end|>\n')
else:
    entry('load_tensors: offloaded\n\rdiffusion step: 3/4 [=] 75%')
    entry('\ntotal time: 5.00ms, time per step: 0.16ms\n')
    entry('\n' + reply + '\n')
sys.stderr.write('NVVM compilation failed: 3\n')
'''


class Fixture(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.models = self.root / "models"
        self.tools = self.models / "llama-b1"
        self.tools.mkdir(parents=True)
        self.cli = self.tools / "llama-diffusion-cli"
        self.cli.write_text(FAKE_CLI)
        self.counter = self.tools / "llama-tokenize"
        self.counter.write_text("#!/bin/sh\necho unrelated\necho 'Total number of tokens: 7'\n")
        for tool in (self.cli, self.counter):
            tool.chmod(0o700)
        self.dream = self.models / "dream coder.gguf"
        self.dream.write_bytes(gguf_bytes("dream", "Dream Coder", 2048))
        self.calls = self.root / "calls"
        env = {"KILIX_CHAT_MODELS": str(self.models), "CHAT_CALLS": str(self.calls), "PATH": "/usr/bin:/bin",
               "KILIX_CHAT_DIFFUSION_CLI": "", "PUZZLLM_DIFFUSION_CLI": "", "KILIX_CHAT_DEVICE": "",
               "PUZZLLM_GPU": "", "OLLAMA_HOST": "127.0.0.1:9", "CHAT_REPLY": "", "CHAT_FAIL": ""}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def runs(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []


class MetadataTest(Fixture):
    def test_scalars_survive_arrays_and_bad_files_are_refused(self):
        info = gguf.metadata(self.dream)
        self.assertEqual((info["general.architecture"], info["general.name"], gguf.context_length(info)),
                         ("dream", "Dream Coder", 2048))
        self.assertNotIn("tokenizer.ggml.tokens", info)
        for data in (b"GGML" + self.dream.read_bytes()[4:], self.dream.read_bytes()[:-3]):
            self.dream.write_bytes(data)
            with self.assertRaises(ValueError):
                gguf.architecture(self.dream)

    def test_catalog_lists_only_diffusion_architectures(self):
        (self.models / "chat.gguf").write_bytes(gguf_bytes("qwen3"))
        (self.models / "half.gguf.part").write_bytes(gguf_bytes("llada"))
        nested = self.models / "more/llada.gguf"
        nested.parent.mkdir()
        nested.write_bytes(gguf_bytes("llada-moe"))
        self.assertEqual(diffusion.local_models(), [(str(self.dream), "dream"), (str(nested), "llada-moe")])
        self.assertEqual([spec for spec, _ in models.catalog()], [str(self.dream), str(nested)])


class PromptTest(unittest.TestCase):
    def test_single_turn_passes_through_and_history_is_labelled(self):
        self.assertEqual(prompt.flatten([{"role": "system", "content": "Be brief."},
                                         {"role": "user", "content": "Hi"}]), ("Be brief.", "Hi"))
        system, user = prompt.flatten([
            {"role": "user", "content": [{"type": "text", "text": "List files"}]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "bash", "content": "a.txt"},
        ], tools=[{"type": "function", "function": {"name": "bash", "description": "Run", "parameters": {}}}])
        self.assertIn('"name": "bash"', system)
        self.assertTrue(system.startswith(prompt.TOOL_INSTRUCTIONS))
        self.assertIn("User: List files", user)
        self.assertIn('Assistant: <tool_call>{"name": "bash", "arguments": {"cmd": "ls"}}</tool_call>', user)
        self.assertTrue(user.endswith("Tool result (bash):\na.txt\n\nReply as the assistant."))

    def test_only_offered_tools_become_calls(self):
        tools = [{"type": "function", "function": {"name": "read"}}]
        content, calls = prompt.tool_calls('Sure.\n<tool_call>{"name": "read", "arguments": {"path": "x"}}</tool_call>'
                                           '<tool_call>{"name": "rm", "arguments": {}}</tool_call>', tools)
        self.assertEqual([json.loads(c["function"]["arguments"]) for c in calls], [{"path": "x"}])
        self.assertEqual(content, "Sure.")
        self.assertEqual(prompt.tool_calls('{"name": "read", "arguments": "{\\"path\\": \\"y\\"}"}', tools)[1][0]
                         ["function"]["arguments"], '{"path": "y"}')
        self.assertEqual(prompt.tool_calls("<tool_call>not json</tool_call>", tools)[1], [])
        self.assertEqual(prompt.tool_calls('{"name": "read", "arguments": {}}', None),
                         ('{"name": "read", "arguments": {}}', []))


class DiffusionTest(Fixture):
    def model(self, **kwargs):
        return models.Model(str(self.dream), **kwargs)

    def test_reply_is_read_from_the_log_file_with_exact_arguments(self):
        os.environ["PUZZLLM_GPU"] = "Vulkan9"
        reply, calls = self.model().reply([{"role": "system", "content": "Rule"},
                                           {"role": "user", "content": "hello"}], max_tokens=40)
        self.assertEqual((reply.text, calls, reply.prompt_tokens, reply.completion_tokens), ("Echo: hello", [], 71, 40))
        (args, system, text, libraries), = self.runs()
        option = lambda name: args[args.index(name) + 1]
        self.assertEqual((system, text), ("Rule", "hello"))
        self.assertEqual([option(o) for o in ("-c", "-b", "-ub")], ["111"] * 3)
        self.assertEqual((option("--diffusion-steps"), option("--diffusion-algorithm"), option("--temp"),
                          option("--device"), option("-m")), ("20", "1", "0", "Vulkan9", str(self.dream.resolve())))
        self.assertIn("--no-log-prefix", args)
        self.assertNotIn("--diffusion-visual", args)
        self.assertEqual(libraries.split(os.pathsep)[0], str(self.tools.resolve()))

    def test_visual_frames_reach_the_caller_and_end_markers_are_cut(self):
        os.environ["CHAT_REPLY"] = "abcdefgh"
        frames = []
        reply, _ = self.model().reply([{"role": "user", "content": "x"}], max_tokens=16, steps=8,
                                      on_partial=lambda text, step, steps: frames.append((step, steps, text)))
        self.assertEqual(reply.text, "abcdefgh")
        self.assertTrue(frames and all(total == 8 for _, total, _ in frames))
        self.assertEqual([s for s, _, _ in frames], sorted(s for s, _, _ in frames))
        self.assertTrue(all("total time" not in t for _, _, t in frames))
        self.assertIn("--diffusion-visual", self.runs()[0][0])

    def test_failures_report_llama_cpp_and_not_the_driver(self):
        os.environ["CHAT_FAIL"] = "1"
        with self.assertRaisesRegex(RuntimeError, r"^llama-diffusion-cli exited 1\. error: failed to load model$"):
            self.model().reply([{"role": "user", "content": "x"}])

    def test_long_conversations_drop_their_oldest_turns(self):
        self.counter.write_text("#!/bin/sh\nwc -c < \"$4\" | sed 's/^/Total number of tokens: /'\n")
        history = [{"role": "system", "content": "S"}]
        for n in range(40):
            history += [{"role": "user", "content": f"question {n} " + "x" * 80},
                        {"role": "assistant", "content": "answer " + "y" * 80}]
        history.append({"role": "user", "content": "last"})
        reply, _ = self.model().reply(history, max_tokens=256)
        self.assertEqual(reply.text, "Echo: last")
        args, system, text, _ = self.runs()[-1]
        self.assertEqual(system, "S")
        self.assertNotIn("question 0 ", text)
        self.assertIn("question 39 ", text)
        self.assertLessEqual(int(args[args.index("-ub") + 1]), 2048)

    def test_chat_models_and_missing_tools_are_refused(self):
        chat = self.models / "chat.gguf"
        chat.write_bytes(gguf_bytes("qwen3"))
        with self.assertRaisesRegex(ValueError, "llama-diffusion-cli runs dream, llada, llada-moe, rnd1 models"):
            models.Model(str(chat))
        self.cli.unlink()
        with self.assertRaisesRegex(RuntimeError, "Install llama-diffusion-cli"):
            self.model()
        self.assertEqual(self.runs(), [])

    def test_without_a_tokenizer_the_byte_length_bounds_the_prompt(self):
        self.counter.unlink()
        self.model().reply([{"role": "user", "content": "café"}], max_tokens=32)
        args = self.runs()[0][0]
        self.assertEqual(args[args.index("-ub") + 1], str(len("\ncafé".encode()) + 64 + 32))


class ServerTest(Fixture):
    def start(self, cap=512):
        model = models.Model(str(self.dream))
        ready = threading.Event()
        holder = {}
        def up(instance):
            holder["server"] = instance
            ready.set()
        thread = threading.Thread(target=server.serve, args=(model,), daemon=True,
                                  kwargs=dict(alias="dream-test", port=0, max_tokens_cap=cap, ready=up))
        thread.start()
        self.assertTrue(ready.wait(5))
        self.addCleanup(holder["server"].shutdown)
        return f"http://127.0.0.1:{holder['server'].server_address[1]}"

    def post(self, url, body):
        request = Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            return response.read().decode()

    def test_health_models_and_plain_completion(self):
        url = self.start(cap=100)
        with urlopen(url + "/health", timeout=5) as response:
            self.assertEqual(json.load(response), {"status": "ok"})
        with urlopen(url + "/v1/models", timeout=5) as response:
            self.assertEqual([m["id"] for m in json.load(response)["data"]], ["dream-test"])
        result = json.loads(self.post(url, {"model": "dream-test", "max_tokens": 4096,
                                            "messages": [{"role": "user", "content": "hi"}]}))
        choice = result["choices"][0]
        self.assertEqual((choice["message"]["content"], choice["finish_reason"]), ("Echo: hi", "stop"))
        self.assertEqual(result["usage"]["completion_tokens"], 100)
        args = self.runs()[0][0]
        self.assertEqual(args[args.index("-ub") + 1], str(7 + 64 + 100))

    def test_streamed_tool_calls(self):
        os.environ["CHAT_REPLY"] = '<tool_call>{"name": "read", "arguments": {"path": "GOAL.md"}}</tool_call>'
        url = self.start()
        text = self.post(url, {"stream": True, "messages": [{"role": "user", "content": "go"}],
                               "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}]})
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))
        calls = [c for e in events for c in e["choices"][0]["delta"].get("tool_calls", [])]
        self.assertEqual([(c["index"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                         [(0, "read", {"path": "GOAL.md"})])
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertIn("<tool_call>", self.runs()[0][1])

    def test_model_errors_are_http_500(self):
        os.environ["CHAT_FAIL"] = "1"
        url = self.start()
        with self.assertRaises(Exception) as caught:
            self.post(url, {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(caught.exception.code, 500)
        self.assertIn("failed to load model", json.loads(caught.exception.read())["error"]["message"])


class OllamaTest(unittest.TestCase):
    def test_streamed_chat_and_catalog(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"models":[{"name":"small:v2"},{"name":"a:1"}]}')
            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()
                for part in ({"message": {"content": "Hel"}}, {"message": {"content": "lo"}},
                             {"done": True, "prompt_eval_count": 5, "eval_count": 2}):
                    self.wfile.write(json.dumps(part).encode() + b"\n")
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as http:
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{http.server_address[1]}"):
                    self.assertEqual(ollama.local_models(), ["a:1", "small:v2"])
                    parts = []
                    reply, calls = models.Model("ollama:small:v2").reply(
                        [{"role": "user", "content": "hi"}], max_tokens=9,
                        on_partial=lambda text, step, steps: parts.append(text))
            finally:
                http.shutdown()
        self.assertEqual((reply.text, reply.prompt_tokens, reply.completion_tokens, calls), ("Hello", 5, 2, []))
        self.assertEqual(parts, ["Hel", "Hello", "Hello"])
        self.assertEqual((seen[0]["model"], seen[0]["options"]["num_predict"], seen[0]["think"]), ("small:v2", 9, False))


class WindowTest(Fixture):
    def test_keys_send_pick_and_cycle_lengths(self):
        chat = tui.Chat()
        chat.open_picker()
        self.assertEqual([label for _, label in chat.choices], ["dream coder.gguf  [dream]"])
        tui.handle(chat, "\n")
        self.assertFalse(chat.picking)
        for key in "hey":
            tui.handle(chat, key)
        tui.handle(chat, tui.curses.KEY_F3)
        self.assertEqual(chat.length, 512)
        tui.handle(chat, "\n")
        chat.worker.join(10)
        self.assertEqual([m["content"] for m in chat.messages], ["hey", "Echo: hey"])
        self.assertIsNone(chat.partial)
        args = self.runs()[0][0]
        self.assertEqual(args[args.index("-ub") + 1], str(7 + 64 + 512))
        tui.handle(chat, "\x0e")
        self.assertEqual(chat.messages, [])
        self.assertFalse(tui.handle(chat, "\x11"))

    def test_a_failed_reply_takes_the_message_back(self):
        os.environ["CHAT_FAIL"] = "1"
        chat = tui.Chat(str(self.dream))
        chat.entry = "x"
        chat.send()
        chat.worker.join(10)
        self.assertEqual(chat.messages, [])
        self.assertIn("failed to load model", chat.status)

    def test_transcript_wraps_and_labels_turns(self):
        rows = tui.transcript_lines([{"role": "user", "content": "one two three four"}], ("par", "(step 2/8)"), 12, "M")
        self.assertEqual(rows, [("You", "you"), ("one two", "text"), ("three four", "text"), ("", "text"),
                                ("M (step 2/8)", "them"), ("par", "partial")])

    def test_cli_lists_models_and_asks(self):
        launcher = PROJECT / "kilix-chat"
        listed = subprocess.run([launcher, "models"], capture_output=True, text=True, timeout=20, check=True)
        self.assertEqual(listed.stdout, f"dream coder.gguf  [dream]\t{self.dream}\n")
        asked = subprocess.run([launcher, "--model", str(self.dream), "--max-tokens", "24", "ask", "ping"],
                               capture_output=True, text=True, timeout=20)
        self.assertEqual((asked.returncode, asked.stdout), (0, "Echo: ping\n"))
        refused = subprocess.run([launcher], capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("needs a terminal", refused.stderr)


if __name__ == "__main__":
    unittest.main()
