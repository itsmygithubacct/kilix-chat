#!/usr/bin/env python3
"""kilix-chat offline: GGUF metadata, prompts, tool calls, the diffusion
runner against a stand-in llama-diffusion-cli, the server, Ollama, and the
chat window's keys, slash commands and memory."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import kilix_chat  # noqa: E402
from kilix_chat import commands, diffusion, gguf, memory, models, ollama, prompt, server, tui  # noqa: E402


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
import json, os, sys, time
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
if os.environ.get('CHAT_FAIL') == 'memory':
    entry('alloc_tensor_range: failed to allocate Vulkan1 buffer of size 1054838784\n')
    entry('error: failed to load model\n')
    sys.exit(1)
if os.environ.get('CHAT_FAIL'):
    entry('error: failed to load model\n')
    sys.exit(1)
if os.environ.get('CHAT_SLEEP'):
    time.sleep(float(os.environ['CHAT_SLEEP']))
    open(os.environ['CHAT_CALLS'] + '.finished', 'a').close()
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
               "PUZZLLM_GPU": "", "OLLAMA_HOST": "127.0.0.1:9", "CHAT_REPLY": "", "CHAT_FAIL": "", "CHAT_SLEEP": "",
               "XDG_DATA_HOME": str(self.root / "data")}
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
                          option("--device"), option("-m")), ("20", "4", "0.1", "Vulkan9", str(self.dream.resolve())))
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
        os.environ["CHAT_FAIL"] = "memory"
        with self.assertRaisesRegex(RuntimeError, r"^There is not enough free GPU memory for dream coder\.gguf\. "
                                                  r"Another model may be loaded \(see `ollama ps`\)\.$"):
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


class Served(Fixture):
    """Starts `kilix-chat serve` over the stand-in model."""
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


class ServerTest(Served):
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


class OllamaServeTest(Served):
    """The Ollama dialect, requested the way Kilix Avatar's LocalOllama does."""
    AVATAR_OPTIONS = {"num_ctx": 8192, "num_predict": 2048}

    def request(self, url, method, path, body=None):
        """(status, content type, parsed NDJSON or JSON lines) over http.client, as Avatar reads them."""
        port = int(url.rsplit(":", 1)[1])
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        data = json.dumps(body).encode() if body is not None else None
        connection.request(method, path, data, {"Content-Type": "application/json"} if data else {})
        response = connection.getresponse()
        lines = []
        while True:
            line = response.readline(65537)
            if not line:
                break
            if line.strip():
                lines.append(json.loads(line))
        connection.close()
        return response.status, response.getheader("Content-Type"), lines

    def chat(self, url, model="dream-test", **extra):
        body = {"model": model, "messages": [{"role": "system", "content": "Be Kilix."},
                                             {"role": "user", "content": "hi"}],
                "stream": True, "think": False, "keep_alive": "5m", "options": dict(self.AVATAR_OPTIONS)}
        body.update(extra)
        return self.request(url, "POST", "/api/chat", body)

    def test_tags_version_and_pull_answer_only_for_the_served_model(self):
        url = self.start()
        status, kind, (tags,) = self.request(url, "GET", "/api/tags")
        record, = tags["models"]
        self.assertEqual((status, kind, record["name"], record["model"], record["size"]),
                         (200, "application/json", "dream-test", "dream-test", self.dream.stat().st_size))
        self.assertRegex(record["digest"], r"^[0-9a-f]{64}$")
        self.assertEqual((record["details"]["family"], record["details"]["format"]), ("dream", "gguf"))
        self.assertTrue(record["modified_at"].endswith("Z"))
        self.assertEqual(self.request(url, "GET", "/api/version")[2], [{"version": kilix_chat.__version__}])
        for name in ("dream-test", "dream-test:latest"):
            self.assertEqual(self.request(url, "POST", "/api/pull", {"model": name, "stream": True}),
                             (200, "application/x-ndjson", [{"status": "success"}]))
        self.assertEqual(self.request(url, "POST", "/api/pull", {"name": "dream-test", "stream": False})[2],
                         [{"status": "success"}])
        for path in ("/api/pull", "/api/chat"):
            status, _, (error,) = self.request(url, "POST", path, {"model": "llama3", "stream": True,
                                                                  "messages": [{"role": "user", "content": "x"}]})
            self.assertEqual(status, 404, path)
            self.assertEqual(error, {"error": 'model "llama3" not found; this kilix-chat serves only "dream-test" '
                                              "and does not download models"})
        self.assertEqual(self.request(url, "POST", "/api/chat", {"messages": []})[:2], (400, "application/json"))
        self.assertEqual(self.runs(), [])

    def test_chat_streams_ndjson_with_ollama_done_fields(self):
        url = self.start(cap=100)
        status, kind, lines = self.chat(url)
        self.assertEqual((status, kind), (200, "application/x-ndjson"))
        self.assertEqual("".join(line["message"]["content"] for line in lines), "Echo: hi")
        self.assertTrue(all(line["model"] == "dream-test" and line["done"] is (line is lines[-1]) for line in lines))
        done = lines[-1]
        self.assertEqual((done["done_reason"], done["prompt_eval_count"], done["eval_count"]), ("stop", 7 + 64, 100))
        self.assertIsInstance(done["total_duration"], int)
        args, system, text, _ = self.runs()[-1]
        self.assertEqual((system, text, args[args.index("-ub") + 1]), ("Be Kilix.", "hi", str(7 + 64 + 100)))
        self.chat(url, options={"num_predict": 20})
        self.assertEqual(self.runs()[-1][0][self.runs()[-1][0].index("-ub") + 1], str(7 + 64 + 20))
        unstreamed = self.chat(url, stream=False)
        self.assertEqual((unstreamed[1], len(unstreamed[2])), ("application/json", 1))
        self.assertEqual((unstreamed[2][0]["message"]["content"], unstreamed[2][0]["done"]), ("Echo: hi", True))
        body = {"model": "dream-test", "messages": [{"role": "user", "content": "default"}]}
        status, kind, lines = self.request(url, "POST", "/api/chat", body)
        self.assertEqual((kind, lines[-1]["done"], lines[-2]["message"]["content"]),
                         ("application/x-ndjson", True, "Echo: default"))

    def test_a_slow_reply_keeps_the_stream_alive_and_a_failure_is_an_error_line(self):
        os.environ["CHAT_SLEEP"] = "0.6"
        url = self.start()
        with patch.object(server, "KEEPALIVE_SECONDS", 0.1):
            lines = self.chat(url)[2]
        alive = [line for line in lines[:-2] if line["message"]["content"] == "" and line["done"] is False]
        self.assertGreaterEqual(len(alive), 3)
        self.assertEqual(lines[-2]["message"]["content"], "Echo: hi")
        os.environ.update(CHAT_SLEEP="", CHAT_FAIL="1")
        url = self.start()   # the model reads its environment when it opens
        status, _, lines = self.chat(url)
        self.assertEqual(status, 200)
        self.assertIn("failed to load model", lines[-1]["error"])
        status, _, (error,) = self.chat(url, stream=False)
        self.assertEqual(status, 500)
        self.assertIn("failed to load model", error["error"])

    def test_a_client_that_leaves_stops_the_reply(self):
        os.environ["CHAT_SLEEP"] = "3"
        url = self.start()
        body = json.dumps({"model": "dream-test", "messages": [{"role": "user", "content": "x"}]}).encode()
        with patch.object(server, "KEEPALIVE_SECONDS", 0.05):
            client = socket.create_connection(("127.0.0.1", int(url.rsplit(":", 1)[1])), timeout=10)
            client.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                           + f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
            received = b""
            while b'"done": false' not in received:
                chunk = client.recv(4096)
                if not chunk:
                    self.fail(f"the server closed the stream: {received!r}")
                received += chunk
            client.close()
            time.sleep(4)
        self.assertEqual(len(self.runs()), 1)
        self.assertFalse(Path(str(self.calls) + ".finished").exists())
        os.environ["CHAT_SLEEP"] = ""
        self.assertEqual(self.chat(url)[2][-2]["message"]["content"], "Echo: hi")

    def test_an_unexpected_failure_is_still_answered(self):
        url = self.start()
        with patch.object(models.Model, "reply", side_effect=KeyError("boom")):
            status, _, lines = self.chat(url)
            self.assertEqual((status, lines[-1]), (200, {"error": "kilix-chat failed: KeyError: 'boom'"}))
            status, _, (error,) = self.chat(url, stream=False)
            self.assertEqual((status, error), (500, {"error": "kilix-chat failed: KeyError: 'boom'"}))

    def test_tool_calls_take_ollama_shapes(self):
        os.environ["CHAT_REPLY"] = '<tool_call>{"name": "read", "arguments": {"path": "GOAL.md"}}</tool_call>'
        url = self.start()
        tools = [{"type": "function", "function": {"name": "read", "parameters": {}}}]
        lines = self.chat(url, tools=tools, messages=[
            {"role": "user", "content": "go"},
            {"role": "tool", "tool_name": "read", "content": "a.txt"}])[2]
        self.assertEqual(lines[-2]["message"]["tool_calls"], [{"function": {"name": "read",
                                                                             "arguments": {"path": "GOAL.md"}}}])
        self.assertIn("Tool result (read):\na.txt", self.runs()[-1][2])

    def test_serve_listens_on_loopback_only(self):
        for host in ("127.0.0.1", "127.0.0.2", "::1", "localhost"):
            self.assertTrue(server.is_loopback(host), host)
        for host in ("0.0.0.0", "192.168.1.2", "::", "example.org"):
            self.assertFalse(server.is_loopback(host), host)
        def bound(instance):
            # Were it to bind, stop it at once so the test fails rather than hangs.
            threading.Thread(target=instance.shutdown, daemon=True).start()
        with self.assertRaisesRegex(ValueError, "loopback only"):
            server.serve(models.Model(str(self.dream)), alias="x", host="0.0.0.0", port=0, ready=bound)
        refused = subprocess.run([PROJECT / "kilix-chat", "--model", str(self.dream), "serve", "--host", "0.0.0.0"],
                                 capture_output=True, text=True, timeout=20)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("serve listens on loopback only", refused.stderr)


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


WORD_COUNTER = "#!/bin/sh\nwc -c < \"$4\" | sed 's/^/Total number of tokens: /'\n"


class Screen:
    """Enough of a curses window to draw into and read back."""
    def __init__(self, height=20, width=100):
        self.size, self.rows = (height, width), {}
    def getmaxyx(self):
        return self.size
    def erase(self):
        self.rows = {}
    def addnstr(self, y, x, text, n, attr=0):
        self.rows[y] = text[:n]
    def move(self, y, x):
        pass


class CommandTest(Fixture):
    def chat(self, **kwargs):
        return tui.Chat(str(self.dream), **kwargs)

    def enter(self, chat, line):
        """Type a line and press Enter. Return what handle returned."""
        chat.entry = line
        result = tui.handle(chat, "\n")
        if chat.worker:
            chat.worker.join(10)
        return result

    def test_quit_and_exit_leave_and_unknown_commands_never_reach_the_model(self):
        chat = self.chat()
        for line in ("/quit", "/exit", "/Q", " /exit "):
            self.assertFalse(self.enter(chat, line), line)
        self.assertTrue(self.enter(chat, "/frobnicate now"))
        self.assertIn("Unknown command /frobnicate. /help lists the commands.", chat.status)
        self.assertEqual(chat.entry, "/frobnicate now")
        tui.handle(chat, "\x15")
        self.assertEqual(chat.entry, "")
        self.assertEqual((chat.messages, self.runs()), ([], []))
        self.enter(chat, "//etc/hosts lists hosts")
        self.assertEqual(chat.messages[0], {"role": "user", "content": "/etc/hosts lists hosts"})
        self.assertEqual(self.runs()[0][2], "/etc/hosts lists hosts")

    def test_model_opens_the_picker_or_switches_to_the_named_model(self):
        other = self.models / "llada.gguf"
        other.write_bytes(gguf_bytes("llada", "LLaDA Small", 4096))
        chat = self.chat()
        chat.messages = [{"role": "user", "content": "old"}]
        self.enter(chat, "/model")
        self.assertTrue(chat.picking)
        tui.handle(chat, "\x1b")
        self.assertFalse(chat.picking)
        self.enter(chat, f"/model {other}")
        self.assertEqual((chat.model.name, chat.messages, chat.picking), ("LLaDA Small", [], False))
        self.assertIn("Chatting with LLaDA Small", chat.status)
        self.enter(chat, "/model /no/such.gguf")
        self.assertEqual(chat.model.name, "LLaDA Small")
        self.assertIn("Could not open the model", chat.status)

    def test_commands_that_change_the_chat_wait_for_the_reply(self):
        chat = self.chat()
        chat.messages = [{"role": "user", "content": "keep"}]
        hold = threading.Event()
        chat.worker = threading.Thread(target=hold.wait, daemon=True)
        chat.worker.start()
        self.addCleanup(hold.set)
        for line in ("/clear", "/new", "/retry", "/context"):
            chat.entry = line
            tui.handle(chat, "\n")
            self.assertIn("waits for the reply", chat.status)
        self.assertEqual(chat.messages, [{"role": "user", "content": "keep"}])
        chat.entry = "/length 64"
        tui.handle(chat, "\n")
        self.assertEqual(chat.length, 64)

    def test_context_counts_the_prompt_the_reply_and_the_dropped_turns(self):
        self.counter.write_text(WORD_COUNTER)
        memory.add("Call me Sam.")
        chat = self.chat()
        chat.system = "Be brief."
        for n in range(30):
            chat.messages += [{"role": "user", "content": f"question {n} " + "x" * 80},
                              {"role": "assistant", "content": "answer " + "y" * 80}]
        def tokens(history):
            system, user = prompt.flatten(history)
            return len((system + "\n" + user).encode()) + diffusion.TEMPLATE_TOKENS
        history = chat.conversation()
        self.assertEqual(history[0]["content"], "Be brief.\n\n" + memory.HEADER + "\n- Call me Sam.")
        full, dropped = tokens(history), 0
        while tokens(history) + 256 > 2048:
            del history[1]
            dropped += 1
        self.assertGreater(dropped, 1)
        self.enter(chat, "/context")
        title, lines = chat.info
        self.assertEqual(title, "Context")
        self.assertEqual(lines, [
            "Context   2,048 tokens",
            f"Prompt    {tokens(history):,} tokens, counted by llama-tokenize; system prompt and memory included",
            "Reply     256 tokens reserved (/length)",
            f"Free      {2048 - 256 - tokens(history):,} tokens",
            f"Turns     60 in this chat; the oldest {dropped} would be dropped (the whole chat needs {full:,} tokens)"])
        chat.messages = chat.messages[-2:]
        self.enter(chat, "/length 64")
        self.enter(chat, "/context")
        self.assertIn(f"Prompt    {tokens(chat.conversation()):,} tokens", chat.info[1][1])
        self.assertEqual(chat.info[1][2:], ["Reply     64 tokens reserved (/length)",
                                            f"Free      {2048 - 64 - tokens(chat.conversation()):,} tokens",
                                            "Turns     2 in this chat; none would be dropped"])
        chat.messages = [{"role": "user", "content": "z" * 3000}]
        self.enter(chat, "/context")
        self.assertEqual(chat.info[1][-1], "Even the newest turn does not fit: shorten it, or the reply with /length.")
        tui.handle(chat, "\x1b")
        self.assertIsNone(chat.info)

    def test_memory_is_kept_across_chats_and_joins_the_system_prompt(self):
        chat = self.chat()
        self.enter(chat, "/memory add The cat is   called Miso.")
        self.enter(chat, "/remember Use metric units")
        self.assertEqual((self.root / "data/kilix-chat/memory.md").read_text(),
                         "- The cat is called Miso.\n- Use metric units\n")
        fresh = self.chat()
        self.enter(fresh, "/memory")
        self.assertEqual(fresh.info[1][:2], ["1. The cat is called Miso.", "2. Use metric units"])
        self.enter(fresh, "/system Be terse.")
        self.enter(fresh, "hello")
        self.assertEqual(self.runs()[-1][1], "Be terse.\n\n" + memory.HEADER
                         + "\n- The cat is called Miso.\n- Use metric units")
        self.enter(fresh, "/memory forget 1")
        self.assertEqual(memory.notes(), ["Use metric units"])
        self.enter(fresh, "/memory forget 9")
        self.assertIn("There is no note 9", fresh.status)
        self.enter(fresh, "/memory clear")
        self.assertEqual(memory.notes(), [])
        self.assertIn("Forgot 1 note.", fresh.status)
        self.enter(fresh, "/memory add   ")
        self.assertEqual(memory.notes(), [])
        self.assertIn("Say what to remember", fresh.status)
        self.enter(fresh, "again")
        self.assertEqual(self.runs()[-1][1], "Be terse.")

    def test_new_chat_and_system_prompt(self):
        chat = self.chat()
        for line in ("/new", "/clear"):
            chat.messages = [{"role": "user", "content": "x"}]
            self.enter(chat, line)
            self.assertEqual(chat.messages, [], line)
        self.enter(chat, "/system You are a pirate.")
        self.enter(chat, "/system")
        self.assertEqual(chat.info, ("System prompt", ["You are a pirate."]))
        self.enter(chat, "hi")
        self.assertEqual(self.runs()[-1][1], "You are a pirate.")
        self.enter(chat, "/system clear")
        self.enter(chat, "hi")
        self.assertEqual(self.runs()[-1][1], "")
        self.assertNotIn("-sys", self.runs()[-1][0])

    def test_length_and_steps_reach_llama_diffusion_cli(self):
        chat = self.chat()
        self.enter(chat, "/length 40")
        self.enter(chat, "/steps 5")
        for line in ("/length 3", "/length many", "/steps 0"):
            self.enter(chat, line)
        self.assertEqual((chat.length, chat.steps), (40, 5))
        self.assertIn("from 1 to 4096", chat.status)
        self.enter(chat, "one")
        option = lambda name: self.runs()[-1][0][self.runs()[-1][0].index(name) + 1]
        self.assertEqual((option("-ub"), option("--diffusion-steps")), (str(7 + 64 + 40), "5"))
        self.enter(chat, "/steps auto")
        self.enter(chat, "two")
        self.assertEqual(option("--diffusion-steps"), "20")
        for start, after in ((40, 64), (256, 512), (512, 64), (600, 64)):
            chat.length = start
            tui.handle(chat, tui.curses.KEY_F3)
            self.assertEqual(chat.length, after, start)

    def test_retry_undo_and_device(self):
        chat = self.chat()
        self.enter(chat, "one")
        self.enter(chat, "/retry")
        self.assertEqual([m["content"] for m in chat.messages], ["one", "Echo: one"])
        self.assertEqual(len(self.runs()), 2)
        self.enter(chat, "/undo")
        self.assertEqual((chat.messages, chat.entry), ([], "one"))
        self.enter(chat, "/retry")
        self.assertIn("no reply to retry", chat.status)
        self.enter(chat, "/device Vulkan7")
        self.enter(chat, "two")
        args = self.runs()[-1][0]
        self.assertEqual(args[args.index("--device") + 1], "Vulkan7")
        self.enter(chat, "/device auto")
        self.enter(chat, "three")
        self.assertNotIn("--device", self.runs()[-1][0])

    def test_save_writes_the_chat_as_markdown(self):
        chat = self.chat()
        self.enter(chat, "/save")
        self.assertIn("Nothing to save yet.", chat.status)
        chat.system = "Rule"
        self.enter(chat, "q")
        target = self.root / "out/chat.md"
        self.enter(chat, f"/save {target}")
        text = target.read_text()
        self.assertTrue(text.startswith("# Chat with Dream Coder\n"))
        self.assertIn("\n## System prompt\n\nRule\n\n## You\n\nq\n\n## Dream Coder\n\nEcho: q\n", text)
        self.assertIn(str(target), chat.status)
        self.enter(chat, "/save")
        saved = list((self.root / "data/kilix-chat/chats").glob("chat-*.md"))
        self.assertEqual([p.read_text() for p in saved], [text])

    def test_denoising_shows_masks_by_default_and_can_be_hidden(self):
        # Frames as llama-diffusion-cli writes them: one space per masked token.
        self.assertEqual(diffusion.show_masks("    "), "░░░░")
        self.assertEqual(diffusion.show_masks(" The sea stretches   the     ."), " The sea stretches░░ the░░░░ .")
        self.assertEqual(diffusion.show_masks("horizon.  \nIt   with"), "horizon.░░\nIt░░ with")
        os.environ["CHAT_REPLY"] = "abcdefgh"
        chat = self.chat()
        seen = []
        progress = chat._progress
        def spy(*args):
            progress(*args)
            seen.append(chat.partial)
        chat._progress = spy
        self.assertTrue(chat.visual)
        self.enter(chat, "one")
        self.assertIn("--diffusion-visual", self.runs()[-1][0])
        # The window polls, so it may see only some frames; the last is always shown.
        self.assertEqual(seen[-1], ("abcdefg░", "(step 127/128)"))
        self.assertTrue(all(len(text) == 8 and text.endswith("░") for text, _ in seen))
        self.assertEqual(chat.messages[-1]["content"], "abcdefgh")
        chat.entry = "half typed"
        tui.handle(chat, tui.curses.KEY_F4)
        self.assertEqual((chat.visual, chat.entry), (False, "half typed"))
        seen.clear()
        self.enter(chat, "two")
        self.assertNotIn("--diffusion-visual", self.runs()[-1][0])
        self.assertEqual((seen, chat.messages[-1]["content"]), ([], "abcdefgh"))
        self.enter(chat, "/visual sideways")
        self.assertIn("Use /visual on or /visual off.", chat.status)
        self.enter(chat, "/visual on")
        self.assertTrue(chat.visual)
        self.enter(chat, "/diffusion off")
        self.assertFalse(chat.visual)

    def test_a_tall_denoising_reply_keeps_its_start_in_view(self):
        chat = self.chat()
        chat.messages = [{"role": "user", "content": f"question {n}"} for n in range(20)]
        frame = "\n".join(f"line {n}" for n in range(30))
        screen = Screen(height=12, width=60)
        with patch.object(tui.curses, "color_pair", lambda n: 0):
            chat.partial = (frame, "(step 3/8)")
            tui.draw(screen, chat)
            self.assertEqual([screen.rows[y] for y in (1, 2, 9)], ["Dream Coder (step 3/8)", "line 0", "line 7"])
            chat.scroll = 2
            tui.draw(screen, chat)
            self.assertEqual(screen.rows[1], "question 19")
            chat.scroll = 0
            chat.partial = (frame, "")   # streamed text grows at its end
            tui.draw(screen, chat)
            self.assertEqual(screen.rows[9], "line 29")
            chat.partial = ("short", "(step 1/8)")   # a frame that fits shows like any reply
            tui.draw(screen, chat)
            self.assertEqual([screen.rows[y] for y in (8, 9)], ["Dream Coder (step 1/8)", "short"])

    def test_help_completion_and_the_hint_line(self):
        chat = self.chat()
        self.enter(chat, "/help")
        title, lines = chat.info
        text = "\n".join(lines)
        self.assertEqual(title, "Commands")
        for name in ("quit", "exit", "model", "context", "memory", "clear", "new", "system", "length", "steps",
                     "save", "help", "retry", "undo", "device", "visual", "diffusion"):
            self.assertIn("/" + name, text)
        self.assertIn("F2 model", text)
        for typed, completed in (("/mo", "/model "), ("/s", "/s"), ("/st", "/steps "), ("/re", "/re"),
                                 ("/rem", "/remember "), ("/memory c", "/memory clear "), ("/memory", "/memory "),
                                 ("/zz", "/zz"), ("hello", "hello"), ("//co", "//co")):
            self.assertEqual(commands.complete(typed), completed, typed)
        chat.entry = "/con"
        tui.handle(chat, "\t")
        self.assertEqual(chat.entry, "/context ")
        self.assertEqual(commands.hint("/s"), "/system  /steps  /save   (Tab completes)")
        self.assertEqual(commands.hint("/model x"), "/model [GGUF|ollama:NAME] — choose a model, or switch to the one named")
        self.assertEqual(commands.hint("/zz"), "Unknown command /zz. /help lists the commands.")
        screen = Screen()
        chat.status = "Ready."
        with patch.object(tui.curses, "color_pair", lambda n: 0):
            chat.entry = "/memo"
            tui.draw(screen, chat)
            self.assertEqual(screen.rows[18], commands.hint("/memo"))
            self.assertEqual(screen.rows[19], "> /memo")
            chat.entry = "hello"
            tui.draw(screen, chat)
            self.assertEqual(screen.rows[18], "Ready.")


class OllamaContextTest(unittest.TestCase):
    def serve(self, loaded, shown):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def reply(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def do_GET(self):
                self.reply({"models": loaded})
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.reply(shown)
        http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        self.addCleanup(http.server_close)
        self.addCleanup(http.shutdown)
        patcher = patch.dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{http.server_address[1]}")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_the_window_comes_from_the_loaded_model_then_its_parameters(self):
        shown = {"parameters": "num_keep 24\nstop \"<x>\"",
                 "model_info": {"general.architecture": "llama", "llama.context_length": 8192}}
        self.serve([{"name": "other:1", "context_length": 999}, {"name": "small:v2", "context_length": 4096}], shown)
        model = models.Model("ollama:small:v2")
        self.assertEqual(model.backend.context_length()[0], 4096)
        self.assertEqual(models.Model("ollama:big:1").backend.context_length(),
                         (8192, "the model's trained length; Ollama may load it with a smaller num_ctx"))
        shown["parameters"] = "num_ctx 2048"
        self.assertEqual(models.Model("ollama:big:1").backend.context_length()[0], 2048)
        history = [{"role": "system", "content": "abcd"}] + [{"role": "user", "content": "é" * 400}] * 30
        budget = models.Model("ollama:big:1").budget(history, max_tokens=256)
        per_turn = 800 // ollama.BYTES_PER_TOKEN + ollama.MESSAGE_TOKENS
        self.assertEqual((budget.context, budget.exact, budget.turns), (2048, False, 30))
        self.assertEqual(budget.full_tokens, 5 + 30 * per_turn)
        self.assertEqual(budget.dropped, 30 - (2048 - 256 - 5) // per_turn)
        self.assertEqual(budget.prompt_tokens, 5 + (30 - budget.dropped) * per_turn)


if __name__ == "__main__":
    unittest.main()
