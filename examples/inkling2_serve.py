#!/usr/bin/env python3
# OpenAI compatible chat completions server for examples/inkling2.py, with reasoning and tool calls, for agents like opencode
#   python examples/inkling2_serve.py --model /path/to/Inkling-2-Mini --port 8000
import json, time, uuid, argparse, threading, http.server, socketserver
from typing import Iterator
from pathlib import Path
from tinygrad import Tensor, GlobalCounters
from tinygrad.llm.serve import normalize_messages
from examples.inkling2 import load, Inkling2

def load_template(model_dir:Path):
  import jinja2, jinja2.ext
  class Generation(jinja2.ext.Extension):
    # transformers' {% generation %} marks the assistant tokens for training masks, it renders its body as is
    tags = {"generation"}
    def parse(self, parser):
      lineno = next(parser.stream).lineno
      body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
      return jinja2.nodes.Scope(body, lineno=lineno)
  env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols, Generation])
  env.filters["tojson"] = lambda obj, indent=None, separators=None, sort_keys=False: json.dumps(obj, indent=indent, separators=separators, sort_keys=sort_keys,
                                                                                               ensure_ascii=False)
  def raise_exception(msg): raise ValueError(msg)
  env.globals["raise_exception"] = raise_exception
  return env.from_string((model_dir / "chat_template.jinja").read_text())

class ResponseParser:
  """splits generated tokens into reasoning, content and tool calls, following the response_template in tokenizer_config.json"""
  def __init__(self, tok, special:dict[str, int]):
    self.tok, self.sp = tok, special
    self.kinds = {special["<|content_thinking|>"]: "reasoning_content", special["<|content_text|>"]: "content",
                  special["<|content_tool_error|>"]: "content", special["<|content_invoke_tool_text|>"]: "content",
                  special["<|content_invoke_tool_json|>"]: "tool"}
    self.kind: str|None = None  # None between messages, "header" for the name between <|message_model|> and the content type
    self.ids: list[int] = []    # tokens of the current message body
    self.sent = ""              # text of the current message body already streamed
    self.tool_calls: list[dict] = []
  def feed(self, t:int) -> list[tuple[str, str]]:
    """returns (field, text) deltas for the stream"""
    if t == self.sp["<|message_model|>"]: self.kind, self.ids, self.sent = "header", [], ""; return []
    if t in self.kinds: self.kind, self.ids, self.sent = self.kinds[t], [], ""; return []
    if t == self.sp["<|end_message|>"]:
      out = self.flush(final=True)
      self.kind = None
      return out
    if t >= 199998: return []  # other special tokens never carry text
    if self.kind is None: self.kind, self.ids, self.sent = "content", [], ""  # text outside of a message is content
    self.ids.append(t)
    return self.flush()
  def flush(self, final:bool=False) -> list[tuple[str, str]]:
    if self.kind is None or (self.kind == "header" and not final): return []
    if self.kind == "header": self.kind = "content"  # a message without a content type: its text is content
    text = self.tok.decode(self.ids, skip_special_tokens=False)
    if self.kind == "tool":
      if not final: return []
      try:
        call = json.loads(text)
        self.tool_calls.append({"index": len(self.tool_calls), "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                                "function": {"name": call["name"], "arguments": json.dumps(call.get("args", {}))}})
        return []
      except (json.JSONDecodeError, KeyError, TypeError): return [("content", text)]  # don't drop output the client can't use
    if not final and text.endswith("�"): return []  # wait for the rest of a multi byte character
    delta, self.sent = text[len(self.sent):], text
    return [(self.kind, delta)] if delta else []

class Session:
  """one conversation's worth of cached state: the tokens the model has seen, and conv state snapshots at earlier prompt ends"""
  def __init__(self, model:Inkling2): self.model, self.seen, self.snapshots = model, [], {}
  def prefill(self, ids:list[int]) -> Iterator[tuple[int, int, Tensor|None]]:
    """yields (start, done, logits) as the prompt is prefilled, logits is set on the last one"""
    common = next((i for i, (a, b) in enumerate(zip(self.seen, ids)) if a != b), min(len(self.seen), len(ids)))
    # a snapshot is the state after the tokens before it, the ones past the first difference were from another sequence
    for p in [p for p in self.snapshots if p > common]: del self.snapshots[p]
    start = min(common, len(ids) - 1)  # at least one token has to run to get logits
    if start < len(self.seen):  # rewind: the keys and values can be overwritten in place, the conv states come from a snapshot
      start = max((p for p in self.snapshots if p <= start), default=0)
      self.model.restore(self.snapshots.get(start))
    self.seen = ids[:start]
    if start < len(ids) - 1:
      for i, _ in enumerate(self.model.run_chunks(ids[start:-1], start)): yield start, i, None
    # snapshot before the last token, so the same prompt (a retry) or one that continues it can resume here
    self.snapshots[len(ids) - 1] = self.model.snapshot()
    for p in sorted(self.snapshots)[:-8]: del self.snapshots[p]
    self.seen = list(ids)
    yield start, len(ids), self.model(ids[-1:], len(ids) - 1)
  def step(self, t:int) -> Tensor:
    logits = self.model([t], len(self.seen))
    self.seen.append(t)
    return logits

class Handler(http.server.BaseHTTPRequestHandler):
  server: "Server"
  protocol_version = "HTTP/1.1"
  def log_message(self, *args): pass
  def send_json(self, obj, code=200):
    body = json.dumps(obj).encode()
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)
  def do_GET(self):
    if self.path.rstrip("/").endswith("/models"): self.send_json({"object": "list", "data": [{"id": self.server.name, "object": "model"}]})
    else: self.send_json({"error": {"message": f"unknown path {self.path}"}}, 404)
  def do_POST(self):
    if not self.path.rstrip("/").endswith("/chat/completions"): return self.send_json({"error": {"message": f"unknown path {self.path}"}}, 404)
    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
    if self.server.log_requests is not None: (self.server.log_requests / f"{time.time_ns()}.json").write_text(json.dumps(body, indent=1))
    normalize_messages(body["messages"])
    effort = body.get("reasoning_effort", self.server.effort)
    try: prompt = self.server.template.render(messages=body["messages"], tools=body.get("tools") or None, reasoning_effort=effort)
    except Exception as e: return self.send_json({"error": {"message": f"{type(e).__name__}: {e}", "type": "invalid_request_error"}}, 400)
    ids = self.server.tok.encode(prompt, add_special_tokens=False).ids
    room = self.server.model.max_context - len(ids)
    if room <= 1:
      return self.send_json({"error": {"message": f"prompt has {len(ids)} tokens, the context is {self.server.model.max_context}",
                                       "type": "invalid_request_error", "code": "context_length_exceeded"}}, 400)
    max_tokens = min(body.get("max_completion_tokens") or body.get("max_tokens") or room, room)
    temperature = body.get("temperature")
    chunks = self.server.generate(ids, max_tokens, self.server.temperature if temperature is None else float(temperature))
    meta = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "created": int(time.time()), "model": body.get("model", self.server.name)}
    if body.get("stream"):
      self.send_response(200)
      self.send_header("Content-Type", "text/event-stream")
      self.send_header("Cache-Control", "no-cache")
      self.end_headers()
      include_usage = (body.get("stream_options") or {}).get("include_usage", False)
      try:
        self.wfile.write(f"data: {json.dumps({**meta, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n".encode())
        for c in chunks:
          if c["type"] == "keepalive":  # clients drop a stream that sends nothing for a while (node's fetch after 300 s), a long prefill can take longer
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()
            continue
          if c["type"] == "delta": choice = {"index": 0, "delta": c["delta"], "finish_reason": None}
          else: choice = {"index": 0, "delta": {}, "finish_reason": c["finish_reason"]}
          self.wfile.write(f"data: {json.dumps({**meta, 'object': 'chat.completion.chunk', 'choices': [choice]})}\n\n".encode())
          self.wfile.flush()
          if c["type"] == "done" and include_usage:
            self.wfile.write(f"data: {json.dumps({**meta, 'object': 'chat.completion.chunk', 'choices': [], 'usage': c['usage']})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
      except (BrokenPipeError, ConnectionResetError): pass
      finally: chunks.close()  # a client that went away mid stream must not leave the generator holding the model lock
      self.close_connection = True
    else:
      message: dict = {"role": "assistant", "content": ""}
      for c in chunks:
        if c["type"] == "done": finish_reason, usage = c["finish_reason"], c["usage"]
        elif c["type"] == "delta":
          for k, v in c["delta"].items():
            if k == "tool_calls": message.setdefault("tool_calls", []).extend({kk: vv for kk, vv in tc.items() if kk != "index"} for tc in v)
            else: message[k] = message.get(k, "") + v
      self.send_json({**meta, "object": "chat.completion", "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}], "usage": usage})

class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
  daemon_threads, allow_reuse_address = True, True
  def __init__(self, addr, model:Inkling2, tok, template, name:str, effort, temperature:float):
    super().__init__(addr, Handler)
    self.model, self.tok, self.template, self.name, self.effort, self.temperature = model, tok, template, name, effort, temperature
    self.special = {t.content: i for i, t in tok.get_added_tokens_decoder().items()}
    self.stops = {self.special["<|content_model_end_sampling|>"], self.special["<|endoftext|>"]}
    self.session, self.lock, self.log_requests = Session(model), threading.Lock(), None

  def generate(self, ids:list[int], max_tokens:int, temperature:float):
    while not self.lock.acquire(timeout=10): yield {"type": "keepalive"}  # one generation at a time, they share the model's cache
    try:
      st = time.perf_counter()
      for start, _, logits in self.session.prefill(ids): yield {"type": "keepalive"}
      assert logits is not None
      pt = time.perf_counter()
      parser, n, finish_reason = ResponseParser(self.tok, self.special), 0, "length"
      while n < max_tokens:
        t = int((logits.argmax() if temperature == 0 else (logits / temperature).softmax().multinomial()).item())
        n += 1
        if t in self.stops: finish_reason = "stop"; break
        for field, text in parser.feed(t): yield {"type": "delta", "delta": {field: text}}
        logits = self.session.step(t)
      for field, text in parser.flush(final=True): yield {"type": "delta", "delta": {field: text}}
      if parser.tool_calls:
        yield {"type": "delta", "delta": {"tool_calls": parser.tool_calls}}
        if finish_reason == "stop": finish_reason = "tool_calls"
      et = time.perf_counter()
      print(f"prompt {len(ids):6d} (reused {start:6d}, prefill {(len(ids)-start)/(pt-st):6.1f} tok/s)  out {n:5d} ({n/(et-pt) if n else 0:5.1f} tok/s)  "
            f"{finish_reason}", flush=True)
      yield {"type": "done", "finish_reason": finish_reason, "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n,
                                                                       "prompt_tokens_details": {"cached_tokens": start}}}
    finally: self.lock.release()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Serve Inkling 2 Mini with an OpenAI compatible API")
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--max_context", type=int, default=32768)
  parser.add_argument("--effort", default="low", help="default thinking effort: none, minimal, low, medium, high, max or a number in [0, 0.99]")
  parser.add_argument("--temperature", type=float, default=0.6, help="used when a request doesn't set one")
  parser.add_argument("--name", default="inkling-2-mini")
  parser.add_argument("--log_requests", type=Path, help="directory to save each request body in")
  args = parser.parse_args()
  from tokenizers import Tokenizer
  st = time.perf_counter()
  model = load(args.model, args.max_context)
  print(f"loaded weights in {time.perf_counter()-st:.2f} s, {GlobalCounters.mem_used/1e9:.2f} GB")
  effort = args.effort if not args.effort.replace(".", "", 1).isdigit() else float(args.effort)
  server = Server(("127.0.0.1", args.port), model, Tokenizer.from_file(str(args.model / "tokenizer.json")), load_template(args.model), args.name, effort,
                  args.temperature)
  server.log_requests = args.log_requests
  if args.log_requests is not None: args.log_requests.mkdir(parents=True, exist_ok=True)
  print(f"serving {args.name} on http://127.0.0.1:{args.port}/v1")
  server.serve_forever()
