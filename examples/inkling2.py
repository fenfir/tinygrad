#!/usr/bin/env python3
# Inkling 2 Mini (https://huggingface.co/thinkingmachines/Inkling-2-Mini), text decoder only
# port of transformers/models/inkling2_loop/modeling_inkling2_loop.py that runs the original NVFP4/MXFP8 checkpoint without unpacking it
# 24 layers in 4 blocks of 6, blocks 2 and 3 (layers 6-17) run 3 times, attention residuals across blocks, 128 experts with 8 active
import json, math, time, argparse
from typing import Iterator
from pathlib import Path
from tinygrad import Tensor, TinyJit, Variable, dtypes, Device, GlobalCounters
from tinygrad.nn.state import safe_load

def rms(x:Tensor, eps:float=1e-6) -> Tensor: return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()
def l2clip(x:Tensor, thr:float) -> Tensor: return x * (thr / (x.square().sum(-1, keepdim=True).sqrt() + 1e-6)).minimum(1.0)
def softcap(x:Tensor, cap:float) -> Tensor: return (x / cap).tanh() * cap

# ***** quantized experts *****

# small floats are decoded by moving their bits into a float32 and rebiasing the exponent with integer math, subnormal codes are exact too
# NOTE: rescaling with a large float constant instead is wrong, tinygrad folds it with other constants and (x*2**120)*2**10 overflows to inf
def bits_to_float(bits:Tensor) -> Tensor: return bits.cast(dtypes.uint32).bitcast(dtypes.float)
def e2m1(c:Tensor) -> Tensor:
  normal = bits_to_float(((c & 0x8) << 28) | (((c & 0x7) << 22) + (126 << 23)))
  return ((c & 0x6) == 0).where((c & 0x8 != 0).where(-0.5, 0.5) * (c & 1).cast(dtypes.float), normal)
def fp8e4m3(b:Tensor) -> Tensor:
  normal = bits_to_float(((b & 0x80) << 24) | (((b & 0x7F) << 20) + (120 << 23)))
  return ((b & 0x78) == 0).where((b & 0x80 != 0).where(-2.0**-9, 2.0**-9) * (b & 0x7).cast(dtypes.float), normal)

def to_cols_rows(t:Tensor) -> Tensor:
  """(experts, rows, cols) bytes -> (experts, cols, rows/4) uint32, so the threads computing adjacent rows load adjacent bytes"""
  return t.bitcast(dtypes.uint8).permute(0, 2, 1).contiguous().bitcast(dtypes.uint32).contiguous()

def unpack_bytes(w:Tensor) -> Tensor: return Tensor.stack(*[(w >> (8*j)) & 0xFF for j in range(4)], dim=-1).reshape(*w.shape[:-1], -1).cast(dtypes.int32)

def dequant(q:dict[str, Tensor], sel:Tensor) -> Tensor:
  """returns the float (len(sel), cols, rows) transposed weights of experts sel"""
  if "s" not in q: return q["w"][sel].float()  # unquantized
  b = unpack_bytes(q["w"][sel])
  n, R = b.shape[0], b.shape[-1]
  if "f" in q:  # NVFP4: two e2m1 per byte (even column in the low nibble), an fp8e4m3 scale per 16 values, a scale per expert
    w = Tensor.stack(e2m1(b & 0xF), e2m1(b >> 4), dim=2).reshape(n, -1, R)
    s = fp8e4m3(unpack_bytes(q["s"][sel])) * q["f"][sel].reshape(-1, 1, 1)
  else:  # MXFP8: fp8e4m3 values, an e8m0 scale (2**(s-127)) per 32 values
    w, s = fp8e4m3(b), bits_to_float(unpack_bytes(q["s"][sel]) << 23)
  return (w.reshape(n, s.shape[1], -1, R) * s.unsqueeze(2)).reshape(n, -1, R)

def matvec(x:Tensor, w:Tensor, split:bool) -> Tensor:
  """x (n, C) times w (n, C, R). with split, the reduction is done in chunks of 48 first so a single token has enough parallelism"""
  n, C, R = w.shape
  if not split or C % 48: return (x.unsqueeze(1) @ w).reshape(n, R)
  return (x.reshape(n, C // 48, 48, 1) * w.reshape(n, C // 48, 48, R)).sum(2).contiguous().sum(1)

# ***** model *****

class Inkling2:
  def __init__(self, cfg:dict, weights:dict[str, Tensor], max_context:int=4096, cache_dtype=dtypes.half):
    self.cfg, self.W, self.max_context, self.cache_dtype = cfg, weights, max_context, cache_dtype
    self.H, self.n_layers, self.block = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["n_layers_per_attn_res_block"]
    self.loop_start, self.loop_end, self.passes = (cfg["loop_config"][k] for k in ("core_start", "core_end", "num_passes"))
    self.local = set(cfg["local_layer_ids"])
    self.K = cfg["sconv_kernel_size"]
    self.experts = {}
    for i in range(self.n_layers):
      for name in ("w13", "w2"):
        p = f"model.llm.layers.{i}.mlp.experts.{name}_weight"
        q = {"w": weights.pop(p)}
        if (s:=weights.pop(p+"_scale", None)) is not None: q["s"] = s
        if (amax:=weights.pop(p+"_amax", None)) is not None:
          q["f"] = (amax.float() / weights.pop(p+"_outer_scale_denominator").float()).contiguous().realize()
        if "s" in q: q["w"], q["s"] = to_cols_rows(q["w"]).realize(), to_cols_rows(q["s"]).realize()
        else: q["w"] = q["w"].permute(0, 2, 1).contiguous().realize()
        self.experts[(i, name)] = q
    n_loop = self.loop_end - self.loop_start
    qs, kvs = (weights[f"model.llm.middle_cycle_cross_pass_{n}"].float().reshape(n_loop, 2, self.H) for n in ("queries", "kv_queries"))
    self.message_queries = Tensor.stack(qs[:, 0], kvs[:, 0], kvs[:, 1], qs[:, 1], dim=1)  # (n_loop, 4: q k v mlp, H)
    # sliding layers keep their keys and values in a ring big enough for the window plus a prefill chunk, full layers keep max_context
    self.chunks, self.ring = (128, 32, 1), 2048
    assert self.ring >= cfg["sliding_window_size"] + max(self.chunks) and self.ring % max(self.chunks) == 0
    self.app_layers = [*range(self.loop_start), *list(range(self.loop_start, self.loop_end)) * self.passes, *range(self.loop_end, self.n_layers)]
    self.caches = [self.new_cache(i) for i in self.app_layers]
    self.jits: dict[int, TinyJit] = {}

  def new_cache(self, i:int) -> dict[str, Tensor]:
    cfg, full = self.cfg, i not in self.local
    n_kv, hd = (cfg["num_key_value_heads"], cfg["head_dim"]) if full else (cfg["swa_num_key_value_heads"], cfg["swa_head_dim"])
    zeros = lambda *shape, dtype=dtypes.float: Tensor.zeros(*shape, dtype=dtype).contiguous().realize()
    # kpos is the position held by each key slot, empty slots are far in the future so they are never visible
    return {"ks": zeros(self.K-1, n_kv * hd), "vs": zeros(self.K-1, n_kv * hd), "attn": zeros(self.K-1, self.H), "mlp": zeros(self.K-1, self.H),
            "kv": zeros(2, self.max_context if full else self.ring, n_kv, hd, dtype=self.cache_dtype),
            "kpos": Tensor.full((self.max_context if full else self.ring,), 1 << 30, dtype=dtypes.int32).contiguous().realize()}

  # the short convolution states can't be rewound like the keys and values, so they are saved where a prompt may be continued from
  def conv_states(self) -> list[Tensor]: return [c[k] for c in self.caches for k in ("ks", "vs", "attn", "mlp")]
  def snapshot(self) -> list[Tensor]: return [t.clone().realize() for t in self.conv_states()]
  def restore(self, snap:list[Tensor]|None=None):
    """restore a snapshot, or reset to the start of a sequence. keys and values past the restored position are masked or overwritten"""
    states = self.conv_states()
    Tensor.realize(*[t.assign(s if snap is not None else t.zeros_like()) for t, s in zip(states, snap or states)])

  def w(self, name:str) -> Tensor: return self.W["model.llm." + name]

  def attn_res(self, blocks:list[Tensor], partial:Tensor, query:Tensor) -> Tensor:
    """learned softmax mix of the normed block states and the current partial block, one output per query: (T, n_queries, H)"""
    states = rms(Tensor.stack(*blocks, partial, dim=1).float())  # (T, n_blocks+1, H)
    attn = (query.float().reshape(-1, self.H) @ states.transpose(1, 2) * self.H**-0.5).softmax(-1)  # (T, n_queries, n_blocks+1)
    return attn @ states

  def message_gate(self, name:str, h:Tensor, msg:Tensor) -> Tensor:
    g = rms(h).cat(rms(msg), dim=-1) @ self.w(f"{name}.down_weight").T
    return (1 - (g @ self.w(f"{name}.forget_weight").T).tanh()) * h + (g @ self.w(f"{name}.gain_weight").T).tanh() * msg

  def sconv(self, x:Tensor, weight:Tensor, cache:dict, key:str) -> Tensor:
    """causal depthwise conv over the last K tokens plus a residual. the cache holds the K-1 previous inputs"""
    T, K = x.shape[0], self.K
    ctx = cache[key].cat(x, dim=0)
    w = weight.reshape(-1, K).float()  # w[:, K-1] weights the current token
    out = sum(ctx[j:j+T] * w[:, j] for j in range(K))
    cache[key].assign(ctx[-(K-1):])
    return (out + x) / math.sqrt(K + 1)

  def dynamic_sconv(self, x:Tensor, dyn_in:Tensor, p:str, n_heads:int, cache:dict, key:str) -> Tensor:
    """sconv whose taps are static per channel weights plus per token weights shared by each head. tap l weights the token l steps back"""
    T, C, K = *x.shape, self.K
    dyn = (dyn_in @ self.w(p+".weight_proj").T).reshape(T, n_heads, 1, K)
    taps = self.w(p+".static_w").float().T.reshape(n_heads, C // n_heads, K) + dyn  # (T, n_heads, head_dim, K)
    ctx = cache[key].cat(x, dim=0)
    out = sum(ctx[K-1-l:K-1-l+T].reshape(T, n_heads, -1) * taps[..., l] for l in range(K)).reshape(T, C)
    cache[key].assign(ctx[-(K-1):])
    return (out + x) / math.sqrt(2 * K + 1)

  def attention(self, i:int, xq:Tensor, xk:Tensor, xv:Tensor, cache:dict, start_pos, ring_pos, pos:Tensor) -> Tensor:
    cfg, p, full, T = self.cfg, f"layers.{i}.attn", i not in self.local, xq.shape[0]
    nh, nkv, hd = (cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]) if full else \
                  (cfg["swa_num_attention_heads"], cfg["swa_num_key_value_heads"], cfg["swa_head_dim"])
    # full attention layers scale queries and relative states by 1 + alpha * log(max((pos+1) / n_floor, 1))
    tau = (1 + cfg["log_scaling_alpha"] * ((pos + 1).float() / cfg["log_scaling_n_floor"]).maximum(1.0).log()).reshape(T, 1, 1) if full else 1.0

    # contiguous marks the intermediates that would otherwise be recomputed inside the attention reductions
    xq, xk, xv = xq.contiguous(), xk.contiguous(), xv.contiguous()
    q = l2clip((xq @ self.w(p+".wq_du.weight").T).reshape(T, nh, hd) * self.w(p+".q_l2_clip.scale").float(), cfg["qk_l2_clip_threshold"]) * tau
    k, v = xk @ self.w(p+".wk_dv.weight").T, xv @ self.w(p+".wv_dv.weight").T
    if full: k, v = self.dynamic_sconv(k, xk, p+".k_sconv", nkv, cache, "ks"), self.dynamic_sconv(v, xv, p+".v_sconv", nkv, cache, "vs")
    else: k, v = self.sconv(k, self.w(p+".k_sconv.weight"), cache, "ks"), self.sconv(v, self.w(p+".v_sconv.weight"), cache, "vs")
    k, v = l2clip(k.reshape(T, nkv, hd), cfg["qk_l2_clip_threshold"]), rms(v.reshape(T, nkv, hd), cfg["rms_norm_eps"])

    slot = start_pos if full else ring_pos  # chunks start at a multiple of their size, so they never wrap around the ring
    cache["kv"][:, slot:slot+T].assign(Tensor.stack(k, v).cast(cache["kv"].dtype))
    cache["kpos"][slot:slot+T].assign(pos)
    keys, values = (cache["kv"][j].float().permute(1, 0, 2) for j in range(2))  # (nkv, S, hd)
    S = keys.shape[1]

    # hidden state conditioned relative position bias over backward distances [0, rel_extent)
    rel_extent = cfg["rel_extent"] if full else cfg["sliding_window_size"]
    rel = l2clip((xq @ self.w(p+".wr_du.weight").T).reshape(T, nh, -1) * self.w(p+".rel_l2_clip.scale").float(), cfg["rel_l2_clip_threshold"]) * tau
    rel_logits = (rel @ self.w(p+".rel_logits_proj.proj").float()).permute(1, 0, 2).contiguous()  # (nh, T, rel_extent)
    dist = pos.reshape(T, 1) - cache["kpos"].reshape(1, S)
    bias = ((dist >= 0) & (dist < rel_extent)).where(rel_logits.gather(-1, dist.clip(0, rel_extent-1).expand(nh, T, S)), 0)
    visible = (dist >= 0) if full else ((dist >= 0) & (dist < cfg["sliding_window_size"]))

    scores = (q.contiguous().reshape(T, nkv, nh // nkv, hd).permute(1, 2, 0, 3) @ keys.unsqueeze(1).transpose(-1, -2)).reshape(nh, T, S) * hd**-0.5
    scores = visible.where(scores + bias, -math.inf).contiguous()
    # per token attention sink: one extra logit with a zero value, it only adds to the softmax denominator
    sink = (xq @ self.w(p+".attn_sink_proj.weight").T).T.reshape(nh, T, 1).contiguous()
    m = scores.max(-1, keepdim=True).maximum(sink)
    scores = (scores - m).exp()
    scores = (scores / (scores.sum(-1, keepdim=True) + (sink - m).exp())).contiguous()
    out = (scores.reshape(nkv, nh // nkv, T, S) @ values.unsqueeze(1)).reshape(nh, T, hd).permute(1, 0, 2).reshape(T, nh * hd)
    out = out.contiguous() * (xq @ self.w(p+".gate_attn.weight").T).sigmoid().contiguous()
    return out.contiguous() @ self.w(p+".wo_ud.weight").T

  def moe(self, i:int, x:Tensor) -> Tensor:
    cfg, T, k = self.cfg, x.shape[0], self.cfg["num_experts_per_tok"]
    scores = (x @ self.w(f"layers.{i}.mlp.gate.weight").T.float()).sigmoid()  # (T, n_experts)
    sel = (scores + self.w(f"layers.{i}.mlp.gate.bias")).topk(k)[1]
    probs = scores.gather(-1, sel)
    probs = probs / (probs.sum(-1, keepdim=True) + 1e-20) * cfg["route_scale"]
    sel, split = sel.flatten(), T == 1
    xs = x.unsqueeze(1).expand(T, k, self.H).reshape(T * k, self.H)
    gate_up = matvec(xs, dequant(self.experts[(i, "w13")], sel), split).reshape(T * k, -1, 2)  # rows interleave gate, up
    gate, up = gate_up[..., 0], gate_up[..., 1]
    act = softcap(gate, cfg["mlp_gate_softcap"]) * gate.sigmoid() * softcap(up, cfg["mlp_up_softcap"])
    out = matvec(act, dequant(self.experts[(i, "w2")], sel), split).reshape(T, k, self.H)
    return (out * probs.unsqueeze(-1)).sum(1)

  def layer(self, i:int, h:Tensor, blocks:list[Tensor], msgs:Tensor|None, app:int, start_pos, ring_pos, pos:Tensor) -> Tensor:
    """decoder layer i as the app-th layer application. msgs holds the (q, k, v, mlp) messages from the previous pass of the loop"""
    full, cache, eps, L = i not in self.local, self.caches[app], self.cfg["rms_norm_eps"], f"layers.{i}"
    norm = lambda x, name: rms(x, eps) * self.w(f"{L}.{name}.weight").float()
    gates = [f"{L}.controllers.{n}" for n in ("attn", "k", "v", "mlp")]
    if full:
      xq = xk = xv = h
      if i > 0: xq, xk, xv = (x[:, 0] for x in self.attn_res(blocks, h, self.w(f"attn_res.attn_queries.{i-1}")).split(1, dim=1))
      if msgs is not None: xq, xk, xv = (self.message_gate(g, x, msgs[:, j]) for j, (g, x) in enumerate(zip(gates, (xq, xk, xv))))
      xq, xk, xv = norm(xq, "q_input_norm"), norm(xk, "k_input_norm"), norm(xv, "v_input_norm")
    else:
      x = self.attn_res(blocks, h, self.w(f"attn_res.attn_queries.{i-1}"))[:, 0] if i > 0 else h
      if msgs is None: xq = xk = xv = norm(x, "attn_norm")
      else: xq, xk, xv = (norm(self.message_gate(g, x, msgs[:, j]), "attn_norm") for j, g in enumerate(gates[:3]))
    a = self.sconv(self.attention(i, xq, xk, xv, cache, start_pos, ring_pos, pos), self.w(f"{L}.attn_sconv.weight"), cache, "attn")
    # at a block boundary the layer input becomes a block state and the residual stream restarts
    if i % self.block == 0: blocks, residual = blocks + [h], a
    else: residual = h + a
    x = self.attn_res(blocks, residual, self.w(f"attn_res.mlp_queries.{i}"))[:, 0]
    if msgs is not None: x = self.message_gate(gates[3], x, msgs[:, 3])
    return residual + self.sconv(self.moe(i, norm(x, "mlp_norm")), self.w(f"{L}.mlp_sconv.weight"), cache, "mlp")

  def forward(self, tokens:Tensor, start_pos, ring_pos, pos:Tensor) -> Tensor:
    """tokens (T,) at positions pos (T,) starting at start_pos (ring_pos in the ring). returns the softcapped logits of the last token"""
    h = rms(self.w("embed.weight")[tokens].float(), self.cfg["rms_norm_eps"]) * self.w("embed_norm.weight").float()
    blocks: list[Tensor] = []
    def run(i, h, msgs, app):
      out = self.layer(i, h, blocks, msgs, app, start_pos, ring_pos, pos)
      if i % self.block == 0: blocks.append(h)
      return out
    ls, le = self.loop_start, self.loop_end
    for i in range(ls): h = run(i, h, None, i)
    pre_h, n_pre = h, len(blocks)
    for p in range(self.passes):
      msgs_from = None
      if p > 0:
        # later passes restart from the pre loop states and read messages from the previous pass (skipping its first block state)
        msgs_from, h = (blocks[n_pre+1:], h), pre_h
        del blocks[n_pre:]
      for j, i in enumerate(range(ls, le)):
        msgs = None if msgs_from is None else self.attn_res(msgs_from[0], msgs_from[1], self.message_queries[j])
        h = run(i, h, msgs, ls + p * (le - ls) + j)
    for i in range(le, self.n_layers): h = run(i, h, None, i + (self.passes - 1) * (le - ls))
    h = rms(self.attn_res(blocks, h, self.w("attn_res.output_query"))[:, 0], self.cfg["rms_norm_eps"]) * self.w("norm.weight").float()
    logits = h[-1:] @ self.w("unembed.weight")[:self.cfg["unpadded_vocab_size"]].T
    logits = softcap(logits, self.cfg["final_logit_softcapping"])[0]
    Tensor.realize(logits, *[t for c in self.caches for t in c.values()])
    return logits

  def __call__(self, tokens:list[int], start_pos:int) -> Tensor:
    """runs tokens at start_pos (a sequence starting at 0 needs restore() first) and returns the logits after the last one"""
    for logits in self.run_chunks(tokens, start_pos): pass
    return logits

  def run_chunks(self, tokens:list[int], start_pos:int) -> Iterator[Tensor]:
    """like __call__, yielding the logits after each chunk so a caller can do something between them"""
    assert 0 < len(tokens) and start_pos + len(tokens) <= self.max_context, f"{start_pos}+{len(tokens)} doesn't fit in {self.max_context}"
    # fixed size chunks, each starting at a multiple of its size, so every shape is compiled once and jitted
    i = 0
    while i < len(tokens):
      p = start_pos + i
      T = next(t for t in self.chunks if p % t == 0 and len(tokens) - i >= t)
      if T not in self.jits: self.jits[T] = TinyJit(self.forward)
      logits = self.jits[T](Tensor(tokens[i:i+T], dtype=dtypes.int32).realize(), Variable("start_pos", 0, self.max_context-1).bind(p),
                            Variable("ring_pos", 0, self.ring-1).bind(p % self.ring), Tensor(list(range(p, p+T)), dtype=dtypes.int32).realize())
      # without a sync, a long prompt queues more GPU work than the device's wait timeout and the next sync reports a hang
      Device[logits.device].synchronize()
      i += T
      yield logits

def load(model_dir:Path, max_context:int, device:str=Device.DEFAULT) -> Inkling2:
  cfg = json.loads((model_dir / "config.json").read_text())
  cfg = cfg.get("text_config", cfg)
  weights = {}
  for f in sorted(model_dir.glob("*.safetensors")):
    if f.name == "mtp.safetensors": continue
    for k, v in safe_load(f).items():
      if k.startswith("model.llm."): weights[k] = v.to(device)
  Tensor.realize(*weights.values())
  return Inkling2(cfg, weights, max_context)

def chat_prompt(user:str, effort:float=0.0) -> str:
  # chat_template.jinja: a thinking effort system message comes before the first non system message
  return f"<|message_system|><|content_text|>Thinking effort level: {effort if effort else 0}<|end_message|>" \
         f"<|message_user|><|content_text|>{user}<|end_message|>"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run Inkling 2 Mini in tinygrad")
  parser.add_argument("--model", type=Path, required=True, help="directory with config.json, tokenizer.json and the safetensors")
  parser.add_argument("--prompt", default="Explain in one sentence why the sky is usually blue.")
  parser.add_argument("--raw", action="store_true", help="don't wrap the prompt in the chat template")
  parser.add_argument("--effort", type=float, default=0.0, help="thinking effort in [0, 0.99]")
  parser.add_argument("--count", type=int, default=100)
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--max_context", type=int, default=4096)
  args = parser.parse_args()

  from tokenizers import Tokenizer
  tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
  eos = json.loads((args.model / "config.json").read_text())["eos_token_id"]

  st = time.perf_counter()
  model = load(args.model, args.max_context)
  print(f"loaded weights in {time.perf_counter()-st:.2f} s, {GlobalCounters.mem_used/1e9:.2f} GB")

  toks = tokenizer.encode(args.prompt if args.raw else chat_prompt(args.prompt, args.effort), add_special_tokens=False).ids
  start_pos, out = 0, []
  while len(out) < args.count:
    GlobalCounters.reset()
    st = time.perf_counter()
    logits = model(toks[start_pos:], start_pos)
    tok = int((logits.argmax() if args.temperature == 0 else (logits / args.temperature).softmax().multinomial()).item())
    et = time.perf_counter()
    if start_pos == 0: print(f"prefill {len(toks)} tokens in {et-st:.2f} s")
    else: print(f"{(et-st)*1e3:7.2f} ms, {GlobalCounters.global_mem/(et-st)/1e9:6.1f} GB/s", end="\r")
    start_pos = len(toks)
    toks.append(tok)
    out.append(tok)
    if tok == eos: break
  print("\n" + tokenizer.decode(out, skip_special_tokens=False))
