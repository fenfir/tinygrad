#!/usr/bin/env python3
# Inkling 2 Mini (https://huggingface.co/thinkingmachines/Inkling-2-Mini), text decoder only
# port of transformers/models/inkling2_loop/modeling_inkling2_loop.py that runs the original NVFP4/MXFP8 checkpoint without unpacking it
# 24 layers in 4 blocks of 6, blocks 2 and 3 (layers 6-17) run 3 times, attention residuals across blocks, 128 experts with 8 active
import json, math, time, argparse
from pathlib import Path
from tinygrad import Tensor, TinyJit, Variable, dtypes, Device, GlobalCounters
from tinygrad.nn.state import safe_load

def rms(x:Tensor, eps:float=1e-6) -> Tensor: return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt()
def l2clip(x:Tensor, thr:float) -> Tensor: return x * (thr / (x.square().sum(-1, keepdim=True).sqrt() + 1e-6)).minimum(1.0)
def softcap(x:Tensor, cap:float) -> Tensor: return (x / cap).tanh() * cap

# ***** quantized experts *****

def e2m1(c:Tensor) -> Tensor:
  # 4 bit float: sign, 2 exponent bits, 1 mantissa bit. 4*|value| is m*2 for e == 0 and (2+m)<<(e-1) otherwise
  e, m = (c >> 1) & 3, c & 1
  mag = (e == 0).where(m * 2, (m + 2) * (e == 3).where(8, (e == 2).where(4, 2)))
  return (c >= 8).where(-0.25, 0.25) * mag.cast(dtypes.float)

def dequant(q:dict[str, Tensor], sel:Tensor) -> Tensor:
  """returns the float (len(sel), rows, cols) weights of experts sel"""
  w = q["w"][sel]
  if "f" in q:  # NVFP4: two e2m1 per byte (even column in the low nibble), an fp8e4m3 scale per 16 values, a scale per expert
    b = w.cast(dtypes.int32)
    w = Tensor.stack(e2m1(b & 0xF), e2m1(b >> 4), dim=-1).reshape(*w.shape[:-1], -1)
    s = q["s"][sel].cast(dtypes.float) * q["f"][sel].reshape(-1, 1, 1)
  elif "s" in q:  # MXFP8: fp8e4m3 values, an e8m0 scale (2**(s-127)) per 32 values
    s = (q["s"][sel].cast(dtypes.uint32) << 23).bitcast(dtypes.float)
  else: return w.cast(dtypes.float)  # unquantized
  return (w.cast(dtypes.float).reshape(*s.shape, -1) * s.unsqueeze(-1)).reshape(w.shape)

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
          q["f"] = (amax.float() / weights.pop(p+"_outer_scale_denominator").float()).contiguous()
        self.experts[(i, name)] = q
    n_loop = self.loop_end - self.loop_start
    qs, kvs = (weights[f"model.llm.middle_cycle_cross_pass_{n}"].float().reshape(n_loop, 2, self.H) for n in ("queries", "kv_queries"))
    self.message_queries = Tensor.stack(qs[:, 0], kvs[:, 0], kvs[:, 1], qs[:, 1], dim=1)  # (n_loop, 4: q k v mlp, H)
    self.caches: dict[int, dict[str, Tensor]] = {}
    self.decode_jit = TinyJit(self.forward)

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

  def attention(self, i:int, xq:Tensor, xk:Tensor, xv:Tensor, cache:dict, start_pos, pos:Tensor) -> Tensor:
    cfg, p, full, T = self.cfg, f"layers.{i}.attn", i not in self.local, xq.shape[0]
    nh, nkv, hd = (cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]) if full else \
                  (cfg["swa_num_attention_heads"], cfg["swa_num_key_value_heads"], cfg["swa_head_dim"])
    # full attention layers scale queries and relative states by 1 + alpha * log(max((pos+1) / n_floor, 1))
    tau = (1 + cfg["log_scaling_alpha"] * ((pos + 1).float() / cfg["log_scaling_n_floor"]).maximum(1.0).log()).reshape(T, 1, 1) if full else 1.0

    q = l2clip((xq @ self.w(p+".wq_du.weight").T).reshape(T, nh, hd) * self.w(p+".q_l2_clip.scale").float(), cfg["qk_l2_clip_threshold"]) * tau
    k, v = xk @ self.w(p+".wk_dv.weight").T, xv @ self.w(p+".wv_dv.weight").T
    if full: k, v = self.dynamic_sconv(k, xk, p+".k_sconv", nkv, cache, "ks"), self.dynamic_sconv(v, xv, p+".v_sconv", nkv, cache, "vs")
    else: k, v = self.sconv(k, self.w(p+".k_sconv.weight"), cache, "ks"), self.sconv(v, self.w(p+".v_sconv.weight"), cache, "vs")
    k, v = l2clip(k.reshape(T, nkv, hd), cfg["qk_l2_clip_threshold"]), rms(v.reshape(T, nkv, hd), cfg["rms_norm_eps"])

    cache["kv"][:, start_pos:start_pos+T].assign(Tensor.stack(k, v).cast(cache["kv"].dtype))
    S = start_pos + T if isinstance(start_pos, int) else self.max_context
    keys, values = (cache["kv"][j, :S].float().permute(1, 0, 2) for j in range(2))  # (nkv, S, hd)

    # hidden state conditioned relative position bias over backward distances [0, rel_extent)
    rel_extent = cfg["rel_extent"] if full else cfg["sliding_window_size"]
    rel = l2clip((xq @ self.w(p+".wr_du.weight").T).reshape(T, nh, -1) * self.w(p+".rel_l2_clip.scale").float(), cfg["rel_l2_clip_threshold"]) * tau
    rel_logits = (rel @ self.w(p+".rel_logits_proj.proj").float()).permute(1, 0, 2)  # (nh, T, rel_extent)
    dist = pos.reshape(T, 1) - Tensor.arange(S).reshape(1, S)
    bias = ((dist >= 0) & (dist < rel_extent)).where(rel_logits.gather(-1, dist.clip(0, rel_extent-1).expand(nh, T, S)), 0)
    visible = (dist >= 0) if full else ((dist >= 0) & (dist < cfg["sliding_window_size"]))

    scores = (q.reshape(T, nkv, nh // nkv, hd).permute(1, 2, 0, 3) @ keys.unsqueeze(1).transpose(-1, -2)).reshape(nh, T, S) * hd**-0.5
    scores = visible.where(scores + bias, -math.inf)
    # per token attention sink: one extra logit with a zero value
    scores = scores.cat((xq @ self.w(p+".attn_sink_proj.weight").T).T.reshape(nh, T, 1), dim=-1).softmax(-1)[..., :S]
    out = (scores.reshape(nkv, nh // nkv, T, S) @ values.unsqueeze(1)).reshape(nh, T, hd).permute(1, 0, 2).reshape(T, nh * hd)
    out = out * (xq @ self.w(p+".gate_attn.weight").T).sigmoid()
    return out @ self.w(p+".wo_ud.weight").T

  def moe(self, i:int, x:Tensor) -> Tensor:
    cfg, T, k = self.cfg, x.shape[0], self.cfg["num_experts_per_tok"]
    scores = (x @ self.w(f"layers.{i}.mlp.gate.weight").T.float()).sigmoid()  # (T, n_experts)
    sel = (scores + self.w(f"layers.{i}.mlp.gate.bias")).topk(k)[1]
    probs = scores.gather(-1, sel)
    probs = probs / (probs.sum(-1, keepdim=True) + 1e-20) * cfg["route_scale"]
    sel = sel.flatten()
    xs = x.unsqueeze(1).expand(T, k, self.H).reshape(T * k, 1, self.H)
    gate_up = (xs @ dequant(self.experts[(i, "w13")], sel).transpose(-1, -2)).reshape(T * k, -1, 2)  # rows interleave gate, up
    gate, up = gate_up[..., 0], gate_up[..., 1]
    act = softcap(gate, cfg["mlp_gate_softcap"]) * gate.sigmoid() * softcap(up, cfg["mlp_up_softcap"])
    out = (act.unsqueeze(1) @ dequant(self.experts[(i, "w2")], sel).transpose(-1, -2)).reshape(T, k, self.H)
    return (out * probs.unsqueeze(-1)).sum(1)

  def get_cache(self, i:int, app:int) -> dict:
    if app not in self.caches:
      cfg, full = self.cfg, i not in self.local
      kv_dim = cfg["num_key_value_heads" if full else "swa_num_key_value_heads"] * cfg["head_dim" if full else "swa_head_dim"]
      self.caches[app] = {"ks": Tensor.zeros(self.K-1, kv_dim).contiguous().realize(), "vs": Tensor.zeros(self.K-1, kv_dim).contiguous().realize(),
                          "attn": Tensor.zeros(self.K-1, self.H).contiguous().realize(), "mlp": Tensor.zeros(self.K-1, self.H).contiguous().realize(),
                          "kv": Tensor.zeros(2, self.max_context, kv_dim // cfg["head_dim"], cfg["head_dim"], dtype=self.cache_dtype).contiguous().realize()}
    return self.caches[app]

  def layer(self, i:int, h:Tensor, blocks:list[Tensor], msgs:Tensor|None, app:int, start_pos, pos:Tensor) -> Tensor:
    """decoder layer i as the app-th layer application. msgs holds the (q, k, v, mlp) messages from the previous pass of the loop"""
    full, cache, eps, L = i not in self.local, self.get_cache(i, app), self.cfg["rms_norm_eps"], f"layers.{i}"
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
    a = self.sconv(self.attention(i, xq, xk, xv, cache, start_pos, pos), self.w(f"{L}.attn_sconv.weight"), cache, "attn")
    # at a block boundary the layer input becomes a block state and the residual stream restarts
    if i % self.block == 0: blocks, residual = blocks + [h], a
    else: residual = h + a
    x = self.attn_res(blocks, residual, self.w(f"attn_res.mlp_queries.{i}"))[:, 0]
    if msgs is not None: x = self.message_gate(gates[3], x, msgs[:, 3])
    return residual + self.sconv(self.moe(i, norm(x, "mlp_norm")), self.w(f"{L}.mlp_sconv.weight"), cache, "mlp")

  def forward(self, tokens:Tensor, start_pos, pos:Tensor) -> Tensor:
    """tokens (T,) at positions pos (T,) starting at start_pos. returns the softcapped logits of the last token"""
    h = rms(self.w("embed.weight")[tokens].float(), self.cfg["rms_norm_eps"]) * self.w("embed_norm.weight").float()
    blocks: list[Tensor] = []
    def run(i, h, msgs, app):
      out = self.layer(i, h, blocks, msgs, app, start_pos, pos)
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
    Tensor.realize(logits, *[t for c in self.caches.values() for t in c.values()])
    return logits

  def __call__(self, tokens:list[int], start_pos:int) -> Tensor:
    pos = Tensor(list(range(start_pos, start_pos + len(tokens))), dtype=dtypes.int32)
    if len(tokens) == 1 and start_pos > 0:
      return self.decode_jit(Tensor(tokens, dtype=dtypes.int32).contiguous().realize(), Variable("start_pos", 1, self.max_context-1).bind(start_pos), pos.realize())
    return self.forward(Tensor(tokens, dtype=dtypes.int32), start_pos, pos)

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
