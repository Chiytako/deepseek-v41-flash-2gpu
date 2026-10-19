"""DeepSeek-V4.1-Flash-1017 の層ストリーミング参照ハーネス (実験 X6)。

公式 inference/model.py をそのまま使い、GPU 1 枚に載らない部分だけ差し替える:
  - routed experts (289GB): 生成しない。MoE.forward で、その層で使われた専門家だけを
    safetensors から pinned バッファ経由で GPU に読み込む (LRU キャッシュ付き)。
  - Engram 表 (2 x 98GB): GPU に置かない。CPU 側で safetensors を直接 pread して行を集める
    (= Engram の NVMe 配置そのもの)。
  - sparse_attn: tilelang 版は SM120 の共有メモリ上限 (99KB) を超えるので torch 版に置換。
重みは元の FP8/FP4 のまま (無量子化) なので、出力は「正解」の基準として使える。

使い方 (LIL イメージ内で):
  python3 stream_ref.py --prompt "..." --max-new 32            # 貪欲生成
  python3 stream_ref.py --prompts-file p.jsonl --dump out.pt    # 教師 logits / ルーティング統計
"""
import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

REF = os.environ.get("DSV41_REF", "/ref")
CKPT = os.environ.get("DSV41_CKPT", "/model")
sys.path.insert(0, REF)
sys.path.insert(0, os.path.join(CKPT, "encoding"))

import kernel as K  # noqa: E402
import model as M  # noqa: E402

SD = torch.float8_e8m0fnu
FP4X2 = torch.float4_e2m1fn_x2
_DT = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16,
       "F32": torch.float32, "I8": torch.int8, "I64": torch.int64, "F16": torch.float16,
       "I16": torch.int16, "U16": torch.uint16, "I32": torch.int32, "U32": torch.uint32, "U8": torch.uint8}


# --------------------------------------------------------------------------- tensor index
class TensorIndex:
    """safetensors のヘッダから name -> (fd, 絶対オフセット, nbytes, dtype, shape)。"""

    def __init__(self, ckpt):
        idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
        self.meta = {}
        self.fds = {}
        for fname in sorted(set(idx.values())):
            path = os.path.join(ckpt, fname)
            with open(path, "rb") as f:
                hl = int.from_bytes(f.read(8), "little")
                hdr = json.loads(f.read(hl))
            fd = os.open(path, os.O_RDONLY)
            self.fds[fname] = fd
            base = 8 + hl
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.meta[k] = (fd, base + a, b - a, _DT[v["dtype"]], tuple(v["shape"]))

    def read_into(self, name, buf: torch.Tensor):
        fd, off, n, _, _ = self.meta[name]
        mv = memoryview(buf.view(torch.uint8).numpy()).cast("B")
        got = 0
        while got < n:
            r = os.preadv(fd, [mv[got:n]], off + got)
            if r <= 0:
                raise IOError(name)
            got += r

    def load(self, name, device="cpu"):
        _, _, n, dt, shape = self.meta[name]
        t = torch.empty(n, dtype=torch.uint8)
        self.read_into(name, t)
        t = t.view(dt).view(shape)
        return t.to(device) if device != "cpu" else t


# --------------------------------------------------------------------------- sparse attention (torch)
def sparse_attn_torch(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=512):
    """kernel.sparse_attn と同じ意味: idx=-1 は無視、attn_sink は分母にだけ入る。"""
    b, s, h, d = q.shape
    out = torch.empty_like(q)
    sink = attn_sink.float().view(1, 1, h, 1)
    for bi in range(b):
        for s0 in range(0, s, chunk):
            s1 = min(s, s0 + chunk)
            idx = topk_idxs[bi, s0:s1].long()  # [sc, k]
            valid = idx >= 0
            g = kv[bi][idx.clamp_min(0)]  # [sc, k, d]
            sc = torch.einsum("shd,skd->shk", q[bi, s0:s1].float(), g.float()) * softmax_scale
            sc = sc.masked_fill(~valid.unsqueeze(1), float("-inf"))
            m = torch.maximum(sc.amax(-1, keepdim=True), sink.squeeze(0).expand(s1 - s0, h, 1))
            m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
            p = torch.exp(sc - m)
            den = p.sum(-1, keepdim=True) + torch.exp(sink.squeeze(0) - m)
            o = torch.einsum("shk,skd->shd", p.to(torch.bfloat16).float(), g.float()) / den
            out[bi, s0:s1] = o.to(q.dtype)
    return out


# --------------------------------------------------------------------------- engram on host / NVMe
class HostEngramEmbedding(torch.nn.Module):
    """ParallelEngramEmbedding の代替。表は GPU に置かず、行を pread で集める。"""

    def __init__(self, num_embeddings, dim):
        super().__init__()
        self.num_embeddings, self.dim, self.block_size = num_embeddings, dim, 32
        self.tindex = None
        self.wname = self.sname = None
        self.stats = {"lookups": 0, "unique": 0, "seconds": 0.0}
        self.pool = ThreadPoolExecutor(32)

    def bind(self, tindex, layer_id):
        self.tindex = tindex
        self.wname = f"layers.{layer_id}.engram.embed.weight"
        self.sname = f"layers.{layer_id}.engram.embed.scale"
        assert tindex.meta[self.wname][4] == (self.num_embeddings, self.dim)

    def _gather(self, name, rows, row_bytes):
        fd, off, _, _, _ = self.tindex.meta[name]
        out = np.empty((len(rows), row_bytes), dtype=np.uint8)

        def work(lo, hi):
            for i in range(lo, hi):
                r = int(rows[i])
                out[i] = np.frombuffer(os.pread(fd, row_bytes, off + r * row_bytes), dtype=np.uint8)

        n = len(rows)
        step = max(256, (n + 63) // 64)
        list(self.pool.map(lambda a: work(a, min(n, a + step)), range(0, n, step)))
        return out

    def forward(self, indices):
        t0 = time.time()
        flat = indices.reshape(-1).cpu().numpy()
        uniq, inv = np.unique(flat, return_inverse=True)
        w = self._gather(self.wname, uniq, self.dim)
        s = self._gather(self.sname, uniq, self.dim // self.block_size)
        dev = indices.device
        w = torch.from_numpy(w).to(dev).view(torch.float8_e4m3fn)
        s = torch.from_numpy(s).to(dev).view(SD)
        vals = w.float().unflatten(-1, (-1, self.block_size)) * s.float().unsqueeze(-1)
        vals = vals.flatten(-2).to(torch.bfloat16)
        inv_t = torch.from_numpy(inv.reshape(-1)).to(dev)
        self.stats["lookups"] += flat.size
        self.stats["unique"] += uniq.size
        self.stats["seconds"] += time.time() - t0
        return vals[inv_t].view(*indices.shape, self.dim)


# --------------------------------------------------------------------------- streamed experts
class ExpertStore:
    """routed experts を GPU の LRU キャッシュに読み込む。キー (layer_id, expert)。"""

    PARTS = ("w1", "w3", "w2")

    def __init__(self, tindex, n_layers, budget_gb, device, exl3_index=None):
        self.t = tindex
        self.x = exl3_index  # EXL3 パックの TensorIndex (None なら元の FP4)
        self.n_layers = n_layers
        self.dev = device
        self.cache = OrderedDict()
        self.expert_bytes = 0
        self.budget = int(budget_gb * 2**30)
        self.used = 0
        self.pool = ThreadPoolExecutor(16)
        self.stats = {"loaded": 0, "hits": 0, "bytes": 0, "seconds": 0.0}
        self.counts = {}  # layer_id -> np.int64[n_experts] (トークン単位の選択回数)

    def name(self, layer_id, e, part, kind):
        if layer_id < self.n_layers:
            return f"layers.{layer_id}.ffn.experts.{e}.{part}.{kind}"
        return f"mtp.{layer_id - self.n_layers}.ffn.experts.{e}.{part}.{kind}"

    def _is_exl3(self, layer_id, e):
        return self.x is not None and f"layers.{layer_id}.ffn.experts.{e}.w1.trellis" in self.x.meta

    def _read_expert(self, layer_id, e):
        if self._is_exl3(layer_id, e):
            out = []
            for part in self.PARTS:
                d = {}
                for kind in ("trellis", "suh", "svh", "mcg"):
                    nm = f"layers.{layer_id}.ffn.experts.{e}.{part}.{kind}"
                    if nm in self.x.meta:
                        d[kind] = self.x.load(nm)
                out.append(d)
            return ("exl3", out)
        out = []
        for part in self.PARTS:
            for kind in ("weight", "scale"):
                nm = self.name(layer_id, e, part, kind)
                _, _, n, dt, shape = self.t.meta[nm]
                buf = torch.empty(n, dtype=torch.uint8)
                self.t.read_into(nm, buf)
                out.append((buf, dt, shape))
        return out

    def get(self, layer_id, experts):
        """experts: list[int]。返り値 {e: (w1,s1,w3,s3,w2,s2)}。"""
        t0 = time.time()
        res, miss = {}, []
        for e in experts:
            k = (layer_id, e)
            if k in self.cache:
                self.cache.move_to_end(k)
                res[e] = self.cache[k]
                self.stats["hits"] += 1
            else:
                miss.append(e)
        if miss:
            for e, parts in zip(miss, self.pool.map(lambda e: self._read_expert(layer_id, e), miss)):
                gpu = []
                nbytes = 0
                if isinstance(parts, tuple) and parts[0] == "exl3":
                    from exllamav3.modules.quant.exl3 import LinearEXL3
                    ws = []
                    for d in parts[1]:
                        g = {k: v.to(self.dev) for k, v in d.items()}
                        tr = g["trellis"]
                        in_f, out_f = tr.shape[0] * 16, tr.shape[1] * 16
                        lin = LinearEXL3(None, in_f, out_f, suh=g["suh"], svh=g["svh"], trellis=tr,
                                         mcg=g.get("mcg"))
                        ws.append(lin.get_weight_tensor().to(torch.bfloat16).contiguous())  # [in, out]
                        nbytes += ws[-1].numel() * 2
                        del lin
                    tup = ("bf16", ws[0], ws[1], ws[2])  # w1, w3, w2 (いずれも [in, out])
                    self.expert_bytes = nbytes
                    self.cache[(layer_id, e)] = tup
                    self.used += nbytes
                    res[e] = tup
                    self.stats["loaded"] += 1
                    self.stats["bytes"] += nbytes
                    continue
                for buf, dt, shape in parts:
                    g = buf.to(self.dev)
                    nbytes += buf.numel()
                    if dt == torch.int8:
                        g = g.view(FP4X2).view(shape[0], shape[1])
                    else:
                        g = g.view(dt).view(shape)
                    gpu.append(g)
                self.expert_bytes = nbytes
                tup = tuple(gpu)
                self.cache[(layer_id, e)] = tup
                self.used += nbytes
                res[e] = tup
                self.stats["loaded"] += 1
                self.stats["bytes"] += nbytes
            torch.cuda.current_stream().synchronize()
            while self.used > self.budget and len(self.cache) > len(experts):
                k, v = self.cache.popitem(last=False)
                if k[0] == layer_id and k[1] in res:
                    self.cache[k] = v
                    self.cache.move_to_end(k, last=False)
                    break
                self.used -= self.expert_bytes
        self.stats["seconds"] += time.time() - t0
        return res


def fp4_linear(x, w, s):
    a, a_s = K.act_quant(x, 32, "ue8m0", SD)
    return K.fp4_gemm(a, a_s, w, s, SD, act_block_size=32)


def expert_forward(x, W, weights, swiglu_limit):
    if W[0] == "bf16":
        _, w1, w3, w2 = W
        gate = (x @ w1).float()
        up = (x @ w3).float()
        if swiglu_limit > 0:
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
            gate = torch.clamp(gate, max=swiglu_limit)
        h = weights * (F.silu(gate) * up)
        return h.to(x.dtype) @ w2
    w1, s1, w3, s3, w2, s2 = W
    gate = fp4_linear(x, w1, s1).float()
    up = fp4_linear(x, w3, s3).float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    h = F.silu(gate) * up
    h = weights * h
    return fp4_linear(h.to(x.dtype), w2, s2)


STORE: ExpertStore | None = None


def moe_forward(self, x, image_mask=None):
    shape = x.size()
    x = x.view(-1, self.dim)
    weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
    y = torch.zeros_like(x, dtype=torch.float32)
    counts_t = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
    counts = counts_t.tolist()
    acc = STORE.counts.setdefault(self.layer_id, np.zeros(self.n_routed_experts, dtype=np.int64))
    acc += np.asarray(counts, dtype=np.int64)
    used = [i for i, c in enumerate(counts) if c]
    W = STORE.get(self.layer_id, used)
    for i in used:
        idx, top = torch.where(indices == i)
        y[idx] += expert_forward(x[idx], W[i], weights[idx, top, None], self.swiglu_limit)
    y += self.shared_experts(x)
    return y.type_as(x).view(shape)


# --------------------------------------------------------------------------- build / load
def build(args_over, expert_budget_gb, device="cuda", exl3_pack=""):
    global STORE
    from transformers import AutoTokenizer

    torch.set_default_dtype(torch.bfloat16)
    cfg = json.load(open(os.path.join(CKPT, "inference", "config.json")))
    cfg.update(args_over)
    args = M.ModelArgs(**cfg)

    # 1) routed experts を作らない
    real_expert = M.Expert

    def expert_factory(dim, inter_dim, dtype=None, swiglu_limit=0.0):
        if dtype == FP4X2:
            return None
        return real_expert(dim, inter_dim, dtype=dtype, swiglu_limit=swiglu_limit)

    M.Expert = expert_factory
    # 2) Engram 表を GPU に置かない
    M.ParallelEngramEmbedding = HostEngramEmbedding
    # 3) attention を torch 版に
    M.sparse_attn = sparse_attn_torch
    # 4) MoE.forward を差し替え (swiglu_limit を MoE に持たせる)
    M.MoE.forward = moe_forward

    tok = AutoTokenizer.from_pretrained(CKPT)
    t0 = time.time()
    with torch.device(device):
        model = M.Transformer(args, tok)
    M.Expert = real_expert
    for blk in list(model.layers) + list(model.mtp):
        blk.ffn.swiglu_limit = args.swiglu_limit
    print(f"[build] {time.time() - t0:.1f}s", flush=True)

    tindex = TensorIndex(CKPT)
    xindex = TensorIndex(exl3_pack) if exl3_pack else None
    if xindex is not None:
        n = sum(1 for k in xindex.meta if k.endswith(".w1.trellis"))
        print(f"[exl3] pack {exl3_pack}: {n} experts quantized", flush=True)
    STORE = ExpertStore(tindex, args.n_layers, expert_budget_gb, device, xindex)
    for blk in model.layers:
        if blk.engram is not None:
            blk.engram.embed.bind(tindex, blk.layer_id)

    # 5) experts / engram 表以外を読み込む
    t0 = time.time()
    sd = model.state_dict()
    loaded, missing = 0, []
    skip_unused = []
    used_hf = set()
    for key, param in sd.items():
        hf = key
        if key.startswith("mtp.") and key.split(".", 2)[-1] in ("embed.weight", "head.weight"):
            continue  # 本体と共有
        if hf not in tindex.meta:
            missing.append(key)
            continue
        used_hf.add(hf)
        src = tindex.load(hf)
        if key.endswith("wo_a.weight"):
            sc = tindex.load(hf.replace("weight", "scale"))
            used_hf.add(hf.replace("weight", "scale"))
            ob, ib = src.size(0) // sc.size(0), src.size(1) // sc.size(1)
            src = (src.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float()
                   * sc[:, None, :, None].float()).flatten(2, 3).flatten(0, 1)
        if src.dtype == torch.int8:
            src = src.view(FP4X2)
        with torch.no_grad():
            if param.dtype == src.dtype:
                param.copy_(src.view(param.shape))
            else:
                param.copy_(src.to(torch.float32).to(param.dtype).view(param.shape))
        loaded += 1
    for k in tindex.meta:
        if k in used_hf or ".ffn.experts." in k or ".engram.embed." in k:
            continue
        if k.endswith("wo_a.scale"):
            continue
        skip_unused.append(k)
    print(f"[load] {loaded} tensors in {time.time() - t0:.1f}s; missing={len(missing)} unused={len(skip_unused)}", flush=True)
    if missing:
        print("  missing:", missing[:20], flush=True)
    if skip_unused:
        print("  unused:", skip_unused[:20], flush=True)
    print(f"[mem] allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    torch.set_default_device(device)
    return model, tok, args


@torch.inference_mode()
def greedy(model, tokens, max_new, eos_id, keep_logits=False):
    """prefill 1 回 + decode を 1 トークンずつ。返り値 (生成 id, prefill logits(最終位置), 時間)。"""
    x = torch.tensor([tokens], dtype=torch.long)
    t0 = time.time()
    nxt, logits, _ = model.forward(x, 0)
    t_pre = time.time() - t0
    first_logits = logits[0].float().cpu() if keep_logits else None
    out = [int(nxt[0].argmax()) if nxt.dim() > 1 else int(nxt[0])]
    pos = len(tokens)
    t1 = time.time()
    while len(out) < max_new and out[-1] != eos_id:
        nxt, logits, _ = model.forward(torch.tensor([[out[-1]]], dtype=torch.long), pos)
        out.append(int(nxt[0]))
        pos += 1
    t_dec = time.time() - t1
    return out, first_logits, t_pre, t_dec


FULL_LOGITS = False


def _patch_head():
    orig = M.ParallelHead.forward

    def fwd(self, x, full_logits=False):
        return orig(self, x, full_logits or FULL_LOGITS)

    M.ParallelHead.forward = fwd


@torch.inference_mode()
def teacher(model, seqs, out_path, topk=32, ref=None):
    """各系列を 1 回の prefill で通し、score_from 以降の位置の top-k logprob と実トークンの logprob を保存。"""
    global FULL_LOGITS
    recs = []
    for i, sq in enumerate(seqs):
        ids, sf = sq["ids"], sq["score_from"]
        t0 = time.time()
        FULL_LOGITS = True
        _, logits, _ = model.forward(torch.tensor([ids], dtype=torch.long), 0)
        FULL_LOGITS = False
        lg = logits[0, sf - 1 : len(ids) - 1].float()
        lp = torch.log_softmax(lg, -1)
        tgt = torch.tensor(ids[sf:], device=lp.device)
        tlp = lp.gather(1, tgt[:, None])[:, 0]
        v, ix = lp.topk(topk, -1)
        if ref is not None:
            r = ref["recs"][i]
            assert r["ids"] == ids
            tv = r["top_v"].to(lp.device).float()
            ti = r["top_i"].to(lp.device).long()
            pt = tv.exp()
            mass = pt.sum(-1)
            qs = lp.gather(1, ti)
            kl = (pt * (tv - qs)).sum(-1)  # 教師 top-32 上の KL(P||Q) 近似
            top1_t = ti[:, 0]
            agree = (ix_s := lp.argmax(-1)) == top1_t
            print(f"[student] {i}: KL {kl.mean().item():.4f} (p50 {kl.median().item():.4f}, p99 {kl.quantile(0.99).item():.3f}) "
                  f"top1-agree {agree.float().mean().item():.3f} nll {-tlp.mean().item():.4f} vs teacher {-r['tgt_lp'].mean().item():.4f} "
                  f"mass {mass.mean().item():.3f}", flush=True)
            sq["_kl"] = kl.cpu(); sq["_agree"] = agree.cpu()
        recs.append({"i": i, "src": sq.get("src"), "ids": ids, "score_from": sf,
                     "top_v": v.half().cpu(), "top_i": ix.int().cpu(), "tgt_lp": tlp.cpu(),
                     "argmax_match": (ix[:, 0] == tgt).float().mean().item()})
        del logits, lg, lp
        dt = time.time() - t0
        print(f"[teacher] {i}: {len(ids)} tok, scored {len(ids) - sf}, nll {-tlp.mean().item():.4f}, "
              f"top1 {recs[-1]['argmax_match']:.3f}, {dt:.1f}s ({len(ids) / dt:.0f} tok/s)", flush=True)
        torch.save({"recs": recs, "counts": {k: v.copy() for k, v in STORE.counts.items()},
                    "kl": [q.get("_kl") for q in seqs[: i + 1]], "agree": [q.get("_agree") for q in seqs[: i + 1]]}, out_path)
    if ref is not None:
        kl = torch.cat([q["_kl"] for q in seqs]); ag = torch.cat([q["_agree"] for q in seqs]).float()
        print(f"[student] ALL: KL mean {kl.mean().item():.4f} median {kl.median().item():.4f} p99 {kl.quantile(0.99).item():.3f} "
              f"top1-agree {ag.mean().item():.4f} (n={kl.numel()})", flush=True)


def encode_chat(text, thinking="chat"):
    from encoding import encode_messages
    return encode_messages([{"role": "user", "content": text}], thinking_mode=thinking)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--raw", action="store_true", help="チャットテンプレートを使わない")
    ap.add_argument("--thinking", default="chat")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--expert-budget-gb", type=float, default=60)
    ap.add_argument("--out", default="")
    ap.add_argument("--teacher-in", default="")
    ap.add_argument("--teacher-ref", default="", help="教師の .pt (指定すると学生として採点)")
    ap.add_argument("--exl3-pack", default="", help="EXL3 パックのディレクトリ (専門家だけ差し替え)")
    ap.add_argument("--teacher-out", default="")
    a = ap.parse_args()
    model, tok, args = build({"max_batch_size": 1, "max_seq_len": a.max_seq_len, "temperature": 0.0},
                             a.expert_budget_gb, exl3_pack=a.exl3_pack)
    results = []
    if a.teacher_in:
        _patch_head()
        seqs = [json.loads(l) for l in open(a.teacher_in)]
        ref = torch.load(a.teacher_ref, weights_only=False) if a.teacher_ref else None
        teacher(model, seqs, a.teacher_out, ref=ref)
    for p in a.prompt:
        text = p if a.raw else encode_chat(p, a.thinking)
        ids = tok.encode(text, add_special_tokens=False)
        out, lg, tp, td = greedy(model, ids, a.max_new, tok.eos_token_id, keep_logits=True)
        comp = tok.decode(out)
        top5 = torch.topk(torch.log_softmax(lg, -1), 5)
        print(f"\n=== prompt ({len(ids)} tok) prefill {tp:.1f}s decode {len(out)} tok {td:.1f}s ({len(out) / max(td, 1e-9):.2f} tok/s)")
        print("top5 first:", [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top5.values, top5.indices)])
        print(repr(comp))
        results.append({"prompt": p, "ids": ids, "out": out, "text": comp, "prefill_s": tp, "decode_s": td})
    print("[experts]", STORE.stats, flush=True)
    eng = [b.engram.embed.stats for b in model.layers if b.engram is not None]
    print("[engram]", eng, flush=True)
    if a.out:
        json.dump({"results": results, "expert_stats": STORE.stats, "engram": eng,
                   "counts": {str(k): v.tolist() for k, v in STORE.counts.items()}},
                  open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
