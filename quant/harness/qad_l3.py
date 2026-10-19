#!/usr/bin/env python3
"""L3: 全体 KL による scale-only QAD の骨格 (DeepSeek-V4.1-Flash-1017、EXL3 K2 の学生)。

設計書は同じディレクトリの qad_l3_design.md。要点:
- 学習するのは routed experts の suh/svh (EXL3 の入力側と出力側の scale、40 層 x 384 x 22,272 = 342M) だけ (既定)。
  trellis (離散コード) は固定し、ffn_norm と router も固定する (SPEC §7 R-6)。Engram は FP8 のまま固定する。
- 学生の重みは W = diag(su) . Hb . Q . Hb . diag(sv) である。Q は trellis を su=sv=1 で復元した内側の行列、
  Hb は 128 次のブロック Hadamard。exllamav3 の LinearEXL3.get_weight_tensor()
  (build/exllamav3_151/exllamav3/modules/quant/exl3.py:239-250) と同じ式になる。
  W は作らず、x W = had(had(x * su) @ Q) * sv で計算する。su と sv の勾配は要素積の縮約で求まる。
- forward と backward は、公式 ref/model.py の Block/Attention を out-of-place に書き直したもので計算する。
  公式 module のパラメータはそのまま使う。量子化 (act_quant / fp4_act_quant) は STE で通す。
  indexer の top-k (離散) は no_grad で求め、選んだ位置を保存して recompute でも同じものを使う。
- 2 GPU のパイプライン (層 0-19 / 20-39) で 1F1B を回す。activation checkpointing は層単位で行い、
  MoE の出力は保存して recompute を省く。expert の Q は forward と backward で 1 回ずつ復元する。

使い方:
  python3 qad_l3.py selftest     # CPU で合成データの数値テストを回す (GPU 不要、/ref の model.py を使う)
  python3 qad_l3.py estimate     # 時間とメモリの見積もり表を出す
  torchrun --nproc_per_node 2 qad_l3.py train ...   # GPU 用 (未検証。設計書 §9 の手順で確かめる)

既存の stream_ref.py は変更しない。GPU 経路でのみ import して TensorIndex / HostEngramEmbedding を流用する。
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- 定数 (ref/inference_config.json)
DIM, INTER, N_LAYERS, N_EXPERTS, TOPK = 5120, 2304, 40, 384, 6
PARTS = ("w1", "w3", "w2")  # gate, up, down
HAD = 128  # exllamav3 の had_k = had_n = 128 (exl3_lib/quantize.py:15)
# S (scale ベクトル) の順序。ScaleParams と MoEScaleOnlyFn で共通。
SVEC = (("w1", "suh"), ("w1", "svh"), ("w3", "suh"), ("w3", "svh"), ("w2", "suh"), ("w2", "svh"))


# =========================================================================== 1. Hadamard
@functools.lru_cache(maxsize=None)
def _sylvester(n: int) -> torch.Tensor:
    """Sylvester 構成の Hadamard 行列 (±1)。exllamav3 の get_hadamard(128) は hadamard_1.txt からの
    Sylvester 構成なので同じ行列になる (exllamav3/util/hadamard.py:108-121)。対称行列。"""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    if h.shape[0] != n:
        raise ValueError(f"Hadamard {n} は 2 のべきでない")
    return h


_HCACHE: dict = {}


def hadamard(n: int, device, dtype) -> torch.Tensor:
    key = (n, str(device), dtype)
    if key not in _HCACHE:
        _HCACHE[key] = (_sylvester(n) / math.sqrt(n)).to(device=device, dtype=dtype)
    return _HCACHE[key]


def had_last(x: torch.Tensor, n: int = HAD) -> torch.Tensor:
    """最後の次元を n ごとのブロックに分け、右から H を掛ける (= preapply_had_r、had_r_128)。"""
    H = hadamard(n, x.device, x.dtype)
    return (x.unflatten(-1, (-1, n)) @ H).flatten(-2)


def had_first(w: torch.Tensor, n: int = HAD) -> torch.Tensor:
    """[k, n] 行列の行方向に左から H を掛ける (= preapply_had_l)。H は対称なので転置で書ける。"""
    return had_last(w.transpose(-1, -2), n).transpose(-1, -2)


def dense_from_inner(Q: torch.Tensor, su: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
    """W[in,out] = diag(su) Hb Q Hb diag(sv)。exl3.py:239-250 の get_weight_tensor と同じ式。"""
    return su[:, None] * had_last(had_first(Q)) * sv[None, :]


def inner_from_dense(W: torch.Tensor, su: torch.Tensor, sv: torch.Tensor) -> torch.Tensor:
    """逆変換 (テスト用): Q = Hb diag(1/su) W diag(1/sv) Hb。Hb は直交で対称。"""
    return had_last(had_first(W / su[:, None] / sv[None, :]))


# =========================================================================== 2. 内側行列 Q の供給元
class InnerSource:
    """(layer, expert, part) から内側の行列 Q [in, out] (Hadamard 基底、su=sv=1) を返す。
    base_scales(layer) は SVEC 順の 6 本の [E, n] テンソル (checkpoint の suh/svh)。"""

    def inner(self, layer: int, e: int, part: str) -> torch.Tensor:
        raise NotImplementedError

    def inner3(self, layer: int, e: int):
        return tuple(self.inner(layer, e, p) for p in PARTS)

    def inner3_batch(self, layer: int, es: list[int]):
        """bucket 用。[B, in, out] を 3 本返す。GPU 版は ext の batch 復元に置き換える余地がある。"""
        return tuple(torch.stack([self.inner(layer, e, p) for e in es]) for p in PARTS)

    def base_scales(self, layer: int) -> list[torch.Tensor]:
        raise NotImplementedError


class DenseInnerSource(InnerSource):
    """テスト用。dense な W[in,out] と base の su/sv から Q を作り、W を厳密に再現させる。"""

    def __init__(self, mats: dict, base: dict, dtype=torch.float32):
        # mats[(l, e, part)] = W[in, out]、base[l] = SVEC 順の 6 本 [E, n]
        self.base = base
        self.Q = {}
        for (l, e, p), W in mats.items():
            k = PARTS.index(p)
            su, sv = base[l][2 * k][e], base[l][2 * k + 1][e]
            self.Q[(l, e, p)] = inner_from_dense(W.double(), su.double(), sv.double()).to(dtype)

    def inner(self, layer, e, part):
        return self.Q[(layer, e, part)]

    def base_scales(self, layer):
        return self.base[layer]


class Exl3ResidentSource(InnerSource):
    """GPU 用 (未検証)。担当層の trellis を VRAM に常駐させ、ext.reconstruct で Q を復元する。

    1 expert の trellis は w1/w3 が [320, 144, 32] int16、w2 が [144, 320, 32] int16 で計 8.85MB
    (v0 パックのヘッダで確認)。20 層 x 384 で 67.95GB。復元は exl3.py:225-230 の _reconstruct_inner と同じ呼び方。
    """

    def __init__(self, pack_dir: str, layers, device, n_experts: int = N_EXPERTS, workers: int = 16):
        from concurrent.futures import ThreadPoolExecutor

        self.idx = SafetensorsIndex(pack_dir)
        self.dev = torch.device(device)
        self.layers = list(layers)
        self.E = n_experts
        self.trellis = {}  # (l, part) -> int16 [E, tk, tn, 16K]
        self.kmeta = {}  # (l, part) -> (K, mcg)
        self.base = {}
        pool = ThreadPoolExecutor(workers)
        for l in self.layers:
            for p in PARTS:
                shape = self.idx.shape(f"layers.{l}.ffn.experts.0.{p}.trellis")
                T = torch.empty((n_experts,) + tuple(shape), dtype=torch.int16, device=self.dev)
                names = [f"layers.{l}.ffn.experts.{e}.{p}.trellis" for e in range(n_experts)]
                # K (= trellis の最後の次元 / 16) と mcg/mul1 の有無は (層, 射影) ごとに持つ。
                # exl3.py の LinearEXL3 は K = trellis.shape[-1] / 16、mcg = (mcg テンソルがある) で決める
                # (build/exllamav3_151/exllamav3/modules/quant/exl3.py:67, 79)。K の違う層が混ざるパック (K3 混在) で
                # 層 0 の K を全層に使うと、ext.reconstruct が別の K で trellis を読む。
                bad = [nm for nm in names if self.idx.shape(nm) != tuple(shape)]
                if bad:
                    raise NotImplementedError(f"layer {l} {p}: expert ごとに trellis の形 (K) が違う ({bad[0]})。"
                                              "Exl3ResidentSource は (層, 射影) 内で K が同じことを前提にしている")
                pre = f"layers.{l}.ffn.experts.{{}}.{p}"
                has_mcg = [pre.format(e) + ".mcg" in self.idx.meta for e in range(n_experts)]
                if any(pre.format(e) + ".mul1" in self.idx.meta for e in range(n_experts)) or len(set(has_mcg)) > 1:
                    raise NotImplementedError(f"layer {l} {p}: mul1 コードブックか、mcg の有無が混在している")
                if shape[-1] % 16:
                    raise NotImplementedError(f"layer {l} {p}: 半整数 K (trellis {shape}) は mul1 が要る")
                self.kmeta[(l, p)] = (int(shape[-1] // 16), bool(has_mcg[0]))
                for e, arr in enumerate(pool.map(self.idx.read_numpy, names)):
                    T[e].copy_(torch.from_numpy(arr), non_blocking=False)
                self.trellis[(l, p)] = T
            vecs = []
            for p, kind in SVEC:
                names = [f"layers.{l}.ffn.experts.{e}.{p}.{kind}" for e in range(n_experts)]
                vecs.append(torch.stack([torch.from_numpy(a) for a in pool.map(self.idx.read_numpy, names)]).to(self.dev))
            self.base[l] = vecs
        self.K = self.kmeta[(self.layers[0], "w1")][0]  # 互換用 (表示だけ。復元は kmeta を使う)

    def inner(self, layer, e, part):
        from exllamav3.ext import exllamav3_ext as ext

        tr = self.trellis[(layer, part)][e]
        K, mcg = self.kmeta[(layer, part)]
        w = torch.empty((tr.shape[0] * 16, tr.shape[1] * 16), dtype=torch.half, device=tr.device)
        ext.reconstruct(w, tr, K, mcg, False)  # exl3.py:229 の _reconstruct_inner と同じ (mul1 は上で拒否)
        return w

    def base_scales(self, layer):
        return self.base[layer]


# =========================================================================== 3. scale-only の expert 計算
def scaled_proj(x, su, Q, sv, gemm_dtype):
    """x [..., in] -> [..., out] (fp32)。x @ W と同じ値 (W = diag(su) Hb Q Hb diag(sv))。
    su/sv は x にブロードキャストできる形 ([in] か bucket 時の [B, 1, in])。"""
    wd = torch.promote_types(x.dtype, torch.float32)  # 通常 fp32 (テストの double では double)
    a = had_last(x.to(wd) * su.to(wd))
    c = torch.matmul(a.to(gemm_dtype), Q.to(gemm_dtype))
    return had_last(c.to(wd)) * sv.to(wd)


def expert_scaled(x, w, Qs, S, limit, gemm_dtype, act_dtype, round_out=True):
    """stream_ref.expert_forward の bf16 経路 (stream_ref.py:271-280) を scale-only で書いたもの。
    clamp は gate が上側だけ、up が ±limit (ref/model.py:847-849、SPEC §7 R-7)。"""
    Q1, Q3, Q2 = Qs
    su1, sv1, su3, sv3, su2, sv2 = S
    g = scaled_proj(x, su1, Q1, sv1, gemm_dtype)
    u = scaled_proj(x, su3, Q3, sv3, gemm_dtype)
    if limit > 0:
        u = torch.clamp(u, min=-limit, max=limit)
        g = torch.clamp(g, max=limit)
    h = w * (F.silu(g) * u)
    o = scaled_proj(h.to(act_dtype), su2, Q2, sv2, gemm_dtype)
    return o.to(act_dtype) if round_out else o


def route_groups(indices: torch.Tensor, n_experts: int):
    """indices [T, k] -> [(e, tok, slot)]。token は expert ごとに高々 1 回現れる。"""
    k = indices.shape[1]
    flat = indices.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_experts).tolist()
    tok, slot = order // k, order % k
    out, off = [], 0
    for e, c in enumerate(counts):
        if c:
            out.append((e, tok[off:off + c], slot[off:off + c]))
        off += c
    return out


def make_buckets(groups, bucket: int):
    """token 数の多い順に並べ、bucket 本ずつまとめる (padding を減らすため)。"""
    if bucket <= 1:
        return [[g] for g in groups]
    gs = sorted(groups, key=lambda g: -len(g[1]))
    return [gs[i:i + bucket] for i in range(0, len(gs), bucket)]


@dataclass
class MoECall:
    """MoEScaleOnlyFn に渡す非テンソル引数。"""
    layer_id: int
    source: InnerSource
    limit: float
    gemm_dtype: torch.dtype = torch.bfloat16
    bucket: int = 16
    round_out: bool = True
    cache: dict | None = None  # checkpoint 時: 初回 forward の出力を置き、recompute で取り出す
    stats: dict | None = None


def _bucket_inputs(x, weights, bucket):
    B = len(bucket)
    C = max(len(g[1]) for g in bucket)
    X = x.new_zeros(B, C, x.shape[1])
    Wt = weights.new_zeros(B, C, 1)
    for b, (e, tok, slot) in enumerate(bucket):
        n = len(tok)
        X[b, :n] = x[tok]
        Wt[b, :n, 0] = weights[tok, slot]
    return X, Wt


def _moe_apply(x, weights, indices, S, call: MoECall):
    T, D = x.shape
    y = torch.zeros(T, D, dtype=torch.promote_types(x.dtype, torch.float32), device=x.device)
    groups = route_groups(indices, S[0].shape[0])
    for bucket in make_buckets(groups, call.bucket):
        es = [g[0] for g in bucket]
        if len(bucket) == 1:
            e, tok, slot = bucket[0]
            Qs = call.source.inner3(call.layer_id, e)
            o = expert_scaled(x[tok], weights[tok, slot, None], Qs, [s[e] for s in S], call.limit,
                              call.gemm_dtype, x.dtype, call.round_out)
            y.index_add_(0, tok, o.to(y.dtype))
            continue
        X, Wt = _bucket_inputs(x, weights, bucket)
        Qs = call.source.inner3_batch(call.layer_id, es)
        Sb = [s[es].unsqueeze(1) for s in S]
        O = expert_scaled(X, Wt, Qs, Sb, call.limit, call.gemm_dtype, x.dtype, call.round_out)
        for b, (e, tok, slot) in enumerate(bucket):
            y.index_add_(0, tok, O[b, :len(tok)].to(y.dtype))
    if call.stats is not None:
        call.stats["expert_visits"] = call.stats.get("expert_visits", 0) + len(groups)
    return y


class MoEScaleOnlyFn(torch.autograd.Function):
    """routed experts (scale-only 学習)。保存するのは x, weights, indices と scale だけで、
    backward で expert ごとに Q を復元して局所的に recompute する (Q は保存しない)。

    入力: x [T, dim]、weights [T, k] (router の重み、x に依存して微分可能)、indices [T, k] (定数)、
          S = su1, sv1, su3, sv3, su2, sv2 ([E, n] fp32)、call (MoECall)。
    出力: y [T, dim] fp32 (routed 分の和)。
    """

    @staticmethod
    def forward(ctx, x, weights, indices, su1, sv1, su3, sv3, su2, sv2, call: MoECall):
        S = (su1, sv1, su3, sv3, su2, sv2)
        y = None
        if call.cache is not None and call.layer_id in call.cache:
            y = call.cache.pop(call.layer_id)  # checkpoint の recompute: 初回の出力を使う
            if y.shape[0] != x.shape[0]:
                raise RuntimeError("MoE cache の形が合わない")
        if y is None:
            y = _moe_apply(x, weights, indices, S, call)
            if call.cache is not None:
                call.cache[call.layer_id] = y.detach()
        ctx.save_for_backward(x, weights, indices, *S)
        ctx.call = call
        return y

    @staticmethod
    def backward(ctx, dy):
        x, weights, indices, *S = ctx.saved_tensors
        call: MoECall = ctx.call
        need_S = ctx.needs_input_grad[3:9]
        dx = torch.zeros(x.shape, dtype=torch.promote_types(x.dtype, torch.float32), device=x.device)
        dw = torch.zeros_like(weights)
        dS = [torch.zeros_like(s) if n else None for s, n in zip(S, need_S)]
        groups = route_groups(indices, S[0].shape[0])
        for bucket in make_buckets(groups, call.bucket):
            es = [g[0] for g in bucket]
            if len(bucket) == 1:
                e, tok, slot = bucket[0]
                Qs = call.source.inner3(call.layer_id, e)
                with torch.enable_grad():
                    xe = x[tok].detach().requires_grad_(True)
                    we = weights[tok, slot, None].detach().requires_grad_(True)
                    se = [s[e].detach().requires_grad_(bool(n)) for s, n in zip(S, need_S)]
                    o = expert_scaled(xe, we, Qs, se, call.limit, call.gemm_dtype, x.dtype, call.round_out)
                    ins = [xe, we] + [s for s in se if s.requires_grad]
                    gr = torch.autograd.grad(o, ins, dy[tok].to(o.dtype))
                dx.index_add_(0, tok, gr[0].to(dx.dtype))
                dw[tok, slot] = gr[1][:, 0].to(dw.dtype)
                j = 2
                for k, n in enumerate(need_S):
                    if n:
                        dS[k][e] += gr[j]
                        j += 1
                continue
            X, Wt = _bucket_inputs(x, weights, bucket)
            DY = dy.new_zeros(X.shape[0], X.shape[1], dy.shape[1])
            for b, (e, tok, slot) in enumerate(bucket):
                DY[b, :len(tok)] = dy[tok]
            Qs = call.source.inner3_batch(call.layer_id, es)
            with torch.enable_grad():
                Xg = X.detach().requires_grad_(True)
                Wg = Wt.detach().requires_grad_(True)
                Sb = [s[es].unsqueeze(1).detach().requires_grad_(bool(n)) for s, n in zip(S, need_S)]
                O = expert_scaled(Xg, Wg, Qs, Sb, call.limit, call.gemm_dtype, x.dtype, call.round_out)
                ins = [Xg, Wg] + [s for s in Sb if s.requires_grad]
                gr = torch.autograd.grad(O, ins, DY.to(O.dtype))
            for b, (e, tok, slot) in enumerate(bucket):
                n = len(tok)
                dx.index_add_(0, tok, gr[0][b, :n].to(dx.dtype))
                dw[tok, slot] = gr[1][b, :n, 0].to(dw.dtype)
            j = 2
            for k, n in enumerate(need_S):
                if n:
                    dS[k][es] += gr[j][:, 0]
                    j += 1
        return (dx.to(x.dtype), dw, None, *dS, None)


# =========================================================================== 4. 量子化のエミュレーションと STE
_FP8_MAX_INV = torch.tensor(1.0 / 448.0, dtype=torch.float32)
_FP4_MAX_INV = torch.tensor(1.0 / 6.0, dtype=torch.float32)


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2 v) (v > 0、正規数)。kernel.py の fast_round_scale (ビット演算) と同じ値。"""
    m, e = torch.frexp(v)  # v = m * 2^e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)
    return torch.ldexp(torch.ones_like(v), e)


def qdq_fp8_emul(x: torch.Tensor, block: int = 32, pow2: bool = True) -> torch.Tensor:
    """act_quant(inplace=True) のエミュレーション (ref/kernel.py:40-95)。e4m3、block ごとの scale。"""
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = amax * _FP8_MAX_INV.to(x.device)
    if pow2:
        s = _pow2_ceil(s)
    y = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.flatten(-2).to(x.dtype)


def fp4_round(v: torch.Tensor) -> torch.Tensor:
    """e2m1 への最近接丸め (偶数側へ)。格子 {0, .5, 1, 1.5, 2, 3, 4, 6}。v は [-6, 6] に clamp 済み。"""
    a = v.abs()
    q = torch.zeros_like(a)
    q = torch.where(a > 0.25, torch.full_like(a, 0.5), q)
    q = torch.where(a >= 0.75, torch.full_like(a, 1.0), q)
    q = torch.where(a > 1.25, torch.full_like(a, 1.5), q)
    q = torch.where(a >= 1.75, torch.full_like(a, 2.0), q)
    q = torch.where(a > 2.5, torch.full_like(a, 3.0), q)
    q = torch.where(a >= 3.5, torch.full_like(a, 4.0), q)
    q = torch.where(a > 5.0, torch.full_like(a, 6.0), q)
    return torch.copysign(q, v)


def qdq_fp4_emul(x: torch.Tensor, block: int = 32, e4m3: bool = False) -> torch.Tensor:
    """fp4_act_quant(inplace=True) のエミュレーション (ref/kernel.py:127-181)。"""
    xf = x.float().unflatten(-1, (-1, block))
    amax = xf.abs().amax(-1, keepdim=True)
    if e4m3:
        amax = amax.clamp_min(6 * 2.0 ** -9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0 ** -126)
        s = _pow2_ceil(amax * _FP4_MAX_INV.to(x.device))
    y = fp4_round((xf / s).clamp(-6.0, 6.0)) * s
    return y.flatten(-2).to(x.dtype)


class _STEFn(torch.autograd.Function):
    """forward は fn(x)、backward は恒等 (straight-through)。x + (q - x).detach() と違い値が厳密に q になる。"""

    @staticmethod
    def forward(ctx, x, fn):
        return fn(x)

    @staticmethod
    def backward(ctx, g):
        return g, None


def hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc: int = 4, iters: int = 20, eps: float = 1e-6):
    """kernel.hc_split_sinkhorn (ref/kernel.py:406-474) の torch 版。微分できる。"""
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = mixes[..., 2 * hc:].unflatten(-1, (hc, hc)) * hc_scale[2] + hc_base[2 * hc:].view(hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


class KernelSet:
    """量子化と sinkhorn の実装を選ぶ。mode="emul" は CPU とテスト用、"official" は ref/kernel.py の tilelang。
    微分が要る所は STE で包む (forward の値は選んだ実装、backward は恒等)。"""

    def __init__(self, mode: str = "emul", K=None, M=None):
        self.mode, self.K, self.M = mode, K, M

    def act_qdq(self, x, block=32):
        if self.mode == "official":
            y = x.detach().clone().contiguous()
            self.K.act_quant(y, block, "ue8m0", torch.float8_e8m0fnu, True)
            return y
        return qdq_fp8_emul(x, block, pow2=True)

    def fp4_qdq(self, x, block=32, e4m3=False):
        if self.mode == "official":
            y = x.detach().clone().contiguous()
            sd = torch.float8_e4m3fn if e4m3 else torch.float8_e8m0fnu
            self.K.fp4_act_quant(y, block, True, scale_dtype=sd)
            return y
        return qdq_fp4_emul(x, block, e4m3)

    def ste_act(self, x, block=32):
        return _STEFn.apply(x, lambda t: self.act_qdq(t, block))

    def ste_fp4(self, x, block=32, e4m3=False):
        return _STEFn.apply(x, lambda t: self.fp4_qdq(t, block, e4m3))

    def sinkhorn(self, mixes, scale, base, hc, iters, eps):
        # 学習中は常に torch 版 (fp32、[T, 24] の小さな演算)。公式 kernel との差は GPU 手順 G3 で測る。
        return hc_split_sinkhorn_torch(mixes, scale, base, hc, iters, eps)

    def linear_fwd(self, x, weight):
        """FP8 重みの forward。official は公式 linear() (act_quant + fp8_gemm、ref/model.py:179-207)。"""
        if self.mode == "official":
            return self.M.linear(x, weight)
        w = dequant_fp8_weight(weight, weight.scale)
        return F.linear(qdq_fp8_emul(x, 32, True).to(x.dtype), w.to(x.dtype))


def dequant_fp8_weight(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP8 (e4m3) 重み [out, in] と 32x32 ブロックの scale (e8m0) から fp32 の重みを作る。"""
    out_f, in_f = w.shape
    sf = s.float().repeat_interleave(block, 0)[:out_f].repeat_interleave(block, 1)[:, :in_f]
    return w.float() * sf


class Fp8LinearSTE(torch.autograd.Function):
    """FP8 重みの Linear。forward は公式 kernel (かエミュレーション)、backward の dX は
    dY @ W_deq (活性の量子化は STE)。重みの勾配は出さない (固定)。"""

    @staticmethod
    def forward(ctx, x, mod, ks: KernelSet):
        ctx.mod = mod
        ctx.xdtype = x.dtype
        return ks.linear_fwd(x, mod.weight)

    @staticmethod
    def backward(ctx, dy):
        w = dequant_fp8_weight(ctx.mod.weight, ctx.mod.weight.scale)
        dt = torch.bfloat16 if dy.dtype in (torch.bfloat16, torch.float16) else dy.dtype
        dx = dy.to(dt) @ w.to(dt)
        return dx.to(ctx.xdtype), None, None


def lin(mod, x, ks: KernelSet):
    """公式 linear() の微分できる版。FP8 は STE、bf16 / fp32 はそのまま F.linear。"""
    wt = mod.weight
    if wt.dtype == torch.float8_e4m3fn:
        return Fp8LinearSTE.apply(x, mod, ks)
    if wt.dtype == getattr(torch, "float4_e2m1fn_x2", None):
        raise TypeError("FP4 の Linear は routed experts 以外に無いはず")
    return F.linear(x, wt)


# =========================================================================== 5. sparse attention (微分可能)
def sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale, chunk=512, round_p=True, return_lse=False):
    """stream_ref.sparse_attn_torch (stream_ref.py:83-102) と同じ計算。lse も返せるようにした版。"""
    b, s, h, d = q.shape
    wd = torch.promote_types(q.dtype, torch.float32)
    out = torch.empty_like(q)
    lse = torch.empty(b, s, h, dtype=wd, device=q.device)
    sink = attn_sink.to(wd).view(1, h, 1)
    for bi in range(b):
        for s0 in range(0, s, chunk):
            s1 = min(s, s0 + chunk)
            idx = topk_idxs[bi, s0:s1].long()
            valid = idx >= 0
            g = kv[bi][idx.clamp_min(0)].to(wd)
            sc = torch.einsum("shd,skd->shk", q[bi, s0:s1].to(wd), g) * softmax_scale
            sc = sc.masked_fill(~valid.unsqueeze(1), float("-inf"))
            m = torch.maximum(sc.amax(-1, keepdim=True), sink.expand(s1 - s0, h, 1))
            m = torch.where(torch.isinf(m), torch.zeros_like(m), m)
            p = torch.exp(sc - m)
            den = p.sum(-1, keepdim=True) + torch.exp(sink - m)
            pp = p.to(torch.bfloat16).to(wd) if round_p else p
            o = torch.einsum("shk,skd->shd", pp, g) / den
            out[bi, s0:s1] = o.to(q.dtype)
            lse[bi, s0:s1] = (m + den.log())[..., 0]
    return (out, lse) if return_lse else out


class SparseAttnFn(torch.autograd.Function):
    """sparse_attn の autograd 版。K = V = kv (位置ごとに 1 本、ヘッド間で共有)。attn_sink は分母にだけ入る。
    backward は chunk ごとに gather をやり直し、P を lse から再構成する (FlashAttention と同じ形)。
    attn_sink の勾配は出さない (固定)。"""

    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, softmax_scale, chunk=512, round_p=True):
        out, lse = sparse_attn_ref(q, kv, attn_sink, topk_idxs, softmax_scale, chunk, round_p, True)
        ctx.save_for_backward(q, kv, topk_idxs, lse)
        ctx.scale, ctx.chunk = softmax_scale, chunk
        return out

    @staticmethod
    def backward(ctx, do):
        q, kv, topk_idxs, lse = ctx.saved_tensors
        scale, chunk = ctx.scale, ctx.chunk
        b, s, h, d = q.shape
        wd = torch.promote_types(q.dtype, torch.float32)
        dq = torch.zeros(q.shape, dtype=wd, device=q.device)
        dkv = torch.zeros(kv.shape, dtype=wd, device=kv.device)
        for bi in range(b):
            for s0 in range(0, s, chunk):
                s1 = min(s, s0 + chunk)
                idx = topk_idxs[bi, s0:s1].long()
                valid = idx >= 0
                g = kv[bi][idx.clamp_min(0)].to(wd)  # [sc, k, d]
                qf = q[bi, s0:s1].to(wd)
                sc = torch.einsum("shd,skd->shk", qf, g) * scale
                sc = sc.masked_fill(~valid.unsqueeze(1), float("-inf"))
                P = torch.exp(sc - lse[bi, s0:s1].unsqueeze(-1))  # [sc, h, k]
                dO = do[bi, s0:s1].to(wd)
                O = torch.einsum("shk,skd->shd", P, g)
                delta = (dO * O).sum(-1, keepdim=True)
                dP = torch.einsum("shd,skd->shk", dO, g)
                dS = P * (dP - delta)
                dq[bi, s0:s1] = torch.einsum("shk,skd->shd", dS, g) * scale
                dg = torch.einsum("shk,shd->skd", P, dO) + torch.einsum("shk,shd->skd", dS, qf) * scale
                dg = dg * valid.unsqueeze(-1)
                dkv[bi].index_add_(0, idx.clamp_min(0).reshape(-1), dg.reshape(-1, d))
        return dq.to(q.dtype), dkv.to(kv.dtype), None, None, None, None, None


# =========================================================================== 6. 公式 Block の out-of-place 版
def rope_tail(x, freqs, rd, inverse=False):
    """apply_rotary_emb (ref/model.py:393-411) を最後の rd 次元にだけ、out-of-place で掛ける。"""
    head, tail = x[..., :-rd], x[..., -rd:]
    t = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    if t.ndim == 3:
        f = f.view(1, t.size(1), t.size(-1))
    else:
        f = f.view(1, t.size(1), 1, t.size(-1))
    t = torch.view_as_real(t * f).flatten(-2).to(x.dtype)
    return torch.cat([head, t], -1)


def hc_mixes(blk, x, fn, scale, base, ks: KernelSet):
    """Block.hc_mixes (ref/model.py:950-957)。"""
    xf = x.flatten(2).float()
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + blk.norm_eps)
    mixes = F.linear(xf, fn) * rs
    return ks.sinkhorn(mixes, scale, base, blk.hc_mult, blk.hc_sinkhorn_iters, blk.hc_eps)


def hc_post(x, residual, post, comb):
    """Block.hc_post (ref/model.py:964-968)。[b,s,hc,hc,d] の一時テンソル (4096 tok で 1.3GB) を避けて einsum にした。"""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.einsum("bsjk,bsjd->bskd", comb, residual.float())
    return y.type_as(x)


def compressor_forward(C, x, ks):
    """Compressor.forward (ref/model.py:458-485) の prefill 部分。状態バッファには書かない。"""
    ratio = C.compress_ratio
    if ratio == 1:
        return C.norm(lin(C.wkv, x, ks))
    bsz, seqlen, _ = x.shape
    if seqlen < ratio:
        return None
    xf = x.float()
    kv, score = lin(C.wkv, xf, ks), lin(C.wgate, xf, ks)
    cut = seqlen - seqlen % ratio
    kv = kv[:, :cut].unflatten(1, (-1, ratio))
    score = score[:, :cut].unflatten(1, (-1, ratio))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
    return C.norm(kv.to(x.dtype))


def engram_apply(E, h, rows, ks):
    """Engram.forward (ref/model.py:350-366)。表の行 rows (定数) から kv を作り、h に足す。"""
    with torch.no_grad():
        kv = lin(E.wkv, rows.flatten(-2), ks)
    key, value = kv.split([E.hc_mult * E.dim, E.dim], dim=-1)
    key = key.float().unflatten(-1, (E.hc_mult, E.dim))
    weight = E.q_weight.float() * E.k_weight.float()
    hf, eps = h.float(), E.eps
    rstd = torch.rsqrt(hf.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (hf * weight * key).sum(-1) * rstd * E.dim ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(E.clamp_value).sqrt(), dot))
    return (hf + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)


def expert_dense(exp, x, ks):
    """公式 Expert.forward (ref/model.py:843-853)。shared expert 用 (FP8 は STE)。"""
    gate = lin(exp.w1, x, ks).float()
    up = lin(exp.w3, x, ks).float()
    if exp.swiglu_limit > 0:
        up = torch.clamp(up, min=-exp.swiglu_limit, max=exp.swiglu_limit)
        gate = torch.clamp(gate, max=exp.swiglu_limit)
    return lin(exp.w2, (F.silu(gate) * up).to(x.dtype), ks)


# =========================================================================== 7. 学習する scale
class ScaleParams(torch.nn.Module):
    """routed experts の suh/svh を base * exp(theta) で表す。theta (fp32) を学習する。
    - 符号は base のまま保たれる (Hadamard の符号反転を壊さない)。
    - 学習率が expert ごとの scale の大きさに依らない (相対変化で動く)。
    - up の svh と down の suh は clamp の外では積しか効かない冗長な自由度なので、theta の L2 (= 初期値への
      対数スケールの正則化) が必須 (SPEC §7.3)。
    """

    def __init__(self, base: dict, device=None):
        super().__init__()
        self.layers = sorted(base)
        self.theta = torch.nn.ParameterDict()
        for l in self.layers:
            for k in range(len(SVEC)):
                b = base[l][k].detach().to(device=device, dtype=torch.float32)
                self.register_buffer(f"base_{l}_{k}", b, persistent=False)
                self.theta[f"{l}_{k}"] = torch.nn.Parameter(torch.zeros_like(b))

    def scales(self, l: int):
        return [getattr(self, f"base_{l}_{k}") * torch.exp(self.theta[f"{l}_{k}"]) for k in range(len(SVEC))]

    def reg(self):
        """theta^2 の平均。全体の損失に lambda を掛けて足す。"""
        tot = sum(t.square().sum() for t in self.theta.values())
        n = sum(t.numel() for t in self.theta.values())
        return tot / max(n, 1)

    @torch.no_grad()
    def export_fp16(self):
        """{checkpoint のテンソル名: fp16 テンソル} と統計を返す。ds41rt は非有限値だけを拒否する
        (v41_exl3_residency.rs:109-117、SPEC §7.3)。fp16 の範囲 (65504) と非正規数の割合も検査する。"""
        out, n_sub, n_tot, maxrel = {}, 0, 0, 0.0
        for l in self.layers:
            for k, (p, kind) in enumerate(SVEC):
                b = getattr(self, f"base_{l}_{k}")
                v = (b * torch.exp(self.theta[f"{l}_{k}"])).float()
                if not torch.isfinite(v).all() or v.abs().max() > 65504:
                    raise ValueError(f"layer {l} {p}.{kind}: fp16 に入らない値")
                h = v.half()
                n_sub += int(((h.abs() < 6.104e-5) & (h != 0)).sum())
                n_tot += h.numel()
                maxrel = max(maxrel, float(self.theta[f"{l}_{k}"].abs().max().exp() - 1))
                for e in range(h.shape[0]):
                    out[f"layers.{l}.ffn.experts.{e}.{p}.{kind}"] = h[e].cpu().contiguous()
        return out, {"subnormal_frac": n_sub / max(n_tot, 1), "max_rel_change": maxrel}


def base_digests(get_base) -> dict:
    """{"{l}_{k}": sha256} の形で、theta の基準 (checkpoint の suh/svh、fp16) の指紋を返す。
    get_base(l, k) は SVEC[k] の [E, n] (fp16 か、fp16 から変換した fp32)。export で --pack の値と照合する
    (L2 の出力 pack で学習した theta を v1 に当てる、のような取り違えを防ぐ)。"""
    import hashlib
    return {key: hashlib.sha256(np.ascontiguousarray(t.detach().cpu().half().numpy()).tobytes()).hexdigest()
            for key, t in get_base()}


# =========================================================================== 8. 損失 (教師 top-k)
def kl_topk(logits, top_v, top_i, tail=True):
    """教師の top-k logprob (top_v) と id (top_i) に対する位置ごとの KL(P||Q)。
    tail=False は stream_ref.teacher の採点 (stream_ref.py:452-458) と同じ近似 (top-k 外を無視)。
    tail=True は top-k 外を 1 つの桶にまとめた項を足す (粗くした分布どうしの KL なので非負)。"""
    wd = torch.promote_types(logits.dtype, torch.float32)
    logq = torch.log_softmax(logits.to(wd), -1)
    tv = top_v.to(wd)
    p = tv.exp()
    lq = logq.gather(-1, top_i.long())
    kl = (p * (tv - lq)).sum(-1)
    if tail:
        pt = (1.0 - p.sum(-1)).clamp_min(1e-8)
        lse = torch.logsumexp(lq, -1)
        lqt = torch.log((-torch.expm1(lse.clamp_max(-1e-7))).clamp_min(1e-8))
        kl = kl + pt * (pt.log() - lqt)
    return kl, logq


def head_kl_loss(x, head_w, top_v, top_i, w, chunk=1024, tail=True, ckpt=True):
    """最終 norm 後の隠れ状態 x [n, dim] から、chunk ごとに logits を作って重み付き KL を足す。
    logits [n, 129280] fp32 を全部は持たない (4096 位置で 2.1GB、勾配込みでその数倍)。
    返り値: (重み付き KL の和、重みの和、tail なし KL [n] (detach)、top-1 一致 [n])。"""
    n = x.shape[0]
    total = x.new_zeros((), dtype=torch.promote_types(head_w.dtype, torch.float32))
    kl_top_all, agree_all = [], []

    def piece(xc, tv, ti, wc):
        logits = F.linear(xc.to(head_w.dtype), head_w)  # 公式 ParallelHead と同じ fp32 (ref/model.py:1010-1018)
        kl, logq = kl_topk(logits, tv, ti, tail)
        with torch.no_grad():
            tvw = tv.to(logq.dtype)
            klt = (tvw.exp() * (tvw - logq.gather(-1, ti.long()))).sum(-1)
            ag = (logits.argmax(-1) == ti[:, 0].long()).float()
        return (kl * wc).sum(), klt, ag

    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        args = (x[s0:s1], top_v[s0:s1], top_i[s0:s1], w[s0:s1])
        if ckpt and torch.is_grad_enabled():
            l, klt, ag = torch.utils.checkpoint.checkpoint(piece, *args, use_reentrant=False)
        else:
            l, klt, ag = piece(*args)
        total = total + l
        kl_top_all.append(klt)
        agree_all.append(ag)
    return total, w.sum(), torch.cat(kl_top_all), torch.cat(agree_all)


# =========================================================================== 9. データ (教師 top-32 と日本語の重み)
_KANA_RE = re.compile("[぀-ヿㇰ-ㇿｦ-ﾟ]")
_CJK_RE = re.compile("[㐀-䶿一-鿿豈-﫿぀-ヿㇰ-ㇿｦ-ﾟ　-〿！-｠]")


def classify_ja(text: str) -> bool:
    """CJK と仮名が空白以外の 25% 以上で、仮名を含む窓を日本語とする (SPEC §2.1 の区分の規則に
    仮名の条件を足して中国語と分ける)。"""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    n = len(chars)
    cjk = sum(1 for c in chars if _CJK_RE.match(c))
    kana = sum(1 for c in chars if _KANA_RE.match(c))
    return cjk / n >= 0.25 and kana >= max(1, 0.02 * n)


def ja_position_weights(ids, score_from, decode, w_ja=2.0, chunk=32):
    """採点位置 j (ids[score_from + j] を予測する位置) の重み。予測先の token を含む chunk の文字種で決める。
    SPEC §2.5 の c_t (日本語 2.0、それ以外 1.0)。"""
    T = len(ids)
    flags = np.zeros(T, dtype=bool)
    for s0 in range(0, T, chunk):
        flags[s0:s0 + chunk] = classify_ja(decode(ids[s0:s0 + chunk]))
    w = np.ones(T - score_from, dtype=np.float32)
    w[flags[score_from:]] = w_ja
    return w, flags[score_from:]


@dataclass
class Microbatch:
    """1 系列 (<= 4096 トークン)。pos は隠れ状態の添字 (score_from-1 .. len-2)。"""
    ids: torch.Tensor          # [1, T] long
    pos: torch.Tensor          # [n] long
    top_v: torch.Tensor        # [n, 32] fp16 (教師 logprob)
    top_i: torch.Tensor        # [n, 32] int32
    w: torch.Tensor            # [n] fp32 (日本語 2.0)
    meta: dict = field(default_factory=dict)
    # forward 1 回ごとの状態
    ckpt: bool = False
    cache_topk: bool = True
    topk: dict = field(default_factory=dict)      # layer -> 圧縮 KV の top-k 位置 (recompute で再利用)
    moe_y: dict = field(default_factory=dict)     # layer -> routed MoE の出力 (recompute で再利用)
    rt_topk: torch.Tensor | None = None
    engram_rows: dict = field(default_factory=dict)  # layer -> Engram 表の行 (定数、epoch をまたいで使える)

    def new_pass(self, ckpt: bool):
        self.ckpt = ckpt
        self.topk, self.moe_y, self.rt_topk = {}, {}, None
        return self

    def to(self, device):
        for k in ("ids", "pos", "top_v", "top_i", "w"):
            setattr(self, k, getattr(self, k).to(device))
        return self


def microbatch_from_record(r: dict, decode=None, w_ja=2.0, domain=None) -> Microbatch:
    """stream_ref.teacher が保存する rec (stream_ref.py:465-467) から Microbatch を作る。
    L3 用の教師は jsonl の score_from を 1 にして全位置の top-32 を取る (設計書 §5)。"""
    ids, sf = r["ids"], r["score_from"]
    n = len(ids) - sf
    ja = None
    if decode is not None:
        w, flags = ja_position_weights(ids, sf, decode, w_ja)
        ja = torch.from_numpy(np.ascontiguousarray(flags))  # 位置ごとの日本語判定 (summarize の pos:ja に使う)
    else:
        w = np.full(n, w_ja if domain == "ja" else 1.0, dtype=np.float32)
    return Microbatch(ids=torch.tensor([ids], dtype=torch.long),
                      pos=torch.arange(sf - 1, len(ids) - 1, dtype=torch.long),
                      top_v=r["top_v"], top_i=r["top_i"], w=torch.from_numpy(w),
                      meta={"src": r.get("src"), "i": r.get("i"), "domain": domain, "ja": ja})


def load_teacher(path: str):
    d = torch.load(path, map_location="cpu", weights_only=False)
    return d["recs"]


# =========================================================================== 10. 学生のステージ
@dataclass
class L3Config:
    gemm_dtype: torch.dtype = torch.bfloat16
    bucket: int = 16
    attn_chunk: int = 256
    round_p: bool = True          # sparse attn の P を bf16 に丸める (公式 kernel と harness と同じ)
    loss_chunk: int = 1024
    tail: bool = True
    round_expert_out: bool = True


class StudentStage(torch.nn.Module):
    """公式 Transformer (担当層だけ実体化したもの) の層を、微分できる形で前に進める。

    model: ref/model.py の Transformer。layer_ids: このステージの層。first: embed を持つ。last: norm/head を持つ。
    scales: ScaleParams (担当層)。source: InnerSource。"""

    def __init__(self, M, model, layer_ids, first, last, scales: ScaleParams, source: InnerSource,
                 ks: KernelSet, cfg: L3Config):
        super().__init__()
        self.M, self.model = M, model
        self.layer_ids = list(layer_ids)
        self.first, self.last = first, last
        self.scales, self.source, self.ks, self.cfg = scales, source, ks, cfg
        self.hc_mult = model.hc_mult

    def block(self, lid):
        return self.model.layers[lid]

    # ---- attention
    def _attn(self, A, x, mb: Microbatch, lid, ckv):
        ks, cfg = self.ks, self.cfg
        bsz, seqlen, _ = x.shape
        rd = A.rope_head_dim
        freqs = A.freqs_cis[:seqlen]
        qr = A.q_norm(lin(A.wq_a, x, ks))
        q = lin(A.wq_b, qr, ks).unflatten(-1, (A.n_local_heads, A.head_dim))
        q = rope_tail(q, freqs, rd)
        kv = A.kv_norm(lin(A.wkv, x, ks))
        kv = rope_tail(kv, freqs, rd)
        kv = ks.ste_act(kv, 32)
        idx = self.M.get_window_topk_idxs(A.window_size, bsz, seqlen, 0).to(x.device)
        ckv_new = None
        if A.compress_ratio:
            ratio = A.compress_ratio
            clen = seqlen // ratio
            latent = compressor_forward(A.compressor, x, ks) if A.is_kv_source else None
            cidx = mb.topk.get(lid)
            if cidx is None:
                if A.is_index_source:
                    if clen == 0:
                        cidx = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
                    else:
                        if A.indexer.freqs_cis is None:
                            A.indexer.freqs_cis = A.freqs_cis
                        with torch.no_grad():
                            cidx = A.indexer(x.detach(), qr.detach(), None if latent is None else latent.detach(), 0, seqlen)
                    mb.rt_topk = cidx
                else:
                    cidx = mb.rt_topk
                    if cidx is None:
                        raise RuntimeError(f"layer {lid}: 前段の index source が同じステージに無い")
                if mb.cache_topk:
                    mb.topk[lid] = cidx
            if A.is_kv_source:
                if latent is None:
                    src = kv.new_zeros(bsz, 0, kv.shape[-1])
                else:
                    f = A.freqs_cis[: seqlen - seqlen % ratio: ratio]
                    latent = rope_tail(latent, f, rd)
                    latent = ks.ste_fp4(latent, 16, e4m3=True)
                    src = ckv_new = latent
            else:
                if ckv is None:
                    raise RuntimeError(f"layer {lid}: 圧縮 KV の供給元 (kv_source) が同じステージに無い")
                src = ckv
            kv = torch.cat([kv, src[:, :clen]], dim=1)
            idx = torch.cat([idx, cidx], dim=-1)
        o = SparseAttnFn.apply(q, kv, A.attn_sink, idx, A.softmax_scale, cfg.attn_chunk, cfg.round_p)
        o = rope_tail(o, freqs, rd, inverse=True)
        o = o.view(bsz, seqlen, A.n_local_groups, -1)
        o = ks.ste_act(o, 32)
        wo_a = A.wo_a.weight.view(A.n_local_groups, A.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return lin(A.wo_b, o.flatten(2), ks), ckv_new

    # ---- MoE
    def _moe(self, ffn, x, mb: Microbatch, lid):
        shape = x.shape
        x2 = x.reshape(-1, ffn.dim)
        weights, indices = ffn.gate(x2, None)
        S = self.scales.scales(lid)
        call = MoECall(lid, self.source, ffn.shared_experts.swiglu_limit, self.cfg.gemm_dtype, self.cfg.bucket,
                       self.cfg.round_expert_out, cache=mb.moe_y if mb.ckpt else None)
        y = MoEScaleOnlyFn.apply(x2, weights, indices.detach(), *S, call)
        y = y + expert_dense(ffn.shared_experts, x2, self.ks).float()
        return y.type_as(x).view(shape)

    # ---- 1 層 (Block.forward、ref/model.py:970-997)
    def _layer(self, lid, mb: Microbatch, h, pre_mix, ckv):
        blk = self.block(lid)
        ks = self.ks
        if blk.engram is not None:
            h = engram_apply(blk.engram, h, mb.engram_rows[lid].to(h.device, non_blocking=True), ks)
        residual = h
        a_pre, a_post, a_comb = hc_mixes(blk, h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base, ks)
        x = blk.hc_pre(h, pre_mix)
        x = blk.attn_norm(x)
        x, ckv_new = self._attn(blk.attn, x, mb, lid, ckv)
        h = hc_post(x, residual, a_post, a_comb)
        residual = h
        f_pre, f_post, f_comb = hc_mixes(blk, h, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base, ks)
        x = blk.hc_pre(h, a_pre)
        x = blk.ffn_norm(x)
        x = self._moe(blk.ffn, x, mb, lid)
        h = hc_post(x, residual, f_post, f_comb)
        return h, f_pre, ckv_new

    def prepare(self, mb: Microbatch):
        """stage 0: Engram 表の行を集める (定数)。GPU 経路では HostEngramEmbedding の pread になる。"""
        if not self.first or self.model.engram_hash is None:
            return
        need = [l for l in self.layer_ids if self.block(l).engram is not None and l not in mb.engram_rows]
        if not need:
            return
        with torch.no_grad():
            hashes = self.model.engram_hash(mb.ids, 0, None).clone()
            for l in need:
                E = self.block(l).engram
                # ホスト RAM に置く (1 系列 2 層で bf16 約 100MB、1M トークンで約 25GB)。GPU に置くと epoch をまたいで溜まる
                mb.engram_rows[l] = E.embed(hashes[:, :, E.layer_hash_index, :]).to("cpu")

    def forward(self, mb: Microbatch, h=None, pre_mix=None):
        if self.first:
            self.prepare(mb)
            with torch.no_grad():
                e = self.model.embed(mb.ids)
                h = e.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
                pre_mix = self.M.make_identity_pre_mix(h, self.hc_mult)
        ckv = None
        for lid in self.layer_ids:
            if mb.ckpt and torch.is_grad_enabled():
                h, pre_mix, c = torch.utils.checkpoint.checkpoint(self._layer, lid, mb, h, pre_mix, ckv,
                                                                  use_reentrant=False)
            else:
                h, pre_mix, c = self._layer(lid, mb, h, pre_mix, ckv)
            if c is not None:
                ckv = c
        if not self.last:
            return h, pre_mix
        hidden = self.block(self.layer_ids[-1]).hc_pre(h, pre_mix)
        x = self.model.norm(hidden)[0]
        return head_kl_loss(x[mb.pos], self.model.head.weight, mb.top_v, mb.top_i, mb.w,
                            self.cfg.loss_chunk, self.cfg.tail, ckpt=mb.ckpt)


# =========================================================================== 11. パイプライン (1F1B)
def schedule_1f1b(n_micro: int, n_stages: int, stage: int):
    """非 interleaved の 1F1B。stage s は warmup で (S-1-s) 本 forward してから F と B を交互に行う。"""
    warm = min(n_stages - stage - 1, n_micro)
    ops = [("F", i) for i in range(warm)]
    f, b = warm, 0
    while f < n_micro:
        ops.append(("F", f))
        f += 1
        ops.append(("B", b))
        b += 1
    while b < n_micro:
        ops.append(("B", b))
        b += 1
    return ops


class StageRunner:
    """1 ステージ分の F/B を実行する。入出力のやりとりは link (LocalLink か DistLink) が担う。"""

    def __init__(self, stage: StudentStage, sid: int, n_stages: int, link, ckpt=True, accum=1):
        self.stage, self.sid, self.S, self.link, self.ckpt, self.accum = stage, sid, n_stages, link, ckpt, accum
        self.saved = {}
        self.metrics = []

    def ready(self, op):
        kind, i = op
        if kind == "F":
            return self.sid == 0 or self.link.has("act", self.sid, i)
        return self.sid == self.S - 1 or self.link.has("grad", self.sid, i)

    def run(self, op, mbs):
        kind, i = op
        if kind == "F":
            # ステージごとに forward 1 回分の状態 (top-k、MoE の出力) を持つ。同じプロセスで複数ステージを
            # 回すとき (CPU テスト) に、別ステージの new_pass() が backward 前の状態を消さないようにする。
            mb = copy.copy(mbs[i]).new_pass(self.ckpt)
            if self.sid == 0:
                out = self.stage(mb)
                inp = None
            else:
                h, pm = self.link.recv("act", self.sid, i)
                h = h.detach().requires_grad_(True)
                pm = pm.detach().requires_grad_(True)
                inp = (h, pm)
                out = self.stage(mb, h, pm)
            if self.sid == self.S - 1:
                lsum, wsum, klt, ag = out
                loss = lsum / wsum.clamp_min(1e-6) / self.accum
                self.metrics.append({"i": i, "kl_top": float(klt.mean()), "agree": float(ag.mean()),
                                     "loss": float(loss.detach()) * self.accum})
                self.saved[i] = (inp, loss)
            else:
                self.link.send("act", self.sid + 1, i, (out[0].detach(), out[1].detach()))
                self.saved[i] = (inp, out)
        else:
            inp, out = self.saved.pop(i)
            if self.sid == self.S - 1:
                out.backward()
            else:
                gh, gpm = self.link.recv("grad", self.sid, i)
                torch.autograd.backward([out[0], out[1]], [gh, gpm])
            if inp is not None:
                self.link.send("grad", self.sid - 1, i, (inp[0].grad, inp[1].grad))


class LocalLink:
    """1 プロセス内のメールボックス (CPU テスト用)。"""

    def __init__(self):
        self.box = {}

    def has(self, kind, dst, i):
        return (kind, dst, i) in self.box

    def send(self, kind, dst, i, payload):
        self.box[(kind, dst, i)] = payload

    def recv(self, kind, dst, i):
        return self.box.pop((kind, dst, i))


class DistLink:
    """2 プロセス (torch.distributed、NCCL の P2P) 用 (未検証)。形を先に送ってから中身を送る。
    1 回で送るのは h [1, T, 4, 5120] bf16 (4096 トークンで 168MB) と pre_mix [1, T, 4] fp32。
    送信は isend にする。1F1B では stage0 の F(i+1) の送信と stage1 の B(i) の送信が同時に起きるので、
    blocking の send だと互いに相手の recv を待って止まる。"""

    def __init__(self, rank, device):
        import torch.distributed as dist
        self.dist, self.rank, self.dev = dist, rank, device
        self.pending = []

    def has(self, kind, dst, i):
        return True  # 受信は blocking

    def send(self, kind, dst, i, payload):
        h, pm = payload
        shp = torch.tensor(list(h.shape) + list(pm.shape), dtype=torch.long, device=self.dev)
        bufs = (shp, h.contiguous(), pm.contiguous().float())
        self.pending.append((bufs, [self.dist.isend(b, dst) for b in bufs]))

    def recv(self, kind, dst, i):
        src = dst - 1 if kind == "act" else dst + 1
        shp = torch.empty(7, dtype=torch.long, device=self.dev)
        self.dist.recv(shp, src)
        h = torch.empty(tuple(shp[:4].tolist()), dtype=torch.bfloat16, device=self.dev)
        pm = torch.empty(tuple(shp[4:].tolist()), dtype=torch.float32, device=self.dev)
        self.dist.recv(h, src)
        self.dist.recv(pm, src)
        return h, pm

    def flush(self):
        for _, works in self.pending:
            for w in works:
                w.wait()
        self.pending.clear()


def run_pipeline_local(runners, mbs):
    """全ステージの 1F1B 順序を、依存関係を守って 1 スレッドで交互に進める (CPU テスト用)。"""
    queues = [schedule_1f1b(len(mbs), len(runners), r.sid) for r in runners]
    order = []
    while any(queues):
        progressed = False
        for r, q in zip(runners, queues):
            if q and r.ready(q[0]):
                op = q.pop(0)
                r.run(op, mbs)
                order.append((r.sid,) + op)
                progressed = True
        if not progressed:
            raise RuntimeError("pipeline deadlock")
    return order


# =========================================================================== 12. safetensors (pure python) とパックの書き出し
_ST_NP = {"F16": np.float16, "F32": np.float32, "I16": np.int16, "I32": np.int32, "I64": np.int64,
          "U8": np.uint8, "I8": np.int8, "U16": np.uint16, "U32": np.uint32}


class SafetensorsIndex:
    """model.safetensors.index.json と各 shard のヘッダから name -> (file, offset, nbytes, dtype, shape)。
    同じ名前が複数の shard にあるとき (qad_l2.py apply --mode overlay のパック) は weight_map の指す方を使う。
    weight_map に無い名前はファイル名順で後勝ち (stream_ref.TensorIndex と同じ、stream_ref.py:50-62)。"""

    def __init__(self, d: str):
        self.dir = d
        ip = os.path.join(d, "model.safetensors.index.json")
        wm = json.load(open(ip))["weight_map"] if os.path.exists(ip) else {}
        files = sorted(set(wm.values())) if wm else sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
        self.meta = {}
        for fn in files:
            hdr, base = read_st_header(os.path.join(d, fn))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                if k in wm and wm[k] != fn:
                    continue
                a, b = v["data_offsets"]
                self.meta[k] = (fn, base + a, b - a, v["dtype"], tuple(v["shape"]))

    def shape(self, name):
        return self.meta[name][4]

    def read_numpy(self, name):
        fn, off, n, dt, shape = self.meta[name]
        with open(os.path.join(self.dir, fn), "rb") as f:
            f.seek(off)
            buf = f.read(n)
        return np.frombuffer(buf, dtype=_ST_NP[dt]).reshape(shape).copy()


def read_st_header(path):
    with open(path, "rb") as f:
        hl = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(hl))
    return hdr, 8 + hl


def write_safetensors(path, tensors: dict, metadata=None):
    """numpy 配列の dict を safetensors で書く (テストと小さな shard 用)。"""
    inv = {v: k for k, v in _ST_NP.items()}
    hdr, off, blobs = {}, 0, []
    for k, a in tensors.items():
        a = np.ascontiguousarray(a)
        b = a.tobytes()
        hdr[k] = {"dtype": inv[a.dtype.type], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    if metadata:
        hdr["__metadata__"] = metadata
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(len(hj).to_bytes(8, "little"))
        f.write(hj)
        for b in blobs:
            f.write(b)


def write_scales_pack(src: str, dst: str, new: dict, dry_run=False):
    """src パックを dst に組み立て直す。new = {テンソル名: fp16 配列}。
    - new に含まれるテンソルを持つ shard だけ新しいファイル (別 inode) にコピーし、該当範囲を上書きする。
    - それ以外のファイルは hardlink にする (Engram の 2 x 98GB を含む)。
    - src の shard は v1 の作業ディレクトリと hardlink を共有しているので、絶対に上書きしない。
    - dtype と shape は src と同じでなければならない (suh/svh は F16 のまま)。"""
    os.makedirs(dst, exist_ok=True)
    idx = SafetensorsIndex(src)
    by_file = {}
    for k, a in new.items():
        if k not in idx.meta:
            raise KeyError(k)
        fn, off, n, dt, shape = idx.meta[k]
        a = np.ascontiguousarray(np.asarray(a))
        if _ST_NP[dt] != a.dtype.type or tuple(a.shape) != shape or a.nbytes != n:
            raise ValueError(f"{k}: dtype/shape 不一致 ({dt}{shape} vs {a.dtype}{a.shape})")
        by_file.setdefault(fn, []).append((off, a.tobytes()))
    report = {"rewritten": [], "linked": 0}
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            s, d = os.path.join(root, f), os.path.join(dst, rel, f)
            relf = os.path.normpath(os.path.join(rel, f))
            if os.path.lexists(d):
                os.unlink(d)
            if relf in by_file:
                report["rewritten"].append(relf)
                if dry_run:
                    continue
                shutil.copyfile(s, d)
                if os.stat(s).st_ino == os.stat(d).st_ino:
                    raise RuntimeError("コピー先が元と同じ inode")
                with open(d, "r+b") as fh:
                    for off, b in by_file[relf]:
                        fh.seek(off)
                        fh.write(b)
                    fh.flush()
                    os.fsync(fh.fileno())
                with open(d, "rb") as fh:  # 読み戻し確認
                    for off, b in by_file[relf]:
                        fh.seek(off)
                        if fh.read(len(b)) != b:
                            raise IOError(f"{relf}: 読み戻しが一致しない")
            else:
                report["linked"] += 1
                if not dry_run:
                    os.link(s, d)
    return report


# =========================================================================== 13. 見積もり
@dataclass
class Scenario:
    name: str
    eff_tflops: float      # bf16 の実効 (小さな GEMM と torch の演算を含む)
    recon_us: float        # ext.reconstruct 1 tensor (5120x2304) あたり
    ovh_fwd_ms: float      # bucket 1 回の forward の起動と Python の費用
    ovh_bwd_ms: float      # 同 backward (局所 recompute + autograd)
    bw_tbs: float          # sparse attn の gather の実効帯域


SCENARIOS = [Scenario("楽観", 150, 15, 0.15, 0.6, 1.5),
             Scenario("中央", 80, 30, 0.3, 1.2, 1.2),
             Scenario("悲観", 40, 60, 0.6, 2.5, 0.8)]


def flops_per_token_layer(c=None):
    c = c or {}
    dim, ql, H, hd, G, orr = 5120, 1280, 64, 512, 8, 1024
    k_attn = 128 + 512  # window + 圧縮 top-k
    dense = 2 * (dim * ql + ql * H * hd + dim * hd + G * (H * hd // G) * orr + G * orr * dim)
    core = 2 * 2 * H * k_attn * hd
    moe = 2 * TOPK * 3 * DIM * INTER
    shared = 2 * 3 * DIM * INTER
    gate = 2 * DIM * N_EXPERTS
    hc = 2 * 2 * 24 * 4 * DIM
    fwd = dense + core + moe + shared + gate + hc
    bwd = (dense + core) + (dense + 2 * core) + 2 * moe + 2 * shared + 2 * gate + 3 * hc
    return {"fwd": fwd, "bwd_phase": bwd, "dense_attn": dense, "attn_core": core, "moe": moe}


def estimate(T=4096, layers_per_stage=20, bucket=16, accum=4, inflight=(2, 1)):
    f = flops_per_token_layer()
    head = 2 * 129280 * DIM
    engram = 2 * (3 * 8 * 256) * DIM * 5
    n_tensors = layers_per_stage * N_EXPERTS * 3
    n_buckets = layers_per_stage * math.ceil(N_EXPERTS / bucket)
    gather = T * 640 * 512 * 4  # 1 層 1 回の gather (fp32)
    rows = []
    for sc in SCENARIOS:
        st = []
        for s in range(2):
            ffw = layers_per_stage * f["fwd"] * T + (engram * 2 * T if s == 0 else head * T)
            fbw = layers_per_stage * f["bwd_phase"] * T + (engram * 2 * T if s == 0 else 2 * head * T)
            tf = ffw / (sc.eff_tflops * 1e12) + n_tensors * sc.recon_us * 1e-6 + n_buckets * sc.ovh_fwd_ms * 1e-3 \
                + layers_per_stage * (2 * gather) / (sc.bw_tbs * 1e12) + 0.02
            tb = fbw / (sc.eff_tflops * 1e12) + n_tensors * sc.recon_us * 1e-6 + n_buckets * sc.ovh_bwd_ms * 1e-3 \
                + layers_per_stage * (6 * gather) / (sc.bw_tbs * 1e12) + 0.06
            st.append((tf, tb))
        p2p = 2 * (T * 4 * DIM * 2) / 20e9
        latency = st[0][0] + st[1][0] + st[1][1] + st[0][1] + p2p
        per_mb = max(a + b for a, b in st) * (accum + 1) / accum + p2p
        rows.append({"scenario": sc.name, "stage0_fwd": st[0][0], "stage0_bwd": st[0][1], "stage1_fwd": st[1][0],
                     "stage1_bwd": st[1][1], "step_latency_s": latency, "pipelined_s_per_4k": per_mb,
                     "tok_per_s": T / per_mb,
                     "hours_per_Mtok_epoch": 1e6 / (T / per_mb) / 3600})
    GiB = 2 ** 30
    trellis = layers_per_stage * N_EXPERTS * 3 * (320 * 144 * 32 * 2)
    scal = layers_per_stage * N_EXPERTS * 22272 * 4
    act_per_layer = T * 4 * DIM * 2 + T * DIM * 4 + T * 512 * 4
    nonexp = {0: 3.093e9 + 2 * 0.672e9 + 1.324e9, 1: 2.763e9 + 2 * 0.672e9 + 2.648e9}
    mem = {}
    for s in range(2):
        work = 5.0 * GiB if s == 0 else 6.5 * GiB
        m = {"trellis (常駐)": trellis, "base su/sv fp32": scal, "theta fp32": scal, "grad fp32": scal,
             "Adam m,v": 2 * scal, "非 expert 重み": nonexp[s],
             f"保存 activation x{inflight[s]} 系列": inflight[s] * layers_per_stage * act_per_layer,
             "作業領域 (1 層の recompute + backward)": work, "CUDA context と断片化": 2.5 * GiB}
        m["合計"] = sum(m.values())
        mem[s] = {k: v / GiB for k, v in m.items()}
    return {"flops": f, "time": rows, "mem_gib": mem, "gpu_gib": 95.59,
            "dense_Q_all_GB": N_LAYERS * N_EXPERTS * 3 * DIM * INTER * 2 / 1e9}


def print_estimate():
    e = estimate()
    f = e["flops"]
    print(f"FLOP/トークン/層: fwd {f['fwd'] / 1e9:.3f} G、backward 側 (recompute 込み) {f['bwd_phase'] / 1e9:.3f} G")
    print(f"Q を全部 bf16 で持つと {e['dense_Q_all_GB']:.1f} GB (不可)")
    print("| 場合 | s0 fwd | s0 bwd | s1 fwd | s1 bwd | 1 step 単独 (s) | 1F1B 実効 s/4096 tok | tok/s | 1M tok/epoch (h) | x4 epoch (h) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in e["time"]:
        print(f"| {r['scenario']} | {r['stage0_fwd']:.2f} | {r['stage0_bwd']:.2f} | {r['stage1_fwd']:.2f} | {r['stage1_bwd']:.2f} | "
              f"{r['step_latency_s']:.1f} | {r['pipelined_s_per_4k']:.1f} | {r['tok_per_s']:.0f} | "
              f"{r['hours_per_Mtok_epoch']:.2f} | {4 * r['hours_per_Mtok_epoch']:.1f} |")
    for s in (0, 1):
        print(f"GPU{s} (GiB):", ", ".join(f"{k} {v:.2f}" for k, v in e["mem_gib"][s].items()))
    return e


# =========================================================================== 14. GPU 経路 (未検証)
def valid_splits(cfg: dict):
    """2 ステージの境目 s (層 0..s-1 / s..n-1) のうち、層をまたぐ状態が境目を越えないものを返す。
    公式の SharedAttentionRuntime (ref/model.py:1169-1183) では、圧縮 KV と index_k は直前の kv_source、
    top-k 位置は直前の index_source、候補は candidate_source から後続の層へ渡る。後段の層の供給元が前段にあると
    ステージ間で送る必要が生じるので不可とする。実 config では [1, 2, 8, 14, 20]。"""
    n = cfg["n_layers"]
    ratios = cfg["compress_ratios"]
    kvs, ixs = sorted(cfg["kv_source_layers"]), sorted(cfg["index_source_layers"])
    cand = cfg.get("candidate_source_layer", -1)
    out = []
    for s in range(1, n):
        good = True
        for l in range(s, n):
            if not ratios[l]:
                continue
            pk = max([k for k in kvs if k <= l], default=None)
            pi = max([k for k in ixs if k <= l], default=None)
            if (pk is not None and pk < s) or (l not in ixs and pi is not None and pi < s):
                good = False
            if l in ixs and 0 <= cand < l and cand < s:
                good = False
        if good:
            out.append(s)
    return out


def build_stage_real(a, rank: int, n_stages: int = 2):
    """公式 model.py を CPU 上に (テンソルは未初期化のまま) 組み、担当層だけ重みを読んで GPU に移す。
    重みの読み方は stream_ref.build (stream_ref.py:358-397) と同じ (wo_a は FP8 を bf16 に戻す)。"""
    sys.path.insert(0, a.ref)
    sys.path.insert(0, os.path.join(a.ckpt, "encoding"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import kernel as K  # noqa
    import model as M  # noqa
    import stream_ref as SR  # 既存ファイル。import するだけで変更しない
    from transformers import AutoTokenizer

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    torch.set_default_dtype(torch.bfloat16)
    cfg = json.load(open(os.path.join(a.ckpt, "inference", "config.json")))
    cfg.update({"max_batch_size": 1, "max_seq_len": a.max_seq_len, "temperature": 0.0})
    args = M.ModelArgs(**cfg)
    real_expert = M.Expert
    M.Expert = lambda dim, inter, dtype=None, swiglu_limit=0.0: None if dtype == SR.FP4X2 else real_expert(dim, inter, dtype=dtype, swiglu_limit=swiglu_limit)
    M.ParallelEngramEmbedding = SR.HostEngramEmbedding
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    with torch.device("cpu"):
        model = M.Transformer(args, tok)
    M.Expert = real_expert
    split = getattr(a, "split", args.n_layers // 2)
    if n_stages != 2 or split not in valid_splits(cfg):
        raise ValueError(f"split {split} は使えない (候補 {valid_splits(cfg)})")
    owned = list(range(0, split)) if rank == 0 else list(range(split, args.n_layers))
    for i in range(args.n_layers):
        if i not in owned:
            model.layers[i] = torch.nn.Identity()
    model.mtp = torch.nn.ModuleList()
    model.vision = None
    first, last = rank == 0, rank == n_stages - 1
    if not first:
        model.embed, model.engram_hash = None, None
    if not last:
        model.head, model.norm = None, None
    tindex = SR.TensorIndex(a.ckpt)
    sd = model.state_dict()
    with torch.no_grad():
        for key, param in sd.items():
            if key not in tindex.meta:
                continue
            src = tindex.load(key)
            if key.endswith("wo_a.weight"):
                sc = tindex.load(key.replace("weight", "scale"))
                ob, ib = src.size(0) // sc.size(0), src.size(1) // sc.size(1)
                src = (src.unflatten(0, (-1, ob)).unflatten(-1, (-1, ib)).float()
                       * sc[:, None, :, None].float()).flatten(2, 3).flatten(0, 1)
            if src.dtype == torch.int8:
                src = src.view(SR.FP4X2)
            if param.dtype == src.dtype:
                param.copy_(src.view(param.shape))
            else:
                param.copy_(src.to(torch.float32).to(param.dtype).view(param.shape))
    model.to(dev)
    model.requires_grad_(False)
    for l in owned:
        blk = model.layers[l]
        blk.ffn.experts = None
        if blk.engram is not None:
            blk.engram.embed.bind(tindex, l)
    source = Exl3ResidentSource(a.pack, owned, dev)
    scales = ScaleParams({l: source.base_scales(l) for l in owned}, dev)
    ks = KernelSet("official", K=K, M=M)
    cfgl = L3Config(bucket=a.bucket, attn_chunk=a.attn_chunk)
    return StudentStage(M, model, owned, first, last, scales, source, ks, cfgl), tok


def clip_grad_global(params, max_norm, dist=None):
    """2 ステージにまたがる全体の勾配ノルムで clip する (ステージごとの clip とは結果が違う)。"""
    params = [p for p in params if p.grad is not None]
    sq = torch.zeros((), dtype=torch.float32, device=params[0].grad.device if params else "cpu")
    for p in params:
        sq += p.grad.float().square().sum()
    if dist is not None:
        dist.all_reduce(sq)
    norm = float(sq.sqrt())
    if norm > max_norm:
        for p in params:
            p.grad.mul_(max_norm / (norm + 1e-6))
    return norm


@torch.no_grad()
def forward_only(stage: StudentStage, sid: int, n_stages: int, link, mbs):
    """評価用。ステージを順に通し、最後のステージで位置ごとの KL (tail なし、harness と同じ近似) と top-1 一致を返す。"""
    out = []
    for i, mb0 in enumerate(mbs):
        mb = copy.copy(mb0).new_pass(False)
        if sid == 0:
            r = stage(mb)
        else:
            h, pm = link.recv("act", sid, i)
            r = stage(mb, h, pm)
        if sid == n_stages - 1:
            _, _, klt, ag = r
            out.append({"i": i, "domain": mb.meta.get("domain"), "kl": klt.cpu(), "agree": ag.cpu(), "w": mb.w.cpu(),
                        "ja": mb.meta.get("ja")})
        else:
            link.send("act", sid + 1, i, (r[0], r[1]))
            # DistLink の isend はバッファ (h は 1 系列 168MB) を完了まで持つ。検証の全系列を溜めると
            # GPU0 の余裕 (約 6.6GiB) を超えうるので、2 本を超えたら待つ (受信側は順に recv するので止まらない)
            if hasattr(link, "flush") and len(getattr(link, "pending", ())) >= 2:
                link.flush()
    if hasattr(link, "flush"):
        link.flush()
    return out


def summarize(evals):
    """区分ごとの KL 平均と top-1 一致 (QUALITY.md と同じ指標)。

    区分 (domain) は教師ファイル名の先頭なので、calib256k のように 1 ファイルに日本語と他が混ざると
    「どの区分も 2% 以上悪化させない」の検査が日本語に効かない。そこで位置ごとの日本語判定 (r["ja"]) があれば
    "pos:ja" / "pos:other" も足す。重み付き KL (wkl) には domain の項だけを使い、"pos:" の項は検査専用。"""
    by = {}
    for r in evals:
        d = by.setdefault(r["domain"] or "all", {"kl": [], "ag": []})
        d["kl"].append(r["kl"])
        d["ag"].append(r["agree"])
    for r in evals:
        ja = r.get("ja")
        if ja is None:
            continue
        ja = ja.bool().view(-1)
        for key, m in (("pos:ja", ja), ("pos:other", ~ja)):
            if m.any():
                d = by.setdefault(key, {"kl": [], "ag": []})
                d["kl"].append(r["kl"][m])
                d["ag"].append(r["agree"][m])
    return {k: {"kl": float(torch.cat(v["kl"]).mean()), "top1": float(torch.cat(v["ag"]).mean()),
                "n": int(sum(t.numel() for t in v["kl"]))} for k, v in by.items()}


def train_main(a):
    """2 プロセス (torchrun) で 1F1B を回す骨格 (未検証)。設計書 §9 の G1〜G6 で確かめてから使う。
    - 教師: --teacher に stream_ref.py --teacher-out の .pt を複数渡せる (区分名=ファイル名の先頭)。
    - 検証: 学習データの val_frac を取り分け、epoch の 1/eval_every ごとに区分別の KL を測る。
      早期終了は「どの区分も最良値から 2% 以上悪化していない中で、重み付き KL が最小」の時点を残す。"""
    import random
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank, S = dist.get_rank(), dist.get_world_size()
    stage, tok = build_stage_real(a, rank, S)
    dev = torch.device(f"cuda:{rank}")
    # ja_weight = 1 でも位置ごとの日本語判定 (summarize の pos:ja による悪化検査) のために decode する
    decode = lambda ids: tok.decode(ids)  # noqa: E731
    mbs_all = []
    for path in a.teacher:
        dom = os.path.basename(path).split("_")[0]
        mbs_all += [microbatch_from_record(r, decode, a.ja_weight, domain=dom) for r in load_teacher(path)]
    rng = random.Random(a.seed)
    order = list(range(len(mbs_all)))
    rng.shuffle(order)
    n_val = max(1, int(len(order) * a.val_frac))
    val, trn = order[:n_val], order[n_val:]
    params = list(stage.scales.parameters())
    sc = stage.scales
    ident = {"pack": os.path.abspath(a.pack),
             "base_sha256": base_digests(lambda: [(f"{l}_{k}", getattr(sc, f"base_{l}_{k}"))
                                                  for l in sc.layers for k in range(len(SVEC))])}
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.98), eps=1e-8)
    link = DistLink(rank, dev)
    runner = StageRunner(stage, rank, S, link, ckpt=True, accum=a.accum)
    n_groups = math.ceil(len(trn) / a.accum)
    steps = n_groups * a.epochs
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, (st + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(st, steps) / steps)))
    eval_at = set(int(round(n_groups * k / a.eval_every)) for k in range(1, a.eval_every + 1))
    best, best_by, step, t0 = None, {}, 0, time.time()
    for ep in range(a.epochs):
        rng.shuffle(trn)
        for gi, g0 in enumerate(range(0, len(trn), a.accum)):
            grp = [mbs_all[i].to(dev) for i in trn[g0:g0 + a.accum]]
            for op in schedule_1f1b(len(grp), S, rank):
                runner.run(op, grp)
            link.flush()
            (a.reg * stage.scales.reg()).backward()
            gn = clip_grad_global(params, a.clip, dist)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if rank == S - 1:
                m = runner.metrics[-len(grp):]
                print(json.dumps({"step": step, "ep": ep, "kl": sum(x["kl_top"] for x in m) / len(m),
                                  "agree": sum(x["agree"] for x in m) / len(m), "gnorm": gn,
                                  "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)}), flush=True)
            if (gi + 1) in eval_at:
                ev = forward_only(stage, rank, S, link, [mbs_all[i].to(dev) for i in val])
                summ = summarize(ev) if rank == S - 1 else None
                obj = [summ]
                dist.broadcast_object_list(obj, src=S - 1)
                summ = obj[0]
                dom_v = [v for k, v in summ.items() if not k.startswith("pos:")]  # pos:* は検査専用 (二重計上しない)
                wkl = sum(v["kl"] * v["n"] for v in dom_v) / sum(v["n"] for v in dom_v)
                ok_dom = all(v["kl"] <= best_by.get(k, v["kl"]) * 1.02 for k, v in summ.items())
                for k, v in summ.items():
                    best_by[k] = min(best_by.get(k, v["kl"]), v["kl"])
                tag = f"ep{ep}_g{gi + 1}"
                torch.save({"theta": stage.scales.state_dict(), "step": step, "val": summ, "tag": tag, **ident},
                           os.path.join(a.out, f"theta_r{rank}_last.pt"))  # 0.68GB/rank。毎回は残さない
                if ok_dom and (best is None or wkl < best[0]):
                    best = (wkl, tag)
                    torch.save({"theta": stage.scales.state_dict(), "step": step, "val": summ, "tag": tag, **ident},
                               os.path.join(a.out, f"theta_r{rank}_best.pt"))
                if rank == S - 1:
                    print(json.dumps({"eval": tag, "val": summ, "weighted_kl": wkl, "best": best}, ensure_ascii=False), flush=True)
    dist.destroy_process_group()


def export_main(a):
    """CPU で動く。--theta (ランクごとの theta_r*_best.pt) と --pack の base suh/svh から、
    新しい suh/svh (fp16) を作ってパックを書き出す (write_scales_pack)。"""
    idx = SafetensorsIndex(a.pack)
    new, stats = {}, {"n": 0, "max_abs_theta": 0.0}
    stats["base_check"] = []
    for tp in a.theta:
        ck = torch.load(tp, map_location="cpu", weights_only=False)
        sd = ck["theta"]
        want = ck.get("base_sha256")
        got = {}
        for key, th in sd.items():
            if not key.startswith("theta."):
                continue
            l, k = map(int, key.split(".", 1)[1].split("_"))
            p, kind = SVEC[k]
            stats["max_abs_theta"] = max(stats["max_abs_theta"], float(th.abs().max()))
            bases = []
            for e in range(th.shape[0]):
                nm = f"layers.{l}.ffn.experts.{e}.{p}.{kind}"
                b16 = torch.from_numpy(idx.read_numpy(nm))
                bases.append(b16)
                b = b16.float()
                v = b * torch.exp(th[e].float())
                if not torch.isfinite(v).all() or v.abs().max() > 65504:
                    raise ValueError(f"{nm}: fp16 に入らない")
                new[nm] = v.half().numpy()
                stats["n"] += 1
            got[f"{l}_{k}"] = torch.stack(bases)
        if want is None:
            print(f"[export] 注意: {tp} に base_sha256 が無い (古い checkpoint)。base の照合をしない", flush=True)
            stats["base_check"].append("skipped")
            continue
        have = base_digests(lambda: list(got.items()))
        bad = sorted(k for k in have if want.get(k) != have[k])
        if bad:
            msg = (f"{tp}: --pack の suh/svh が学習時の pack ({ck.get('pack', '?')}) と違う (例 {bad[:3]})。"
                   f"theta は学習時の base に対する相対値なので、別の pack には当てられない (--force-base で無視)")
            if not getattr(a, "force_base", False):
                raise ValueError(msg)
            print("[export] 警告: " + msg, flush=True)
        stats["base_check"].append("ok" if not bad else "forced")
    rep = write_scales_pack(a.pack, a.out, new, dry_run=a.dry_run)
    rep.update(stats)
    with open(a.out.rstrip("/") + ".l3_manifest.json", "w") as f:
        json.dump({"pack": a.pack, "theta": a.theta, **rep}, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rewritten"} | {"n_rewritten": len(rep["rewritten"])}))
    return rep


# =========================================================================== 15. CPU テスト
def _tiny_args(M):
    return M.ModelArgs(max_batch_size=1, max_seq_len=64, dtype="bf16", expert_dtype=None, vocab_size=512,
                       dim=256, moe_inter_dim=128, n_layers=6, n_heads=4, n_routed_experts=8, n_activated_experts=2,
                       route_scale=1.5, swiglu_limit=10.0, q_lora_rank=64, head_dim=64, rope_head_dim=16,
                       norm_eps=1e-6, o_groups=2, o_lora_rank=32, window_size=8,
                       compress_ratios=(0, 2, 2, 2, 1, 1), kv_source_layers=(1, 4), index_source_layers=(1, 3, 4, 5),
                       index_n_heads=2, index_head_dim=32, index_topk=4, candidate_source_layer=4,
                       candidate_topk_blocks=2, candidate_block_size=4, hc_mult=4, hc_sinkhorn_iters=20,
                       dspark_block_size=0, n_mtp_layers=0)


def _patch_official(M):
    """公式 model.py の tilelang 呼び出しを CPU のエミュレーションに差し替える (テスト専用)。"""
    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        assert inplace
        x.copy_(qdq_fp8_emul(x, block_size, pow2=scale_fmt is not None))
        return x

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        assert inplace
        x.copy_(qdq_fp4_emul(x, block_size, e4m3=scale_dtype == torch.float8_e4m3fn))
        return x

    M.act_quant = act_quant
    M.fp4_act_quant = fp4_act_quant
    M.hc_split_sinkhorn = lambda mixes, s, b, hc=4, it=20, eps=1e-6: hc_split_sinkhorn_torch(mixes, s, b, hc, it, eps)
    M.sparse_attn = lambda q, kv, sink, idx, scale: sparse_attn_ref(q, kv, sink, idx, scale, round_p=False)


def _build_tiny(M, seed=0, dtype=torch.float32):
    """dtype=bf16 は GPU と同じ形 (Linear は bf16、hc / attn_sink / compressor / head は fp32)。
    乱数は常に fp32 で作って copy_ するので、fp32 版と bf16 版は同じ重みを丸めたものになる。"""
    torch.manual_seed(seed)
    args = _tiny_args(M)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = M.Transformer(args, None)
    finally:
        torch.set_default_dtype(prev)
    if dtype == torch.float32:
        model = model.float()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            r = torch.randn(p.shape, generator=g, dtype=torch.float32)
            if n.endswith("norm.weight"):
                p.copy_(1 + 0.1 * r)
            elif "attn_sink" in n:
                p.copy_(0.5 * r)
            elif "hc_" in n and "scale" in n:
                p.copy_(1 + 0.1 * r)
            elif "hc_" in n or "gate.bias" in n:
                p.copy_(0.1 * r)
            elif p.dim() >= 2:
                p.copy_(r / math.sqrt(p.shape[-1]))
            else:
                p.copy_(0.1 * r)
        model.embed.weight.mul_(math.sqrt(model.embed.weight.shape[-1]))
    model.requires_grad_(False)
    return model, args


def _tiny_source(model, args, seed=1):
    """公式 tiny model の expert 重みを、ランダムな base の su/sv と内側 Q に分解する。"""
    g = torch.Generator().manual_seed(seed)
    mats, base = {}, {}
    E, D, I = args.n_routed_experts, args.dim, args.moe_inter_dim
    for l in range(args.n_layers):
        vecs = []
        for p, kind in SVEC:
            n = (D if kind == "suh" else I) if p in ("w1", "w3") else (I if kind == "suh" else D)
            sign = torch.randint(0, 2, (E, n), generator=g).float() * 2 - 1
            vecs.append(sign * (0.5 + torch.rand(E, n, generator=g)))
        base[l] = vecs
        for e in range(E):
            ex = model.layers[l].ffn.experts[e]
            for p in PARTS:
                mats[(l, e, p)] = getattr(ex, p).weight.t().contiguous().double()
    return DenseInnerSource(mats, base, torch.float32)


def _official_logits(M, model, ids):
    """Transformer.forward (ref/model.py:1245-1275) の層ループを公式 Block でたどり、全位置の logits を返す。"""
    with torch.inference_mode():
        h = model.embed(ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        pre_mix = M.make_identity_pre_mix(h, model.hc_mult)
        for layer in model.layers:
            h, pre_mix = layer(h, 0, pre_mix, None)
        h = layer.hc_pre(h, pre_mix)
        return model.head(model.norm(h), full_logits=True)[0].float()


def _tiny_mb(logits_teacher, ids, k=8, sf=1, w=None):
    lp = torch.log_softmax(logits_teacher[sf - 1: ids.shape[1] - 1], -1)
    v, i = lp.topk(k, -1)
    n = v.shape[0]
    return Microbatch(ids=ids, pos=torch.arange(sf - 1, ids.shape[1] - 1), top_v=v.half(), top_i=i.int(),
                      w=torch.ones(n) if w is None else w)


def selftest(ref_dir: str, teacher_dir: str | None):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)
    res = {}

    def ok(name, cond, info=""):
        res[name] = bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)

    # T1 Hadamard
    H = hadamard(128, "cpu", torch.float64)
    ok("T1 hadamard 直交・対称", torch.allclose(H @ H.t(), torch.eye(128, dtype=torch.float64), atol=1e-12) and torch.equal(H, H.t()))
    try:
        from exllamav3.util.hadamard import get_hadamard
        ok("T1b exllamav3 get_hadamard(128) と一致", torch.equal(get_hadamard(128).double(), _sylvester(128)))
    except Exception as ex:  # noqa
        print("  (exllamav3 なし: T1b を省略)", type(ex).__name__)

    # T2 scale-only の投影 = get_weight_tensor の式
    k_in, n_out = 256, 384
    Q = torch.randn(k_in, n_out, dtype=torch.float64)
    su = torch.randn(k_in, dtype=torch.float64)
    sv = torch.randn(n_out, dtype=torch.float64)
    x = torch.randn(7, k_in, dtype=torch.float64)
    W = dense_from_inner(Q, su, sv)
    try:
        from exllamav3.modules.quant.exl3_lib.quantize import preapply_had_l, preapply_had_r
        w2 = preapply_had_l(Q.clone(), 128) * su[:, None]
        w2 = preapply_had_r(w2, 128) * sv[None, :]
        # preapply_had_l/r は内部で fp32 に落とす (quantize.py:401-418) ので fp32 の精度で比べる
        ok("T2a dense_from_inner = exllamav3 get_weight_tensor の手順", float((W - w2).abs().max()) < 1e-6 * float(W.abs().max()) * 10,
           f"maxdiff {float((W - w2).abs().max()):.2e}")
    except Exception as ex:  # noqa
        print("  (exllamav3 なし: T2a を省略)", type(ex).__name__)
    y1 = scaled_proj(x, su, Q, sv, torch.float64)
    ok("T2b scaled_proj = x @ W", torch.allclose(y1.double(), x @ W, atol=1e-8), f"maxdiff {float((y1.double() - x @ W).abs().max()):.2e}")
    ok("T2c inner_from_dense の往復", torch.allclose(inner_from_dense(W, su, sv), Q, atol=1e-9))

    # T3 MoEScaleOnlyFn: dense 参照との一致と gradcheck
    D, I, E, kk, T = 128, 128, 4, 2, 6
    mats, base = {}, {0: []}
    g = torch.Generator().manual_seed(3)
    for p, kind in SVEC:
        n = (D if kind == "suh" else I) if p in ("w1", "w3") else (I if kind == "suh" else D)
        base[0].append((torch.randint(0, 2, (E, n), generator=g).double() * 2 - 1) * (0.5 + torch.rand(E, n, generator=g, dtype=torch.float64)))
    for e in range(E):
        for p in PARTS:
            shp = (D, I) if p != "w2" else (I, D)
            mats[(0, e, p)] = torch.randn(*shp, generator=g, dtype=torch.float64) * (3.0 / math.sqrt(shp[0]))
    src = DenseInnerSource(mats, base, torch.float64)
    xs = torch.randn(T, D, generator=g, dtype=torch.float64) * 2
    logits_r = torch.randn(T, E, generator=g, dtype=torch.float64)
    ind = logits_r.topk(kk, -1).indices
    wts = torch.rand(T, kk, generator=g, dtype=torch.float64) + 0.2
    S = [b.clone() for b in base[0]]

    def dense_ref(x, w, S):
        y = torch.zeros_like(x)
        for t in range(T):
            for j in range(kk):
                e = int(ind[t, j])
                W1 = dense_from_inner(src.Q[(0, e, "w1")], S[0][e], S[1][e])
                W3 = dense_from_inner(src.Q[(0, e, "w3")], S[2][e], S[3][e])
                W2 = dense_from_inner(src.Q[(0, e, "w2")], S[4][e], S[5][e])
                gt, up = x[t] @ W1, x[t] @ W3
                up, gt = up.clamp(-10, 10), gt.clamp(max=10)
                y[t] = y[t] + (w[t, j] * F.silu(gt) * up) @ W2
        return y

    for bucket in (1, 3):
        call = MoECall(0, src, 10.0, torch.float64, bucket=bucket, round_out=False)
        y = MoEScaleOnlyFn.apply(xs, wts, ind, *S, call)
        yr = dense_ref(xs, wts, S)
        ok(f"T3a MoE forward = dense 参照 (bucket={bucket})", torch.allclose(y.double(), yr, atol=1e-8, rtol=1e-8),
           f"maxdiff {float((y.double() - yr).abs().max()):.2e}")
    # clamp が実際に効いているか
    with torch.no_grad():
        e0 = int(ind[0, 0])
        gt = xs @ dense_from_inner(src.Q[(0, e0, "w1")], S[0][e0], S[1][e0])
        ut = xs @ dense_from_inner(src.Q[(0, e0, "w3")], S[2][e0], S[3][e0])
    print(f"  clamp 発動: gate>10 {int((gt > 10).sum())} / up |>10| {int((ut.abs() > 10).sum())} (/{gt.numel()})")
    xs_ = xs.clone().requires_grad_(True)
    wts_ = wts.clone().requires_grad_(True)
    S_ = [s.clone().requires_grad_(True) for s in S]
    for bucket in (1, 3):
        call = MoECall(0, src, 10.0, torch.float64, bucket=bucket, round_out=False)
        okg = torch.autograd.gradcheck(lambda x, w, *s: MoEScaleOnlyFn.apply(x, w, ind, *s, call), (xs_, wts_, *S_),
                                       eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True)
        ok(f"T3b MoE gradcheck (bucket={bucket})", okg)
    # checkpoint 用 cache: 2 回目の呼び出しは保存した出力を返す
    cache = {}
    call = MoECall(0, src, 10.0, torch.float64, bucket=2, round_out=False, cache=cache)
    y1 = MoEScaleOnlyFn.apply(xs, wts, ind, *S, call)
    y2 = MoEScaleOnlyFn.apply(xs, wts, ind, *S, call)
    ok("T3c MoE cache (recompute で再計算しない)", torch.equal(y1, y2) and not cache)

    # T4 量子化エミュレーション
    xq = torch.randn(4, 64) * 3
    q8 = qdq_fp8_emul(xq)
    ok("T4a fp8 qdq は冪等", torch.equal(qdq_fp8_emul(q8), q8))
    rel = ((q8 - xq).abs() / xq.abs().clamp_min(1e-3)).median()
    ok("T4b fp8 qdq の相対誤差 (中央値 < 4%)", rel < 0.04, f"{float(rel):.4f}")
    v = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.3, 5.1, 0.0])
    ok("T4c fp4 の偶数丸め", torch.equal(fp4_round(v), torch.tensor([0, 1, 1, 2, 2, 4, 4, -0.5, 6, 0.0])))
    q4 = qdq_fp4_emul(xq, 32)
    ok("T4d fp4 qdq は冪等", torch.equal(qdq_fp4_emul(q4, 32), q4))
    q4e = qdq_fp4_emul(xq, 16, e4m3=True)
    ok("T4e fp4 (e4m3 scale) qdq は有限", torch.isfinite(q4e).all())
    ok("T4f pow2_ceil", torch.equal(_pow2_ceil(torch.tensor([1.0, 1.5, 0.5, 3.0, 0.3])), torch.tensor([1.0, 2.0, 0.5, 4.0, 0.5])))
    ks = KernelSet("emul")
    xg = torch.randn(3, 64, requires_grad=True)
    ks.ste_act(xg).sum().backward()
    ok("T4g STE の勾配は恒等", torch.equal(xg.grad, torch.ones_like(xg)))

    # T5 sinkhorn
    mixes = torch.randn(2, 3, 24, dtype=torch.float64)
    scl = torch.rand(3, dtype=torch.float64) + 0.5
    bse = torch.randn(24, dtype=torch.float64) * 0.1
    pre, post, comb = hc_split_sinkhorn_torch(mixes, scl, bse)
    ok("T5a comb は列和 1 (二重確率の近似)", torch.allclose(comb.sum(-2), torch.ones(2, 3, 4, dtype=torch.float64), atol=1e-5))
    ok("T5b sinkhorn gradcheck", torch.autograd.gradcheck(lambda m: hc_split_sinkhorn_torch(m, scl, bse), (mixes.clone().requires_grad_(True),), fast_mode=True))

    # T6 sparse attention
    b_, s_, h_, d_, n_, k_ = 1, 6, 2, 8, 9, 4
    q = torch.randn(b_, s_, h_, d_, dtype=torch.float64)
    kvv = torch.randn(b_, n_, d_, dtype=torch.float64)
    sink = torch.randn(h_, dtype=torch.float32)
    idx = torch.randint(0, n_, (b_, s_, k_), dtype=torch.int32)
    idx[0, 0, 1:] = -1
    idx[0, 2, :] = -1
    o1 = SparseAttnFn.apply(q, kvv, sink, idx, 0.3, 4, False)
    o2 = sparse_attn_ref(q, kvv, sink, idx, 0.3, 4, False)
    ok("T6a SparseAttnFn forward = 参照", torch.allclose(o1, o2) and float(o1[0, 2].abs().max()) == 0.0)
    ok("T6b SparseAttnFn gradcheck", torch.autograd.gradcheck(
        lambda a, b: SparseAttnFn.apply(a, b, sink, idx, 0.3, 4, False), (q.clone().requires_grad_(True), kvv.clone().requires_grad_(True)), fast_mode=True))
    qb, kb = torch.randn(1, 40, 4, 64).bfloat16(), torch.randn(1, 50, 64).bfloat16()
    ib = torch.randint(-1, 50, (1, 40, 16), dtype=torch.int32)
    sk = torch.zeros(4)
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        harness_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream_ref.py")).read()
        ns = {"torch": torch}
        start = harness_src.index("def sparse_attn_torch")
        end = harness_src.index("# ----", start)
        exec(harness_src[start:end], ns)  # stream_ref.py の関数だけを取り出して比べる (ファイルは変更しない)
        oh = ns["sparse_attn_torch"](qb, kb, sk, ib, 0.125)
        om = sparse_attn_ref(qb, kb, sk, ib, 0.125)
        ok("T6c bf16 で stream_ref.sparse_attn_torch と一致", torch.equal(oh, om))
    except Exception as ex:  # noqa
        print("  (stream_ref.py を読めない: T6c 省略)", ex)

    # T7 FP8 Linear の STE
    Wf = torch.randn(64, 96) / 8
    s_blk = torch.zeros(2, 3)
    wq = torch.empty(64, 96, dtype=torch.float8_e4m3fn)
    for i in range(2):
        for j in range(3):
            blk = Wf[32 * i:32 * i + 32, 32 * j:32 * j + 32]
            s = _pow2_ceil(blk.abs().max().clamp_min(1e-4) / 448)
            s_blk[i, j] = s
            wq[32 * i:32 * i + 32, 32 * j:32 * j + 32] = (blk / s).to(torch.float8_e4m3fn)
    try:
        sc8 = s_blk.to(torch.float8_e8m0fnu)
    except Exception:  # noqa
        sc8 = s_blk
    mod = SimpleNamespace(weight=wq)
    mod.weight.scale = sc8
    xl = torch.randn(5, 96, requires_grad=True)
    yl = Fp8LinearSTE.apply(xl, mod, ks)
    wd = dequant_fp8_weight(wq, sc8)
    ok("T7a FP8 Linear forward = F.linear(qdq(x), W_deq)", torch.allclose(yl, F.linear(qdq_fp8_emul(xl.detach()), wd), atol=1e-5))
    gy = torch.randn_like(yl)
    yl.backward(gy)
    ok("T7b FP8 Linear backward = dY @ W_deq", torch.allclose(xl.grad, (gy.bfloat16() @ wd.bfloat16()).float(), atol=1e-2))

    # T8 KL 損失
    lg = torch.randn(5, 30, dtype=torch.float64)
    lt = torch.log_softmax(torch.randn(5, 30, dtype=torch.float64) * 2, -1)
    tv, ti = lt.topk(6, -1)
    kl, _ = kl_topk(lg, tv, ti, tail=True)
    kl0, _ = kl_topk(lt, tv, ti, tail=True)
    ok("T8a KL >= 0、学生 = 教師で 0", (kl >= -1e-9).all() and kl0.abs().max() < 1e-6, f"max|kl0| {float(kl0.abs().max()):.1e}")
    ok("T8b KL gradcheck", torch.autograd.gradcheck(lambda z: kl_topk(z, tv, ti, True)[0], (lg.clone().requires_grad_(True),), fast_mode=True))
    hw = torch.randn(30, 16, dtype=torch.float64)
    xh = torch.randn(9, 16, dtype=torch.float64, requires_grad=True)
    tv9, ti9 = torch.log_softmax(torch.randn(9, 30, dtype=torch.float64), -1).topk(6, -1)
    w9 = torch.rand(9, dtype=torch.float64)
    l1, *_ = head_kl_loss(xh, hw, tv9, ti9, w9, chunk=4, ckpt=True)
    g1, = torch.autograd.grad(l1, xh)
    l2 = (kl_topk(xh @ hw.t(), tv9, ti9, True)[0] * w9).sum()
    g2, = torch.autograd.grad(l2, xh)
    ok("T8c chunk + checkpoint の損失と勾配が一括計算と一致", torch.allclose(l1, l2) and torch.allclose(g1, g2))

    # T9 日本語の重み
    ok("T9a 日本語の判定", classify_ja("これは日本語の文章です。") and not classify_ja("def foo(): return 1")
       and not classify_ja("这是中文句子没有假名"))
    toks = list("abcdefgh") + list("日本語のテストです") + list("xyz")
    w, fl = ja_position_weights(list(range(len(toks))), 1, lambda ids: "".join(toks[i] for i in ids), 2.0, chunk=4)
    ok("T9b 位置の重み", w.max() == 2.0 and w.min() == 1.0 and int((w == 2.0).sum()) >= 6, f"{w.tolist()}")

    # T10 パックの書き出し
    with tempfile.TemporaryDirectory() as td:
        s_dir, d_dir = os.path.join(td, "src"), os.path.join(td, "dst")
        os.makedirs(s_dir)
        t_a = {"layers.0.ffn.experts.0.w1.suh": np.ones(8, np.float16), "layers.0.ffn.experts.0.w1.trellis": np.arange(24, dtype=np.int16).reshape(2, 3, 4)}
        t_b = {"embed.weight": np.zeros((4, 4), np.float32)}
        write_safetensors(os.path.join(s_dir, "model-00001-of-00002.safetensors"), t_a)
        write_safetensors(os.path.join(s_dir, "model-00002-of-00002.safetensors"), t_b)
        json.dump({"weight_map": {**{k: "model-00001-of-00002.safetensors" for k in t_a}, **{k: "model-00002-of-00002.safetensors" for k in t_b}}},
                  open(os.path.join(s_dir, "model.safetensors.index.json"), "w"))
        newv = np.arange(8, dtype=np.float16)
        rep = write_scales_pack(s_dir, d_dir, {"layers.0.ffn.experts.0.w1.suh": newv})
        di = SafetensorsIndex(d_dir)
        si = SafetensorsIndex(s_dir)
        same_ino = os.stat(os.path.join(s_dir, "model-00002-of-00002.safetensors")).st_ino == os.stat(os.path.join(d_dir, "model-00002-of-00002.safetensors")).st_ino
        ok("T10 パックの書き出し (対象 shard は新しいファイル、他は hardlink、元は不変)",
           np.array_equal(di.read_numpy("layers.0.ffn.experts.0.w1.suh"), newv)
           and np.array_equal(di.read_numpy("layers.0.ffn.experts.0.w1.trellis"), t_a["layers.0.ffn.experts.0.w1.trellis"])
           and np.array_equal(si.read_numpy("layers.0.ffn.experts.0.w1.suh"), t_a["layers.0.ffn.experts.0.w1.suh"])
           and same_ino and rep["rewritten"] == ["model-00001-of-00002.safetensors"], str(rep))

    # T11 ScaleParams
    sp = ScaleParams({0: [b.float() for b in base[0]]})
    ok("T11a theta=0 で scale = base", all(torch.equal(a, b.float()) for a, b in zip(sp.scales(0), base[0])))
    with torch.no_grad():
        sp.theta["0_1"].fill_(0.1)
    exp_, st = sp.export_fp16()
    nm = "layers.0.ffn.experts.2.w1.svh"
    ok("T11b export (fp16、名前、値)", exp_[nm].dtype == torch.float16 and torch.allclose(exp_[nm].float(), (base[0][1][2] * math.exp(0.1)).float(), rtol=1e-3)
       and len(exp_) == E * 6, str(st))

    # T12 公式 model.py (tiny、CPU) との一致、パイプラインとの勾配一致
    try:
        sys.path.insert(0, ref_dir)
        import model as M
    except Exception as ex:  # noqa
        print("  (ref/model.py を import できない: T12 省略)", ex)
        M = None
    if M is not None:
        _patch_official(M)
        model, args = _build_tiny(M)
        source = _tiny_source(model, args)
        cfg = L3Config(gemm_dtype=torch.float32, bucket=3, attn_chunk=8, round_p=False, loss_chunk=7, tail=True)
        ksm = KernelSet("emul")
        ids = torch.randint(0, args.vocab_size, (1, 32), generator=torch.Generator().manual_seed(5))
        lg_off = _official_logits(M, model, ids)
        sp_all = ScaleParams({l: source.base_scales(l) for l in range(args.n_layers)})
        full = StudentStage(M, model, range(args.n_layers), True, True, sp_all, source, ksm, cfg)
        # 学生の logits (head_kl_loss の手前まで)。(i) 量子化を外すと計算の論理だけを比べられる (fp32 の丸め誤差だけが残る)。
        # (ii) 量子化ありでは、上流の 1e-6 程度の差で indexer の top-k や FP4 の丸めの境目が反転し、局所的に大きな差が出る。
        def student_logits(ids_):
            with torch.no_grad():
                mbx = _tiny_mb(_official_logits(M, model, ids_), ids_).new_pass(False)
                st0 = StudentStage(M, model, range(args.n_layers), True, False, sp_all, source, ksm, cfg)
                h, pm = st0(mbx)
                return model.head(model.norm(model.layers[-1].hc_pre(h, pm)), full_logits=True)[0].float()

        g_ = globals()
        saved = g_["qdq_fp8_emul"], g_["qdq_fp4_emul"]
        g_["qdq_fp8_emul"] = lambda x, block=32, pow2=True: x.clone()
        g_["qdq_fp4_emul"] = lambda x, block=32, e4m3=False: x.clone()
        try:
            rels = []
            for sd in (5, 6, 7):
                ids_ = torch.randint(0, args.vocab_size, (1, 32), generator=torch.Generator().manual_seed(sd))
                lo = _official_logits(M, model, ids_)
                rels.append(float((student_logits(ids_) - lo).abs().max() / lo.abs().max()))
        finally:
            g_["qdq_fp8_emul"], g_["qdq_fp4_emul"] = saved
        ok("T12a 量子化なし: 学生 (θ=0、Q 分解) の logits = 公式 Block (相対 < 1e-5)", max(rels) < 1e-5, f"相対差 {[f'{r:.1e}' for r in rels]}")
        meds, maxs = [], []
        for sd in (5, 6, 7):
            ids_ = torch.randint(0, args.vocab_size, (1, 32), generator=torch.Generator().manual_seed(sd))
            lo = _official_logits(M, model, ids_)
            d = (student_logits(ids_) - lo).abs()
            meds.append(float(d.median() / lo.abs().max()))
            maxs.append(float(d.max() / lo.abs().max()))
        ok("T12a' 量子化あり: 中央値の相対差 < 1e-3 (最大は離散判定の反転で大きくなりうる)", max(meds) < 1e-3,
           f"中央値 {[f'{r:.1e}' for r in meds]}、最大 {[f'{r:.1e}' for r in maxs]}")
        lg_off = _official_logits(M, model, ids)
        # 教師 = 公式 logits を少しずらしたもの (KL > 0 にして勾配を出す)
        lg_t = lg_off + 0.5 * torch.randn_like(lg_off)
        mbs = [_tiny_mb(lg_t, torch.randint(0, args.vocab_size, (1, 32), generator=torch.Generator().manual_seed(10 + i))) for i in range(3)]
        for i, mb in enumerate(mbs):
            t = _tiny_mb(_official_logits(M, model, mb.ids) + 0.5 * torch.randn(32, args.vocab_size), mb.ids)
            mb.top_v, mb.top_i = t.top_v, t.top_i
        # 一括 (checkpoint なし)
        sp_all.zero_grad()
        for mb in mbs:
            mb.new_pass(False)
            lsum, wsum, klt, ag = full(mb)
            (lsum / wsum / len(mbs)).backward()
        g_mono = {k: v.grad.clone() for k, v in sp_all.theta.items()}
        # 2 ステージ 1F1B (checkpoint あり)
        sp0 = ScaleParams({l: source.base_scales(l) for l in range(4)})
        sp1 = ScaleParams({l: source.base_scales(l) for l in range(4, 6)})
        s0 = StudentStage(M, model, range(4), True, False, sp0, source, ksm, cfg)
        s1 = StudentStage(M, model, range(4, 6), False, True, sp1, source, ksm, cfg)
        link = LocalLink()
        r0 = StageRunner(s0, 0, 2, link, ckpt=True, accum=len(mbs))
        r1 = StageRunner(s1, 1, 2, link, ckpt=True, accum=len(mbs))
        order = run_pipeline_local([r0, r1], mbs)
        diffs = []
        for k, v in list(sp0.theta.items()) + list(sp1.theta.items()):
            diffs.append(float((v.grad - g_mono[k]).abs().max() / (g_mono[k].abs().max() + 1e-12)))
        gnorm = sum(float(v.norm()) for v in g_mono.values())
        ok("T12b 2 ステージ 1F1B + checkpoint の θ 勾配 = 一括計算", max(diffs) < 1e-4 and gnorm > 0,
           f"max rel diff {max(diffs):.2e}, |g| {gnorm:.3e}, 順序 {order}")
        # 陰性対照: top-k を保存しないと、stage0 の recompute が後の系列の index_k を拾って勾配がずれるはず
        sp0b = ScaleParams({l: source.base_scales(l) for l in range(4)})
        sp1b = ScaleParams({l: source.base_scales(l) for l in range(4, 6)})
        s0b = StudentStage(M, model, range(4), True, False, sp0b, source, ksm, cfg)
        s1b = StudentStage(M, model, range(4, 6), False, True, sp1b, source, ksm, cfg)
        for mb in mbs:
            mb.cache_topk = False
        try:
            linkc = LocalLink()
            run_pipeline_local([StageRunner(s0b, 0, 2, linkc, True, len(mbs)), StageRunner(s1b, 1, 2, linkc, True, len(mbs))], mbs)
            dneg = max(float((v.grad - g_mono[k]).abs().max() / (g_mono[k].abs().max() + 1e-12))
                       for k, v in sp0b.theta.items() if v.grad is not None)
            ok("T12c 陰性対照 (top-k を保存しないと勾配がずれる)", dneg > 1e-3, f"stage0 の相対差 {dneg:.2e}")
        except Exception as ex:  # noqa
            ok("T12c 陰性対照 (top-k を保存しないと勾配がずれる)", True, f"recompute が例外で止まる: {type(ex).__name__}")
        finally:
            for mb in mbs:
                mb.cache_topk = True
        # Engram: 公式 Engram.forward と engram_apply
        lay = SimpleNamespace(layer_ids=(0,), num_embeddings=(64,), head_dim=32, max_ngram_size=3, n_heads=2)
        Eg = M.Engram(args, 0, lay).float()
        with torch.no_grad():
            for n, p in Eg.named_parameters():
                if p.dtype.is_floating_point:
                    p.copy_(torch.randn(p.shape) * (0.3 if p.dim() < 2 else 1 / math.sqrt(p.shape[-1])))
        rows = torch.randn(1, 10, 4, 32)

        class _Rows(torch.nn.Module):
            def forward(self, idx):
                return rows

        Eg.embed = _Rows()
        hh = torch.randn(1, 10, 4, args.dim)
        with torch.no_grad():
            e_off = Eg(hh, torch.zeros(1, 10, 4, dtype=torch.long))
            e_my = engram_apply(Eg, hh, rows, ksm)
        ok("T12d engram_apply = 公式 Engram.forward", torch.allclose(e_off, e_my, atol=1e-5), f"maxdiff {float((e_off - e_my).abs().max()):.1e}")

        # 評価経路 (forward_only + LocalLink) は学習時の top-1 と KL を同じ値で返す
        link_e = LocalLink()
        for mb in mbs:
            mb.meta["domain"] = "ja" if mb.ids[0, 0] % 2 else "agent"
        _ = forward_only(s0, 0, 2, link_e, mbs)
        ev = forward_only(s1, 1, 2, link_e, mbs)
        sm = summarize(ev)
        kl_full = []
        with torch.no_grad():
            for mb in mbs:
                _, _, klt, _ = full(copy.copy(mb).new_pass(False))
                kl_full.append(klt)
        ok("T12e 評価経路 (2 ステージ forward_only) の KL = 一括 forward", torch.allclose(torch.cat([r["kl"] for r in ev]), torch.cat(kl_full), atol=1e-6),
           json.dumps(sm, ensure_ascii=False))
        gn = clip_grad_global(list(sp0.parameters()) + list(sp1.parameters()), 1e-6)
        gn2 = math.sqrt(sum(float(p.grad.square().sum()) for p in list(sp0.parameters()) + list(sp1.parameters())))
        ok("T12f clip_grad_global", gn > 1e-6 and abs(gn2 - 1e-6) < 1e-8, f"norm {gn:.3e} -> {gn2:.3e}")

        # T16 bf16 (GPU と同じ dtype 構成) の経路: パイプライン = 一括、勾配は有限、fp32 版と同じ向き
        def grads_tiny(dtype, pipe):
            mdl, ar = _build_tiny(M, dtype=dtype)
            srcb = _tiny_source(mdl, ar)
            cfgb = L3Config(gemm_dtype=torch.bfloat16 if dtype == torch.bfloat16 else torch.float32, bucket=3,
                            attn_chunk=8, round_p=dtype == torch.bfloat16, loss_chunk=7)
            if not pipe:
                spb = ScaleParams({l: srcb.base_scales(l) for l in range(6)})
                stb = StudentStage(M, mdl, range(6), True, True, spb, srcb, ksm, cfgb)
                for mb in mbs:
                    lsb, wsb, *_ = stb(copy.copy(mb).new_pass(False))
                    (lsb / wsb / len(mbs)).backward()
                return {k: v.grad.float() for k, v in spb.theta.items()}
            spa = ScaleParams({l: srcb.base_scales(l) for l in range(4)})
            spc = ScaleParams({l: srcb.base_scales(l) for l in range(4, 6)})
            lk = LocalLink()
            run_pipeline_local([StageRunner(StudentStage(M, mdl, range(4), True, False, spa, srcb, ksm, cfgb), 0, 2, lk, True, len(mbs)),
                                StageRunner(StudentStage(M, mdl, range(4, 6), False, True, spc, srcb, ksm, cfgb), 1, 2, lk, True, len(mbs))], mbs)
            return {k: v.grad.float() for k, v in list(spa.theta.items()) + list(spc.theta.items())}

        gb_m, gb_p, gf_m = grads_tiny(torch.bfloat16, False), grads_tiny(torch.bfloat16, True), grads_tiny(torch.float32, False)
        same = max(float((gb_p[k] - gb_m[k]).abs().max()) for k in gb_m)
        fin = all(torch.isfinite(v).all() for v in gb_p.values())
        cs = sum(float((gb_m[k] * gf_m[k]).sum()) for k in gb_m) / (
            math.sqrt(sum(float(gb_m[k].square().sum()) for k in gb_m)) * math.sqrt(sum(float(gf_m[k].square().sum()) for k in gb_m)))
        ok("T16 bf16: 1F1B = 一括、勾配は有限、fp32 との cos > 0.5", same == 0.0 and fin and cs > 0.5,
           f"差 {same:.1e}, cos(bf16, fp32) {cs:.3f} (量子化の離散判定の反転で 1 にはならない)")

    # T13 教師ファイルの形式 (実ファイル)
    if teacher_dir and os.path.exists(os.path.join(teacher_dir, "ja16_teacher.pt")):
        recs = load_teacher(os.path.join(teacher_dir, "ja16_teacher.pt"))
        n = sum(len(r["ids"]) - r["score_from"] for r in recs)
        mb = microbatch_from_record(recs[0], None, 2.0, domain="ja")
        ok("T13 ja16_teacher.pt の形式 (16 系列、採点 9,014 位置、top-32)",
           len(recs) == 16 and n == 9014 and tuple(recs[0]["top_v"].shape[1:]) == (32,) and recs[0]["top_v"].dtype == torch.float16
           and mb.pos.numel() == mb.top_v.shape[0], f"n={n}")

    # T15 export (theta の checkpoint と pack の base から新しい suh/svh を書く)
    with tempfile.TemporaryDirectory() as td:
        s_dir, d_dir = os.path.join(td, "pack"), os.path.join(td, "out")
        os.makedirs(s_dir)
        Ex = 2
        tens, wm = {}, {}
        rng = np.random.default_rng(0)
        for e in range(Ex):
            for p_, kind in SVEC:
                n = 4 if kind == "suh" else 3
                tens[f"layers.7.ffn.experts.{e}.{p_}.{kind}"] = (rng.standard_normal(n) * 0.02).astype(np.float16)
            for p_ in PARTS:
                tens[f"layers.7.ffn.experts.{e}.{p_}.trellis"] = rng.integers(-3000, 3000, (1, 1, 32)).astype(np.int16)
        write_safetensors(os.path.join(s_dir, "model-00001-of-00001.safetensors"), tens)
        json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tens}}, open(os.path.join(s_dir, "model.safetensors.index.json"), "w"))
        base7 = [torch.from_numpy(np.stack([tens[f"layers.7.ffn.experts.{e}.{p_}.{kind}"] for e in range(Ex)])) for p_, kind in SVEC]
        sp7 = ScaleParams({7: base7})
        with torch.no_grad():
            for t_ in sp7.theta.values():
                t_.normal_(0, 0.05)
        torch.save({"theta": sp7.state_dict()}, os.path.join(td, "theta_r0_best.pt"))
        rep = export_main(SimpleNamespace(pack=s_dir, theta=[os.path.join(td, "theta_r0_best.pt")], out=d_dir, dry_run=False))
        di = SafetensorsIndex(d_dir)
        want = sp7.scales(7)[3][1].detach().half().numpy()
        got = di.read_numpy("layers.7.ffn.experts.1.w3.svh")
        tr_ok = np.array_equal(di.read_numpy("layers.7.ffn.experts.0.w2.trellis"), tens["layers.7.ffn.experts.0.w2.trellis"])
        ok("T15 export: suh/svh = base * exp(theta) (fp16)、trellis は不変", np.array_equal(got, want) and tr_ok and rep["n"] == Ex * 6, f"n={rep['n']}")

    # T17 ステージの境目 (実 config)
    rc = os.path.join(ref_dir, "inference_config.json")
    if os.path.exists(rc):
        vs = valid_splits(json.load(open(rc)))
        ok("T17 実 config で使える境目 = [1, 2, 8, 14, 20]", vs == [1, 2, 8, 14, 20], str(vs))
        ok("T17b tiny config の境目に 4 が入る (T12 の分け方)", 4 in valid_splits(dataclasses.asdict(_tiny_args(M))) if M is not None else True)

    # T18 Exl3ResidentSource: K と mcg は (層, 射影) ごと (K の違う層が混ざるパック)。CPU では base と kmeta だけ見る
    with tempfile.TemporaryDirectory() as td:
        tens = {}
        rng = np.random.default_rng(1)
        for l_, Kl in ((3, 2), (4, 3)):
            for e in range(2):
                for p_, kind in SVEC:
                    tens[f"layers.{l_}.ffn.experts.{e}.{p_}.{kind}"] = (rng.standard_normal(16) * 0.02).astype(np.float16)
                for p_ in PARTS:
                    tens[f"layers.{l_}.ffn.experts.{e}.{p_}.trellis"] = rng.integers(-3000, 3000, (1, 1, 16 * Kl)).astype(np.int16)
                    tens[f"layers.{l_}.ffn.experts.{e}.{p_}.mcg"] = np.zeros(1, np.int32)
        write_safetensors(os.path.join(td, "model-00001-of-00001.safetensors"), tens)
        json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tens}}, open(os.path.join(td, "model.safetensors.index.json"), "w"))
        srcK = Exl3ResidentSource(td, [3, 4], "cpu", n_experts=2, workers=2)
        ok("T18a Exl3ResidentSource: K と mcg を (層, 射影) ごとに持つ (K2 と K3 の混在)",
           srcK.kmeta[(3, "w1")] == (2, True) and srcK.kmeta[(4, "w2")] == (3, True) and srcK.K == 2, str(srcK.kmeta))
        del tens["layers.4.ffn.experts.1.w3.mcg"]
        write_safetensors(os.path.join(td, "model-00001-of-00001.safetensors"), tens)
        json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in tens}}, open(os.path.join(td, "model.safetensors.index.json"), "w"))
        try:
            Exl3ResidentSource(td, [4], "cpu", n_experts=2, workers=2)
            ok("T18b mcg の有無が混在するパックは拒否", False)
        except NotImplementedError:
            ok("T18b mcg の有無が混在するパックは拒否", True)

    # T19 summarize: 位置ごとの日本語判定があれば pos:ja / pos:other を足す (1 ファイルに言語が混ざる calib 用)
    ev = [{"domain": "calib256k", "kl": torch.tensor([1.0, 2.0, 3.0, 4.0]), "agree": torch.tensor([1.0, 0.0, 1.0, 1.0]),
           "ja": torch.tensor([True, True, False, False])},
          {"domain": "calib256k", "kl": torch.tensor([5.0]), "agree": torch.tensor([0.0]), "ja": None}]
    sm = summarize(ev)
    ok("T19 summarize: pos:ja / pos:other (domain の値は従来どおり)",
       sm["calib256k"]["n"] == 5 and abs(sm["calib256k"]["kl"] - 3.0) < 1e-6 and sm["pos:ja"]["n"] == 2
       and abs(sm["pos:ja"]["kl"] - 1.5) < 1e-6 and abs(sm["pos:other"]["kl"] - 3.5) < 1e-6
       and abs(sm["pos:ja"]["top1"] - 0.5) < 1e-6, json.dumps(sm))

    # T14 見積もり
    e = estimate()
    ok("T14 見積もりが計算できる", e["mem_gib"][0]["合計"] < e["gpu_gib"] and e["mem_gib"][1]["合計"] < e["gpu_gib"],
       f"GPU0 {e['mem_gib'][0]['合計']:.1f} GiB, GPU1 {e['mem_gib'][1]['合計']:.1f} GiB")
    bad = [k for k, v in res.items() if not v]
    print(f"\n{len(res) - len(bad)}/{len(res)} PASS" + (f"、FAIL: {bad}" if bad else ""))
    return not bad


# =========================================================================== 16. CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("selftest")
    t.add_argument("--ref", default=os.environ.get("DSV41_REF", "/ref"))
    t.add_argument("--teacher-dir", default=os.environ.get("DSV41_TEACHER", "/t"))
    sub.add_parser("estimate")
    tr = sub.add_parser("train")
    tr.add_argument("--ref", default="/ref")
    tr.add_argument("--ckpt", default="/model")
    tr.add_argument("--pack", required=True, help="学生の EXL3 パック (v1 か、qad_l2.py apply --mode shards の出力)")
    tr.add_argument("--teacher", action="append", required=True,
                    help="stream_ref.py --teacher-out の .pt (score_from=1 で取ったもの)。区分ごとに複数可")
    tr.add_argument("--out", required=True)
    tr.add_argument("--max-seq-len", type=int, default=4096)
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--accum", type=int, default=4)
    tr.add_argument("--lr", type=float, default=1e-3, help="theta (対数スケール) の学習率")
    tr.add_argument("--reg", type=float, default=1e-2, help="theta^2 平均の係数")
    tr.add_argument("--clip", type=float, default=1.0)
    tr.add_argument("--ja-weight", type=float, default=2.0)
    tr.add_argument("--val-frac", type=float, default=0.05)
    tr.add_argument("--eval-every", type=int, default=4, help="1 epoch あたりの検証回数")
    tr.add_argument("--bucket", type=int, default=16)
    tr.add_argument("--attn-chunk", type=int, default=256)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--split", type=int, default=20, help="stage0 = 層 0..split-1。valid_splits() の値だけ (実 config で 8/14/20)")
    ex = sub.add_parser("export")
    ex.add_argument("--pack", required=True)
    ex.add_argument("--theta", action="append", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--dry-run", action="store_true")
    ex.add_argument("--force-base", action="store_true", help="theta の base_sha256 が --pack と合わなくても書く")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if selftest(a.ref, a.teacher_dir) else 1)
    if a.cmd == "estimate":
        print_estimate()
        return
    if a.cmd == "train":
        os.makedirs(a.out, exist_ok=True)
        train_main(a)
    if a.cmd == "export":
        export_main(a)


if __name__ == "__main__":
    main()
