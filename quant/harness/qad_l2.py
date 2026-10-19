#!/usr/bin/env python3
"""L2 局所 QAD: EXL3 パック (v1) の suh/svh を、MXFP4 教師の routed expert 出力へ層ごとに蒸留する。

学生の重み (EXL3 の [in, out] 向き):
    W_e = diag(su) · Q̃ · diag(sv),   Q̃ = H_in · Q · H_out (trellis を su=sv=1 で復元したもの)
exllamav3 の get_weight_tensor (modules/quant/exl3.py:239-248) は
    w = had_l(inner); w *= suh[:, None]; w = had_r(w); w *= svh[None, :]
で、had_l は左から H_in を掛け (quantize.py:401-408)、had_r は右から H_out を掛ける (quantize.py:411-418)。
suh の行スケールは左からの掛け算、had_r は右からの掛け算なので、結合法則
    (diag(su) · H_in Q) · H_out = diag(su) · (H_in Q H_out)
により「had_r の前に suh を掛ける」と「Q̃ を作ってから suh を掛ける」は実数演算で一致する。
一方 diag(su) と H_in は行列として可換ではない (diag(su) H_in != H_in diag(su)) ので、
suh を had_l より前 (inner の側) に入れることはできない。本実装は Q̃ を ones で作り、外側から su/sv を掛ける。
差は get_weight_tensor が各段で fp16 に丸める分 (相対 1e-3 程度) だけで、test_qad_l2.py で確かめている。

学習対象: 各 expert の w1/w3/w2 の suh/svh (6 本、22,272 値/expert)。su = su_v1 · (1 + δ) の乗法形で
δ を fp32 で学習し、保存時に fp16 に丸める (符号は保たれる。v1 の |suh| は約 0.017 なので絶対値での lr は不適)。

使い方 (コンテナ内、GPU 1 枚):
  python3 qad_l2.py train --capture /cap/p0 --capture /cap/p1 --layers 0-19 \
      --orig-ckpt /model --pack /pack --out /out/qad [--loss expert|layer] [--ja-weight 2.0]
  python3 qad_l2.py apply --pack /pack --diff /out/qad --out /models/v1-qad --mode overlay|shards
  python3 qad_l2.py estimate --tokens 600000
  python3 qad_l2.py check-decomp --pack /pack --layer 0 --experts 0,7      (GPU。分解の実測)
  python3 qad_l2.py check-teacher --orig-ckpt /model --capture /cap/p0 --layer 0 --experts 0,7  (GPU。教師の一致)

入力 (--capture): capture_moe.py の記録ディレクトリ (dsv41.moe-capture.v1)。capture_moe.load_layer で読み、
  x (ffn_norm 後の MoE 入力 bf16)、idx (top-6)、w (route weight、route_scale 込み)、flags の F_JA (日本語文脈)、
  seq (系列番号、検証の分割に使う)、kp (保持確率。1/kp を重要度重みにする) を使う。F_DUP 行は捨てる。
  代わりに --layer-file で torch.save の dict {x, indices, weights, lang, seq?, img?} も読める。

画像トークン (capture の field tt >= 0。画像付きの記録だけが持つ):
  - 損失の重み: c_t = imp × (ja_weight if 日本語) × (img_weight if 画像)。--img-weight の既定は 1.0
    (画像の無い記録では結果が変わらない)。r6_vision_qad.md は 1.5 を推奨。
  - 画像の守り (expert mode): 検証に画像の (token, expert) 対が --min-val-img 以上ある expert は、学習後の
    画像だけの検証誤差が v1 の (1 + --img-guard-tol) 倍を超えたら採らない (status "rejected_img")。
    日本語・全体の誤差が下がっていても、画像の入力分布で悪くなる scale は捨てる。
  - 検証の分割は画像を含む系列を別の層として選ぶ (系列が少なくても検証に画像が入るように)。
  - 層の報告 layer_val に img / text を足す (画像の行が検証にあるときだけ)。
"""
import argparse
import ast
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

PARTS = ("w1", "w3", "w2")  # gate, up, down (stream_ref.ExpertStore.PARTS と同じ順)
FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)  # ref/convert.py:13-15
FP8_MAX = 448.0
HERE = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# =========================================================================== safetensors (最小実装)
_ST_DT = {"F16": torch.float16, "F32": torch.float32, "BF16": torch.bfloat16, "I8": torch.int8, "U8": torch.uint8,
          "I16": torch.int16, "I32": torch.int32, "I64": torch.int64, "F8_E8M0": torch.float8_e8m0fnu,
          "F8_E4M3": torch.float8_e4m3fn}
_ST_NAME = {v: k for k, v in _ST_DT.items()}


def st_header(path):
    with open(path, "rb") as f:
        hl = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(hl))
    return hdr, 8 + hl


def st_save(path, tensors, metadata=None):
    """safetensors 形式で書く (ヘッダは 8 byte 境界に空白で詰める)。tmp に書いてから rename。"""
    hdr, off, blobs = {}, 0, []
    for k in sorted(tensors):
        t = tensors[k].detach().contiguous().cpu()
        b = t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
        hdr[k] = {"dtype": _ST_NAME[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)


def st_load(path):
    hdr, base = st_header(path)
    out = {}
    with open(path, "rb") as f:
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            raw = bytearray(f.read(b - a))
            t = torch.frombuffer(raw, dtype=torch.uint8) if raw else torch.empty(0, dtype=torch.uint8)
            out[k] = t.view(_ST_DT[v["dtype"]]).view(v["shape"]).clone()
    return out, hdr.get("__metadata__", {})


# =========================================================================== stream_ref.TensorIndex
def stream_ref_tensor_index():
    """stream_ref.py の TensorIndex をそのまま使う。

    stream_ref を普通に import すると kernel (tilelang) と model を読み込む (stream_ref.py:32-33) ので、
    既定では stream_ref.py から `_DT` と `class TensorIndex` だけを AST で取り出して実行する (ファイルは変更しない)。
    QAD_IMPORT_STREAM_REF=1 なら通常の import を使う。
    """
    if os.environ.get("QAD_IMPORT_STREAM_REF") == "1":
        sys.path.insert(0, HERE)
        import stream_ref  # noqa: E402
        return stream_ref.TensorIndex
    path = os.path.join(HERE, "stream_ref.py")
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    keep = [n for n in tree.body
            if (isinstance(n, ast.ClassDef) and n.name == "TensorIndex")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "_DT" for t in n.targets))]
    assert len(keep) == 2, "stream_ref.py に _DT / TensorIndex が見つからない"
    g = {"json": json, "os": os, "torch": torch, "__name__": "stream_ref_tensorindex"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), g)
    return g["TensorIndex"]


def ename(layer, e, part, kind):
    return f"layers.{layer}.ffn.experts.{e}.{part}.{kind}"


# =========================================================================== 教師 (MXFP4) の torch 再現
def e8m0_to_f32(s):
    """E8M0 (符号なし指数のみ) -> 2^(u-127)。0xFF (NaN) は扱わない。"""
    u = s.view(torch.uint8).to(torch.int32)
    return torch.ldexp(torch.ones(u.shape, dtype=torch.float32, device=u.device), u - 127)


def fp4_dequant(w, s, out_dtype=torch.bfloat16):
    """w: [out, in/2] int8 (下位 nibble が偶数番目、ref/convert.py:31-34)、s: [out, in/32] E8M0。

    返り値 [out, in]。FP4 値 × 2 冪なので bf16 で厳密に表せる (丸めなし)。
    """
    u = w.view(torch.uint8)
    lut = torch.tensor(FP4_LUT, dtype=torch.float32, device=w.device)
    v = torch.stack([lut[(u & 0x0F).long()], lut[(u >> 4).long()]], dim=-1).flatten(-2)
    o, i = v.shape
    assert s.shape == (o, i // 32), (s.shape, v.shape)
    v = v.view(o, i // 32, 32) * e8m0_to_f32(s).view(o, i // 32, 1)
    return v.view(o, i).to(out_dtype)


def act_quant_sim(x, block=32):
    """kernel.act_quant(x, 32, "ue8m0") の torch 再現 (ref/kernel.py:20-37, 72-89):
    amax = max(|x|, 1e-4) をブロック 32 ごとに取り、s = 2^ceil(log2(amax/448))、y = e4m3(clamp(x/s)) * s。"""
    xf = x.float()
    shp = xf.shape
    xb = xf.reshape(*shp[:-1], shp[-1] // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    v = amax * (1.0 / FP8_MAX)
    m, ex = torch.frexp(v)  # v = m * 2^ex, m ∈ [0.5, 1)
    ce = torch.where(m == 0.5, ex - 1, ex)  # ceil(log2 v) (fast_log2_ceil と同じ: 仮数 0 なら切り上げなし)
    sc = torch.ldexp(torch.ones_like(v), ce)
    q = (xb / sc).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * sc
    return q.reshape(shp)


def teacher_expert(x, W, g, limit, act_quant=True, cdt=torch.bfloat16):
    """stream_ref.expert_forward の FP4 経路 (stream_ref.py:281-289) と同じ式。
    W = (w1, w3, w2) を [out, in] の dequant 済み重みで渡す。g は [n, 1] の route weight。返り値 bf16 [n, dim]。"""
    w1, w3, w2 = W
    a = act_quant_sim(x) if act_quant else x.float()
    a = a.to(cdt)
    gate = (a @ w1.t()).to(torch.bfloat16).float()  # fp4_gemm の出力は既定 dtype (bf16)
    up = (a @ w3.t()).to(torch.bfloat16).float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)  # up は ±limit (ref/model.py:848)
        gate = torch.clamp(gate, max=limit)  # gate は上側だけ (ref/model.py:849)
    h = (g * (F.silu(gate) * up)).to(torch.bfloat16)
    a2 = act_quant_sim(h) if act_quant else h.float()
    return (a2.to(cdt) @ w2.t()).to(torch.bfloat16)


# =========================================================================== 学生 (suh/svh だけ学習)
class ScaleBank(torch.nn.Module):
    """E 本の expert の su/sv を持つ。su = su0 · (1 + du)、sv = sv0 · (1 + dv)、du/dv を学習 (fp32)。

    qt[j] = (Q̃_w1, Q̃_w3, Q̃_w2) は [in, out] で凍結 (cdt)。学生の 1 射影は
        x · diag(su) Q̃ diag(sv) = ((x ⊙ su) @ Q̃) ⊙ sv
    で、W を作らずに計算する。勾配は su/sv にだけ流れ、Q̃ の勾配 GEMM は要らない。
    """

    def __init__(self, su0, sv0, qt, limit, cdt):
        super().__init__()
        for i in range(3):
            self.register_buffer(f"su0_{i}", su0[i].float())
            self.register_buffer(f"sv0_{i}", sv0[i].float())
        self.du = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(su0[i], dtype=torch.float32)) for i in range(3)])
        self.dv = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(sv0[i], dtype=torch.float32)) for i in range(3)])
        self.qt = qt
        self.limit = limit
        self.cdt = cdt

    def su(self, i, j):
        return getattr(self, f"su0_{i}")[j] * (1 + self.du[i][j])

    def sv(self, i, j):
        return getattr(self, f"sv0_{i}")[j] * (1 + self.dv[i][j])

    def lin(self, i, j, a):
        return ((a.float() * self.su(i, j)).to(self.cdt) @ self.qt[j][i]).float() * self.sv(i, j)

    def forward(self, j, x, g):
        gate = self.lin(0, j, x)
        up = self.lin(1, j, x)
        if self.limit > 0:
            up = torch.clamp(up, min=-self.limit, max=self.limit)
            gate = torch.clamp(gate, max=self.limit)
        h = F.silu(gate) * up * g
        return self.lin(2, j, h)

    def reg(self, rows=None):
        tot = 0.0
        for p in list(self.du) + list(self.dv):
            q = p if rows is None else p[rows]
            tot = tot + q.pow(2).mean()
        return tot / 6.0

    def get_delta(self):
        return [p.detach().clone() for p in self.du] + [p.detach().clone() for p in self.dv]

    @torch.no_grad()
    def set_delta(self, d):
        for p, v in zip(list(self.du) + list(self.dv), d):
            p.copy_(v)

    @torch.no_grad()
    def round_fp16_(self):
        """保存形式 (fp16) に丸めた値に δ を合わせる (評価を保存値で行うため)。"""
        for i in range(3):
            for pre, d in (("su0", self.du[i]), ("sv0", self.dv[i])):
                base = getattr(self, f"{pre}_{i}")
                full = (base * (1 + d)).half().float()
                d.copy_(torch.where(base != 0, full / torch.where(base != 0, base, torch.ones_like(base)) - 1, torch.zeros_like(d)))

    @torch.no_grad()
    def export(self, j):
        """[(suh fp16, svh fp16) for w1, w3, w2]"""
        return [(self.su(i, j).half(), self.sv(i, j).half()) for i in range(3)]


# =========================================================================== 入力源
def qtilde_exl3(trellis, mcg, device):
    """Q̃ = trellis を suh=svh=1 で復元した [in, out] fp16 (stream_ref.py:226-228 と同じ呼び方で ones を渡す)。"""
    from exllamav3.modules.quant.exl3 import LinearEXL3
    tr = trellis.to(device)
    in_f, out_f = tr.shape[0] * 16, tr.shape[1] * 16
    lin = LinearEXL3(None, in_f, out_f, suh=torch.ones(in_f, dtype=torch.half, device=device),
                     svh=torch.ones(out_f, dtype=torch.half, device=device), trellis=tr,
                     mcg=None if mcg is None else mcg.to(device))
    w = lin.get_weight_tensor()
    del lin
    return w


class TeacherFP4:
    """原本の MXFP4 experts (layers.L.ffn.experts.E.wN.{weight,scale})。"""

    def __init__(self, ckpt):
        self.t = stream_ref_tensor_index()(ckpt)

    def raw(self, layer, e):
        return {p: (self.t.load(ename(layer, e, p, "weight")), self.t.load(ename(layer, e, p, "scale"))) for p in PARTS}

    @staticmethod
    def weights(raw, device, cdt):
        return tuple(fp4_dequant(raw[p][0].to(device), raw[p][1].to(device), cdt) for p in PARTS)


class PackEXL3:
    """学生パック (v1)。layers.L.ffn.experts.E.wN.{trellis,suh,svh,mcg}。"""

    qtilde_fn = staticmethod(qtilde_exl3)

    def __init__(self, pack):
        self.t = stream_ref_tensor_index()(pack)

    def raw(self, layer, e):
        out = {}
        for p in PARTS:
            d = {}
            for kind in ("trellis", "suh", "svh", "mcg"):
                nm = ename(layer, e, p, kind)
                if nm in self.t.meta:
                    d[kind] = self.t.load(nm)
            out[p] = d
        return out


# =========================================================================== データ
def _pick(d, names, required=True):
    for n in names:
        if n in d:
            return d[n]
    if required:
        raise KeyError(f"層ファイルに {names} のどれも無い (keys={list(d)})")
    return None


F_JA, F_DUP = 4, 128  # capture_moe.py:44 のフラグ


def load_capture(dirs, layer, use_kp=True):
    """capture_moe.py の記録 (dsv41.moe-capture.v1) を読む。capture_moe.load_layer をそのまま使う。

    imp = 1/kp (間引きの保持確率の逆数)。capture は日本語文脈を ja_boost 倍で残す (capture_moe.py:276-281) ので、
    imp を掛けて自然な分布に戻してから ja_weight を掛ける (掛けないと日本語の実効重みが ja_boost × ja_weight になる)。
    """
    sys.path.insert(0, HERE)
    import capture_moe  # numpy と標準ライブラリだけ
    t = capture_moe.load_layer(list(dirs), layer, as_torch=True)
    metas = [json.load(open(os.path.join(d, "meta.json"))) for d in dirs]
    fl = t["flags"].to(torch.int32)
    live = (fl & F_DUP) == 0
    kp = t["kp"].float()
    imp = torch.where(kp > 0, 1.0 / kp.clamp_min(1e-12), torch.zeros_like(kp)) if use_kp else torch.ones_like(kp)
    sel = live & (imp > 0)
    out = {"x": t["x"][sel], "indices": t["idx"][sel].long(), "weights": t["w"][sel].float(),
           "lang": ((fl & F_JA) > 0)[sel], "seq": t["seq"][sel].long(), "imp": imp[sel],
           "stream": sorted({m.get("stream", "?") for m in metas})}
    if "tt" in t:  # 画像付きの記録 (capture_moe の VL_FIELD)
        out["img"] = (t["tt"].to(torch.int32) >= 0)[sel]
    return out


def load_layer_files(paths, lang_ja_value=1):
    xs, ids, ws, ls, ss, ims = [], [], [], [], [], []
    seq_off = 0
    for p in paths:
        try:
            d = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        except Exception:
            d = torch.load(p, map_location="cpu", weights_only=False)
        x = _pick(d, ("x", "moe_in", "h"))
        n = x.shape[0]
        idx = _pick(d, ("indices", "idx", "topk_idx")).long()
        w = _pick(d, ("weights", "route_w", "topk_w", "w")).float()
        lang = _pick(d, ("lang", "is_ja", "ja"), required=False)
        if lang is None:
            log(f"[data] {p}: lang が無い -> 全トークンを非日本語として扱う")
            lang = torch.zeros(n, dtype=torch.bool)
        else:
            lang = torch.as_tensor(lang)
            lang = lang.bool() if lang.dtype == torch.bool else (lang == lang_ja_value)
        seq = _pick(d, ("seq", "seq_id", "sid"), required=False)
        seq = torch.full((n,), -1, dtype=torch.long) if seq is None else torch.as_tensor(seq).long() + seq_off
        if (seq >= 0).any():
            seq_off = int(seq.max()) + 1
        img = _pick(d, ("img", "is_img", "image"), required=False)
        ims.append(torch.zeros(n, dtype=torch.bool) if img is None else torch.as_tensor(img).bool().view(-1))
        xs.append(x); ids.append(idx); ws.append(w); ls.append(lang.view(-1)); ss.append(seq.view(-1))
    n = sum(v.shape[0] for v in xs)
    out = {"x": torch.cat(xs) if len(xs) > 1 else xs[0], "indices": torch.cat(ids), "weights": torch.cat(ws),
           "lang": torch.cat(ls), "seq": torch.cat(ss), "imp": torch.ones(n), "stream": ["?"]}
    img = torch.cat(ims)
    if img.any():
        out["img"] = img
    return out


def split_val(lang, seq, frac, seed, block=256, img=None, frac_img=None):
    """検証用トークンの印 [N] bool。seq があれば系列単位 (日本語主/それ以外で層別。img に画像があれば
    画像を含む系列を 3 つ目の層にする)、無ければ 256 トークンの塊単位。img が None か画像なしなら従来と同じ。
    frac_img: 画像を含む系列の層だけの検証割合 (None なら frac)。画像系列は少ない (vl_calib は 24 本) ので、
    frac 0.1 だと検証が 2 本になり、画像の守りが expert ごとの検証対の下限 (--min-val-img) に届かない。"""
    n = lang.numel()
    gen = torch.Generator().manual_seed(seed)
    val = torch.zeros(n, dtype=torch.bool)
    if (seq >= 0).all():
        uniq = torch.unique(seq)
        ja_major = torch.tensor([lang[seq == s].float().mean().item() >= 0.5 for s in uniq.tolist()])
        groups = (uniq[ja_major], uniq[~ja_major])
        if img is not None and bool(img.any()):
            has_img = torch.isin(uniq, torch.unique(seq[img]))
            groups = (uniq[has_img], uniq[ja_major & ~has_img], uniq[~ja_major & ~has_img])
        fracs = (frac,) * len(groups)
        if len(groups) == 3:
            fracs = (frac if frac_img is None else frac_img, frac, frac)
        for grp, fr in zip(groups, fracs):
            if grp.numel() == 0:
                continue
            k = max(1, round(fr * grp.numel())) if grp.numel() >= 2 else 0
            pick = grp[torch.randperm(grp.numel(), generator=gen)[:k]]
            val |= torch.isin(seq, pick)
        if val.any():
            return val
    nb = (n + block - 1) // block
    pick = torch.rand(nb, generator=gen) < frac
    return pick.repeat_interleave(block)[:n]


# =========================================================================== 見積もり
def estimate(n_tok, topk=6, dim=5120, inter=2304, n_exp=384, epochs=4, batch=1024, min_steps=50, max_steps=400,
             val_frac=0.1, eval_every=10, counts=None, loss="expert", tflops=(80.0, 150.0), nvme_gbs=5.0):
    fwd = 3 * 2 * dim * inter  # 1 (token, expert) の expert forward FLOP
    pairs = n_tok * topk
    if counts is None:
        counts = [pairs / n_exp] * n_exp
    teacher = pairs * fwd
    if loss == "expert":
        train_pairs = 0.0
        evals = 0.0
        for c in counts:
            ntr = c * (1 - val_frac)
            st = min(max_steps, max(min_steps, epochs * math.ceil(max(ntr, 1) / batch)))
            train_pairs += st * min(batch, max(ntr, 1))
            evals += (st / eval_every + 2) * c * val_frac
        train = 2 * fwd * train_pairs + fwd * evals
    else:
        train = 2 * fwd * pairs * (1 - val_frac) * epochs + fwd * pairs * val_frac * (epochs * 4 + 2)
    tot = teacher + train
    io_gb = n_exp * (3 * (dim * inter / 2 + dim * inter / 32) + 3 * dim * inter * 2 / 8 + 6 * 7424) / 1e9 \
        + n_tok * dim * 2 / 1e9
    x_gb = n_tok * dim * 2 / 2**30
    hot = max(counts)
    mem_expert = x_gb + n_tok * topk * 12 / 2**30 + hot * dim * 2 * 2 / 2**30 + 3 * dim * inter * 2 * 2 / 2**30 \
        + batch * (dim * 4 * 3 + inter * 4 * 8) / 2**30 + 1.0
    mem_layer = x_gb + n_tok * dim * 2 / 2**30 + n_exp * 3 * dim * inter * 2 / 2**30 \
        + 16384 * topk * (dim * 4 * 3 + inter * 4 * 8) / 2**30 + 2.0
    return {
        "tokens": n_tok, "pairs": pairs, "teacher_TFLOP": teacher / 1e12, "train_TFLOP": train / 1e12,
        "total_TFLOP": tot / 1e12, "compute_s": [tot / (t * 1e12) for t in tflops[::-1]],
        "io_GB": io_gb, "io_s": io_gb / nvme_gbs,
        "gpu_mem_GiB": mem_expert if loss == "expert" else mem_layer, "hot_expert_pairs": hot,
    }


# =========================================================================== 学習
class Cfg:
    pass


def _batched(fn, n, bs):
    return [fn(slice(a, min(n, a + bs))) for a in range(0, n, bs)]


def _rel_err(yT, yS, c):
    """Σ c‖yT−yS‖² / Σ c‖yT‖² (c は日本語重み)。"""
    num = (c * (yT.float() - yS.float()).pow(2).sum(-1)).sum()
    den = (c * yT.float().pow(2).sum(-1)).sum()
    return num, den


def _prefetch(pool, teacher, pack, layer, experts, ahead=3):
    futs = {}
    order = list(experts)

    def submit(k):
        if k < len(order) and order[k] not in futs:
            e = order[k]
            futs[e] = (pool.submit(teacher.raw, layer, e), pool.submit(pack.raw, layer, e))

    for k in range(min(ahead, len(order))):
        submit(k)
    for k, e in enumerate(order):
        submit(k + ahead)
        ft, fp = futs.pop(e)
        yield e, ft.result(), fp.result()


def _expert_tensors(raw_p, device, cdt, qtilde_fn):
    qt, su0, sv0 = [], [], []
    for p in PARTS:
        d = raw_p[p]
        assert "trellis" in d and "suh" in d and "svh" in d, f"{p}: trellis/suh/svh が無い"
        qt.append(qtilde_fn(d["trellis"], d.get("mcg"), device).to(cdt).contiguous())
        su0.append(d["suh"].to(device).float()[None])
        sv0.append(d["svh"].to(device).float()[None])
    return qt, su0, sv0


def train_layer(layer, data, teacher, pack, cfg, device):
    """1 層を学習し、(新しい suh/svh の dict, 報告 dict) を返す。"""
    t_start = time.time()
    cdt = cfg.cdt
    x = data["x"].to(device=device, dtype=torch.bfloat16)
    idx = data["indices"].to(device)
    rw = data["weights"].to(device).float()
    lang = data["lang"].to(device)
    img_cpu = data.get("img")
    if img_cpu is None:
        img_cpu = torch.zeros_like(data["lang"], dtype=torch.bool)
    img = img_cpu.to(device)
    val = split_val(data["lang"], data["seq"], cfg.val_frac, cfg.seed + layer, img=img_cpu,
                    frac_img=getattr(cfg, "val_frac_img", None)).to(device)
    n, k = idx.shape
    n_exp = cfg.n_experts or int(idx.max()) + 1
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_exp)
    starts = torch.cumsum(counts, 0) - counts
    pair_tok = order // k
    pair_k = order % k
    imp = data.get("imp")
    imp = torch.ones(n, device=device) if imp is None else imp.to(device).float()
    imp = imp / imp.mean().clamp_min(1e-30)
    cw = imp * torch.where(lang, torch.tensor(cfg.ja_weight, device=device), torch.tensor(1.0, device=device))
    img_w = float(getattr(cfg, "img_weight", 1.0))
    if img_w != 1.0 and bool(img.any()):
        cw = cw * torch.where(img, torch.tensor(img_w, device=device), torch.tensor(1.0, device=device))
    # 日本語比率は自然分布 (imp 重み) で数える
    ptr = ~val[pair_tok]
    ja_share_layer = ((imp[pair_tok] * lang[pair_tok].float())[ptr].sum() / imp[pair_tok][ptr].sum().clamp_min(1e-30)).item() if n else 0.0
    est = estimate(n, topk=k, dim=x.shape[1], n_exp=n_exp, epochs=cfg.epochs, batch=cfg.batch, min_steps=cfg.min_steps,
                   max_steps=cfg.max_steps, val_frac=cfg.val_frac, eval_every=cfg.eval_every,
                   counts=counts.tolist(), loss=cfg.loss)
    log(f"[L{layer}] N={n} top{k} experts={n_exp} val={int(val.sum())} ja_share={ja_share_layer:.3f} "
        f"loss={cfg.loss} 見積 {est['total_TFLOP']:.0f} TFLOP ({est['compute_s'][0]:.0f}-{est['compute_s'][1]:.0f} s) "
        f"GPU {est['gpu_mem_GiB']:.1f} GiB")
    groups = {}
    for e in range(n_exp):
        c = int(counts[e])
        if c == 0:
            continue
        sl = slice(int(starts[e]), int(starts[e]) + c)
        tok = pair_tok[sl]
        groups[e] = (tok, rw[tok, pair_k[sl]][:, None])
    pool = ThreadPoolExecutor(8)
    if cfg.loss == "expert":
        res = _train_expert_mode(layer, x, groups, lang, val, cw, imp, ja_share_layer, teacher, pack, cfg, device, pool,
                                 img=img)
    else:
        res = _train_layer_mode(layer, x, groups, lang, val, cw, teacher, pack, cfg, device, pool, n_exp, img=img)
    pool.shutdown()
    new, rep = res
    rep["layer"] = layer
    rep["stream"] = data.get("stream")
    rep["seconds"] = time.time() - t_start
    rep["estimate"] = est
    rep["config"] = {k2: (str(v) if isinstance(v, torch.dtype) else v) for k2, v in vars(cfg).items()}
    return new, rep


def _train_one(bank, xtr, gtr, ctr, ytr, xva, gva, cva, yva, jva, use_ja, cfg):
    """1 expert を Adam で学習。返り値 (init_val, best_val, steps)。bank は best に戻して返す。"""
    ntr = xtr.shape[0]
    den_tr = (ctr * ytr.float().pow(2).sum(-1)).mean().clamp_min(1e-30)

    def evaluate():
        with torch.no_grad():
            if use_ja:
                m = jva
                xs, gs, cs, ys = xva[m], gva[m], cva[m], yva[m]
            else:
                xs, gs, cs, ys = xva, gva, cva, yva
            if xs.shape[0] == 0:
                return float("nan")
            parts = _batched(lambda s: _rel_err(ys[s], bank(0, xs[s], gs[s]), cs[s]), xs.shape[0], cfg.eval_batch)
            num = sum(p[0] for p in parts)
            den = sum(p[1] for p in parts)
            return (num / den.clamp_min(1e-30)).item()

    init = evaluate()
    best, best_d, bad = init, bank.get_delta(), 0
    steps = min(cfg.max_steps, max(cfg.min_steps, cfg.epochs * math.ceil(ntr / cfg.batch)))
    opt = torch.optim.Adam(bank.parameters(), lr=cfg.lr)
    gen = torch.Generator(device=xtr.device).manual_seed(cfg.seed)
    perm = torch.randperm(ntr, device=xtr.device, generator=gen)
    pos, done = 0, 0
    for st in range(steps):
        if pos >= ntr:
            perm = torch.randperm(ntr, device=xtr.device, generator=gen)
            pos = 0
        b = perm[pos:pos + cfg.batch]
        pos += cfg.batch
        yS = bank(0, xtr[b], gtr[b])
        loss = (ctr[b] * (ytr[b].float() - yS).pow(2).sum(-1)).mean() / den_tr + cfg.l2 * bank.reg()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        done = st + 1
        if done % cfg.eval_every == 0 or done == steps:
            v = evaluate()
            if v < best * (1 - cfg.es_tol):
                best, best_d, bad = v, bank.get_delta(), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    bank.set_delta(best_d)
    return init, best, done, evaluate


def _subset_err(bank, xs, gs, cs, ys, cfg):
    """bank の今の scale での rel-MSE (Σ c‖yT−yS‖² / Σ c‖yT‖²)。"""
    with torch.no_grad():
        if xs.shape[0] == 0:
            return float("nan")
        parts = _batched(lambda s: _rel_err(ys[s], bank(0, xs[s], gs[s]), cs[s]), xs.shape[0], cfg.eval_batch)
        return (sum(p[0] for p in parts) / sum(p[1] for p in parts).clamp_min(1e-30)).item()


def _train_expert_mode(layer, x, groups, lang, val, cw, imp, ja_share_layer, teacher, pack, cfg, device, pool, img=None):
    cdt = cfg.cdt
    if img is None:
        img = torch.zeros_like(lang)
    min_val_img = int(getattr(cfg, "min_val_img", 32))
    img_tol = float(getattr(cfg, "img_guard_tol", 0.0))
    n_rej_img = n_guard_img = 0
    new, per = {}, {}
    rep_layer = cfg.layer_report
    n, dim = x.shape
    vtok = torch.nonzero(val).view(-1)
    if rep_layer:
        vpos = torch.full((n,), -1, dtype=torch.long, device=device)
        vpos[vtok] = torch.arange(vtok.numel(), device=device)
        accT = torch.zeros(vtok.numel(), dim, device=device)
        acc0 = torch.zeros_like(accT)
        acc1 = torch.zeros_like(accT)
    t_io = t_teacher = t_train = 0.0
    n_acc = n_rej = n_cold = 0
    for e, raw_t, raw_p in _prefetch(pool, teacher, pack, layer, sorted(groups)):
        t0 = time.time()
        tok, g = groups[e]
        WT = teacher.weights(raw_t, device, cdt)
        qt, su0, sv0 = _expert_tensors(raw_p, device, cdt, pack.qtilde_fn)
        t1 = time.time()
        xe = x[tok]
        yT = torch.cat(_batched(lambda s: teacher_expert(xe[s], WT, g[s], cfg.limit, cfg.teacher_act_quant, cdt),
                                xe.shape[0], cfg.eval_batch))
        del WT
        t2 = time.time()
        bank = ScaleBank(su0, sv0, [qt], cfg.limit, cdt).to(device)
        ve = val[tok]
        te = ~ve
        ce = cw[tok]
        je = lang[tok]
        ntr, nva = int(te.sum()), int(ve.sum())
        ie = imp[tok][te]
        ja_e = ((ie * je[te].float()).sum() / ie.sum().clamp_min(1e-30)).item() if ntr else 0.0
        use_ja = (ja_share_layer > 0 and ja_e >= cfg.ja_expert_ratio * ja_share_layer
                  and int((je & ve).sum()) >= cfg.min_val_ja)
        info = {"n_train": ntr, "n_val": nva, "ja_share": round(ja_e, 4), "es_metric": "ja" if use_ja else "all"}
        if ntr < cfg.min_tokens or nva < cfg.min_val:
            info["status"] = "cold"
            n_cold += 1
        else:
            init, best, steps, evaluate = _train_one(
                bank, xe[te], g[te], ce[te], yT[te], xe[ve], g[ve], ce[ve], yT[ve], je[ve], use_ja, cfg)
            bank.round_fp16_()
            fin = evaluate()
            info.update({"init_val": init, "best_val": best, "final_val_fp16": fin, "steps": steps})
            iv = ve & img[tok]
            img_bad = False
            if int(iv.sum()) >= min_val_img:  # 画像の守り: 画像だけの検証誤差が v1 より悪くなる scale は採らない
                img_fin = _subset_err(bank, xe[iv], g[iv], ce[iv], yT[iv], cfg)
                saved = bank.get_delta()
                bank.set_delta([torch.zeros_like(d) for d in saved])
                img_init = _subset_err(bank, xe[iv], g[iv], ce[iv], yT[iv], cfg)
                bank.set_delta(saved)
                img_bad = not (img_fin <= img_init * (1 + img_tol))
                n_guard_img += 1
                info.update({"n_val_img": int(iv.sum()), "img_init_val": img_init, "img_final_val": img_fin})
            if fin < init and img_bad:
                info["status"] = "rejected_img"
                n_rej_img += 1
                bank.set_delta([torch.zeros_like(d) for d in bank.get_delta()])
            elif fin < init:
                info["status"] = "accepted"
                n_acc += 1
                for p, (su, sv) in zip(PARTS, bank.export(0)):
                    assert torch.isfinite(su).all() and torch.isfinite(sv).all()
                    new[ename(layer, e, p, "suh")] = su.cpu()
                    new[ename(layer, e, p, "svh")] = sv.cpu()
            else:
                info["status"] = "rejected"
                n_rej += 1
                bank.set_delta([torch.zeros_like(d) for d in bank.get_delta()])
        if rep_layer and nva:
            with torch.no_grad():
                pv = vpos[tok[ve]]
                accT.index_add_(0, pv, yT[ve].float())
                acc1.index_add_(0, pv, bank(0, xe[ve], g[ve]))
                saved = bank.get_delta()
                bank.set_delta([torch.zeros_like(d) for d in saved])
                acc0.index_add_(0, pv, bank(0, xe[ve], g[ve]))
                bank.set_delta(saved)
        t3 = time.time()
        t_io += t1 - t0
        t_teacher += t2 - t1
        t_train += t3 - t2
        per[e] = info
        if cfg.verbose or info["status"] != "cold" and e % 32 == 0:
            log(f"[L{layer}] e{e}: {info}")
        del bank, qt, yT, xe
    rep = {"experts": per, "accepted": n_acc, "rejected": n_rej, "cold": n_cold,
           "time": {"io_qtilde": t_io, "teacher": t_teacher, "train": t_train}}
    if bool(img.any()):
        rep["rejected_img"] = n_rej_img
        rep["img_guarded"] = n_guard_img  # 守りを掛けられた (画像の検証対が --min-val-img 以上の) expert 数
    if rep_layer and vtok.numel():
        cv = cw[vtok]
        jv = lang[vtok]

        def rel(a, m):
            num, den = _rel_err(accT[m], a[m], cv[m])
            return (num / den.clamp_min(1e-30)).item() if m.any() else float("nan")

        allm = torch.ones_like(jv)
        grp = [("weighted", allm), ("ja", jv), ("other", ~jv)]
        iv = img[vtok]
        if bool(iv.any()):
            grp += [("img", iv), ("text", ~iv)]
        rep["layer_val"] = {nm: {"v1": rel(acc0, m), "qad": rel(acc1, m)} for nm, m in grp}
        log(f"[L{layer}] MoE 出力 (routed の和) の検証 rel-MSE: {rep['layer_val']}")
    log(f"[L{layer}] accepted {n_acc} rejected {n_rej} cold {n_cold}"
        + (f" rejected_img {n_rej_img} img_guarded {n_guard_img}" if bool(img.any()) else "") + f" 時間 {rep['time']}")
    return new, rep


def _train_layer_mode(layer, x, groups, lang, val, cw, teacher, pack, cfg, device, pool, n_exp, img=None):
    """層の MoE 出力 (top-k の和) の誤差で、全 expert の su/sv を一緒に学習する。Q̃ を全 expert 分 GPU に置く。"""
    cdt = cfg.cdt
    n, dim = x.shape
    experts = sorted(groups)
    eid = {e: j for j, e in enumerate(experts)}
    T = torch.zeros(n, dim, device=device)
    qts, su0s, sv0s = [], [[], [], []], [[], [], []]
    trainable = []
    for e, raw_t, raw_p in _prefetch(pool, teacher, pack, layer, experts):
        tok, g = groups[e]
        WT = teacher.weights(raw_t, device, cdt)
        xe = x[tok]
        yT = torch.cat(_batched(lambda s: teacher_expert(xe[s], WT, g[s], cfg.limit, cfg.teacher_act_quant, cdt),
                                xe.shape[0], cfg.eval_batch))
        T.index_add_(0, tok, yT.float())
        qt, su0, sv0 = _expert_tensors(raw_p, device, cdt, pack.qtilde_fn)
        qts.append(qt)
        for i in range(3):
            su0s[i].append(su0[i])
            sv0s[i].append(sv0[i])
        ntr = int((~val[tok]).sum())
        trainable.append(ntr >= cfg.min_tokens)
        del WT, yT, xe
    T = T.to(torch.bfloat16)
    bank = ScaleBank([torch.cat(s) for s in su0s], [torch.cat(s) for s in sv0s], qts, cfg.limit, cdt).to(device)
    mask = torch.tensor(trainable, device=device)
    if cfg.init_diff:
        d0, _ = st_load(cfg.init_diff)
        with torch.no_grad():
            for e in experts:
                j = eid[e]
                for i, p in enumerate(PARTS):
                    for kind, dl, base in (("suh", bank.du, f"su0_{i}"), ("svh", bank.dv, f"sv0_{i}")):
                        nm = ename(layer, e, p, kind)
                        if nm in d0:
                            b0 = getattr(bank, base)[j]
                            dl[i][j] = torch.where(b0 != 0, d0[nm].to(device).float() / torch.where(b0 != 0, b0, torch.ones_like(b0)) - 1, 0.0)
    # 1 batch ごとに expert を束ねて forward
    tok_of = {e: groups[e][0] for e in experts}
    g_of = {e: groups[e][1] for e in experts}
    pos_in = torch.full((n,), -1, dtype=torch.long, device=device)

    def moe_student(tok_sel):
        """tok_sel のトークンについて routed 和を返す (自動微分つき)。"""
        pos_in[tok_sel] = torch.arange(tok_sel.numel(), device=device)
        ps, ys = [], []
        for e in experts:
            tk = tok_of[e]
            p = pos_in[tk]
            m = p >= 0
            if not m.any():
                continue
            ps.append(p[m])
            ys.append(bank(eid[e], x[tk[m]], g_of[e][m]))
        pos_in[tok_sel] = -1
        Y = torch.zeros(tok_sel.numel(), dim, device=device)
        return Y.index_add(0, torch.cat(ps), torch.cat(ys)) if ps else Y

    vtok = torch.nonzero(val).view(-1)
    ttok = torch.nonzero(~val).view(-1)
    den = (cw[ttok] * T[ttok].float().pow(2).sum(-1)).mean().clamp_min(1e-30)

    def evaluate(m=None):
        with torch.no_grad():
            sel = vtok if m is None else vtok[m[vtok]]
            if sel.numel() == 0:
                return float("nan")
            num = dd = 0.0
            for a in range(0, sel.numel(), cfg.layer_batch):
                s = sel[a:a + cfg.layer_batch]
                nu, de = _rel_err(T[s], moe_student(s), cw[s])
                num, dd = num + nu, dd + de
            return (num / dd.clamp_min(1e-30)).item()

    init = evaluate()
    init_ja, init_other = evaluate(lang), evaluate(~lang)
    has_img = img is not None and bool(img[vtok].any())
    init_img = evaluate(img) if has_img else None
    best, best_d, bad = init, bank.get_delta(), 0
    opt = torch.optim.Adam(bank.parameters(), lr=cfg.lr)
    steps = max(1, cfg.epochs * math.ceil(ttok.numel() / cfg.layer_batch))
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    perm = ttok[torch.randperm(ttok.numel(), device=device, generator=gen)]
    pos, done = 0, 0
    log(f"[L{layer}] layer mode: experts {len(experts)} trainable {int(mask.sum())} steps {steps} init val {init:.5f}")
    for st in range(steps):
        if pos >= perm.numel():
            perm = ttok[torch.randperm(ttok.numel(), device=device, generator=gen)]
            pos = 0
        b = perm[pos:pos + cfg.layer_batch]
        pos += cfg.layer_batch
        Y = moe_student(b)
        loss = (cw[b] * (T[b].float() - Y).pow(2).sum(-1)).mean() / den + cfg.l2 * bank.reg(mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in bank.parameters():
            if p.grad is not None:
                p.grad.mul_(mask.float()[:, None])
        opt.step()
        done = st + 1
        if done % cfg.eval_every == 0 or done == steps:
            v = evaluate()
            log(f"[L{layer}] step {done}/{steps} loss {loss.item():.5f} val {v:.5f}")
            if v < best * (1 - cfg.es_tol):
                best, best_d, bad = v, bank.get_delta(), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    bank.set_delta(best_d)
    bank.round_fp16_()
    fin = evaluate()
    rep = {"layer_val": {"weighted": {"v1": init, "qad": fin}, "ja": {"v1": init_ja, "qad": evaluate(lang)},
                         "other": {"v1": init_other, "qad": evaluate(~lang)}},
           "best_val": best, "steps": done, "trainable": int(mask.sum()), "experts_seen": len(experts)}
    img_bad = False
    if has_img:  # 層全体での画像の守り (expert mode の守りと同じ基準)
        fin_img = evaluate(img)
        rep["layer_val"]["img"] = {"v1": init_img, "qad": fin_img}
        img_bad = not (fin_img <= init_img * (1 + float(getattr(cfg, "img_guard_tol", 0.0))))
    new = {}
    if fin < init and img_bad:
        rep["status"] = "rejected_img"
    elif fin < init:
        rep["status"] = "accepted"
        for e in experts:
            j = eid[e]
            if not trainable[j]:
                continue
            for p, (su, sv) in zip(PARTS, bank.export(j)):
                new[ename(layer, e, p, "suh")] = su.cpu()
                new[ename(layer, e, p, "svh")] = sv.cpu()
    else:
        rep["status"] = "rejected"
    log(f"[L{layer}] layer mode {rep['status']}: {rep['layer_val']}")
    return new, rep


# =========================================================================== apply (パックへの反映)
def _link_tree(src, dst, skip):
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in files:
            r = os.path.normpath(os.path.join(rel, f))
            if r in skip:
                continue
            os.link(os.path.join(root, f), os.path.join(dst, r))


def base_digest(read_bytes, keys):
    """差分が置き換える suh/svh の「元の値」の指紋 (sha256)。train 時の --pack と apply 時の --pack が
    同じ値を持つことを確かめる (v0 や別の L2 出力に当てると、学習していない base に絶対値を上書きしてしまうため)。"""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(keys):
        h.update(k.encode())
        h.update(b"\0")
        h.update(read_bytes(k))
    return h.hexdigest()


def _tensor_bytes(t):
    return t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""


def _pack_reader(pack, wm):
    """index の weight_map に従って pack から 1 テンソルの生バイトを読む関数を返す (overlay にも対応)。"""
    hdrs = {}

    def read(k):
        fn = wm[k]
        if fn not in hdrs:
            hdrs[fn] = st_header(os.path.join(pack, fn))
        hdr, base = hdrs[fn]
        a, b = hdr[k]["data_offsets"]
        with open(os.path.join(pack, fn), "rb") as f:
            f.seek(base + a)
            return f.read(b - a)

    return read


def _collect_diffs(diff_dir):
    files = sorted(os.path.join(diff_dir, f) for f in os.listdir(diff_dir)
                   if f.startswith("qad_l2_L") and f.endswith(".safetensors"))
    if not files:
        raise SystemExit(f"{diff_dir} に qad_l2_L*.safetensors が無い")
    return files


def apply_pack(pack, diff_dir, out, mode, force_base=False):
    """mode=shards: 差分を含む層の shard だけ実体コピーして suh/svh のバイト範囲を書き換え、他は hardlink。
    mode=overlay: 全ファイルを hardlink し、差分ファイルを追加して index.json だけ書き直す (ハーネス専用)。
    元のパック (v1、assemble_v1.sh で work/shards と hardlink 共有) には一切書き込まない。
    差分の metadata に base_sha256 (train 時の pack の元の値の指紋) があれば、この pack の値と照合する。"""
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} が空でない")
    if mode == "shards" and os.path.exists(os.path.join(pack, "QAD_OVERLAY_HARNESS_ONLY")):
        # overlay の weight_map は zz- を指すので、shards にしても zz- の複製が書き換わるだけで ds41rt 用にならない
        raise SystemExit(f"{pack} は overlay (ハーネス専用)。shards は実体の pack (v1 など) に全差分を当てて作る")
    idx_path = os.path.join(pack, "model.safetensors.index.json")
    index = json.load(open(idx_path))
    wm = index["weight_map"]
    diffs = _collect_diffs(diff_dir)
    by_shard = {}
    loaded = {}
    read = _pack_reader(pack, wm)
    for df in diffs:
        tens, md = st_load(df)
        loaded[df] = tens
        missing = [k for k in tens if k not in wm]
        if missing:
            raise SystemExit(f"{df}: {missing[0]} が pack の index に無い")
        want = md.get("base_sha256")
        if want is None:
            log(f"[apply] 注意: {os.path.basename(df)} に base_sha256 が無い (古い形式)。base の照合をしない")
        elif tens and base_digest(read, tens) != want:
            msg = (f"{df}: この pack の suh/svh は学習時の --pack ({md.get('pack', '?')}) と違う。"
                   f"別の base に絶対値を上書きすることになる (--force-base で無視)")
            if not force_base:
                raise SystemExit(msg)
            log("[apply] 警告: " + msg)
        for k, t in tens.items():
            if not (k.endswith(".suh") or k.endswith(".svh")):
                raise SystemExit(f"{df}: suh/svh 以外のキー {k}")
            if k not in wm:
                raise SystemExit(f"{df}: {k} が pack の index に無い")
            if t.dtype != torch.float16 or not torch.isfinite(t.float()).all():
                raise SystemExit(f"{df}: {k} が fp16 でないか非有限値を含む")
            by_shard.setdefault(wm[k], {})[k] = t
    os.makedirs(out, exist_ok=True)
    manifest = {"pack": pack, "mode": mode, "diffs": diffs, "patched": {}}
    if mode == "shards":
        _link_tree(pack, out, skip=set(by_shard))
        for shard, tens in sorted(by_shard.items()):
            src = os.path.join(pack, shard)
            dst = os.path.join(out, shard)
            tmp = dst + ".qadtmp"
            shutil.copyfile(src, tmp)  # 新しい inode (ext4 に reflink は無い)
            assert not os.path.samefile(src, tmp)
            hdr, base = st_header(tmp)
            with open(tmp, "r+b") as f:
                for k, t in sorted(tens.items()):
                    h = hdr[k]
                    if h["dtype"] != "F16" or list(h["shape"]) != list(t.shape):
                        raise SystemExit(f"{shard}:{k} の dtype/shape が違う {h} vs {tuple(t.shape)}")
                    a, b = h["data_offsets"]
                    raw = t.contiguous().view(torch.uint8).numpy().tobytes()
                    assert len(raw) == b - a
                    f.seek(base + a)
                    f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            with open(tmp, "rb") as f:  # 読み戻し確認
                for k, t in tens.items():
                    a, b = hdr[k]["data_offsets"]
                    f.seek(base + a)
                    assert f.read(b - a) == t.contiguous().view(torch.uint8).numpy().tobytes(), k
            os.replace(tmp, dst)
            manifest["patched"][shard] = len(tens)
            log(f"[apply] {shard}: {len(tens)} tensors 書き換え")
    elif mode == "overlay":
        _link_tree(pack, out, skip={"model.safetensors.index.json"})
        new_index = json.loads(json.dumps(index))
        for df in diffs:
            name = "zz-" + os.path.basename(df)  # TensorIndex はファイル名順に読み後勝ち (stream_ref.py:50, 58-62)
            dst = os.path.join(out, name)
            if os.path.lexists(dst):
                # pack 自体が overlay のとき、同名の zz- は _link_tree が元 pack と hardlink した物。
                # そのまま copyfile すると同じ inode (= 元 pack の差分) を上書きするので、先に外す。
                os.unlink(dst)
            shutil.copyfile(df, dst)
            assert not os.path.samefile(df, dst)
            for k in loaded[df]:
                new_index["weight_map"][k] = name
            manifest["patched"][name] = len(loaded[df])
        with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
            json.dump(new_index, f, indent=2)
        with open(os.path.join(out, "QAD_OVERLAY_HARNESS_ONLY"), "w") as f:
            f.write("stream_ref.py --exl3-pack 専用。ds41rt / stage_hf.sh には使わない (shard の重複キーと連番違反)。\n")
    else:
        raise SystemExit(f"mode {mode}")
    with open(out.rstrip("/") + ".qad_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"[apply] {mode}: {sum(manifest['patched'].values())} tensors -> {out}")
    return manifest


# =========================================================================== GPU 確認用
def check_decomp(pack_dir, layer, experts, device="cuda"):
    from exllamav3.modules.quant.exl3 import LinearEXL3
    pk = PackEXL3(pack_dir)
    for e in experts:
        raw = pk.raw(layer, e)
        for p in PARTS:
            d = {k: v.to(device) for k, v in raw[p].items()}
            tr = d["trellis"]
            in_f, out_f = tr.shape[0] * 16, tr.shape[1] * 16
            w_ref = LinearEXL3(None, in_f, out_f, suh=d["suh"], svh=d["svh"], trellis=tr, mcg=d.get("mcg")).get_weight_tensor().float()
            q = qtilde_exl3(tr, d.get("mcg"), device).float()
            w_fac = d["suh"].float()[:, None] * q * d["svh"].float()[None, :]
            rel = ((w_fac - w_ref).norm() / w_ref.norm()).item()
            print(f"L{layer} e{e} {p}: |W_fac - W_ref|/|W_ref| = {rel:.3e}  max abs {(w_fac - w_ref).abs().max().item():.3e}")


def check_teacher(ckpt, capture, layer, experts, n_tok=512, device="cuda"):
    os.environ.setdefault("DSV41_REF", "/ref")
    sys.path.insert(0, HERE)
    import stream_ref as SR  # kernel (tilelang) が要る
    torch.set_default_dtype(torch.bfloat16)
    ti = SR.TensorIndex(ckpt)
    data = load_capture(capture, layer) if os.path.isdir(capture[0]) else load_layer_files(capture)
    x = data["x"][:n_tok].to(device, torch.bfloat16)
    g = data["weights"][:n_tok, :1].to(device).float()
    tch = TeacherFP4(ckpt)
    limit = json.load(open(os.path.join(ckpt, "inference", "config.json"))).get("swiglu_limit", 10.0)
    for e in experts:
        Wk = []
        for p in PARTS:
            w = ti.load(ename(layer, e, p, "weight")).to(device)
            s = ti.load(ename(layer, e, p, "scale")).to(device)
            Wk += [w.view(SR.FP4X2).view(w.shape[0], w.shape[1]), s]
        y_k = SR.expert_forward(x, tuple(Wk), g, limit).float()
        y_t = teacher_expert(x, TeacherFP4.weights(tch.raw(layer, e), device, torch.bfloat16), g, limit, True).float()
        rel = ((y_t - y_k).norm() / y_k.norm()).item()
        print(f"L{layer} e{e}: torch 教師 vs fp4_gemm  rel {rel:.3e}  max abs {(y_t - y_k).abs().max().item():.3e}")


# =========================================================================== CLI
def parse_layers(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--capture", action="append", default=[], help="capture_moe.py の記録ディレクトリ (part ごとに繰り返す)")
    t.add_argument("--layer-file", action="append", default=[], help="代わりに .pt の層ファイル。{L} を層番号に置換")
    t.add_argument("--no-kp-weight", action="store_true", help="capture の 1/kp 重要度重みを使わない")
    t.add_argument("--layers", required=True, help="例 0-19 / 3,5")
    t.add_argument("--orig-ckpt", required=True, help="原本 (MXFP4 experts)")
    t.add_argument("--pack", required=True, help="学生パック (v1)")
    t.add_argument("--out", required=True)
    t.add_argument("--loss", choices=("expert", "layer"), default="expert")
    t.add_argument("--ja-weight", type=float, default=2.0)
    t.add_argument("--img-weight", type=float, default=1.0,
                   help="画像トークン (capture の tt >= 0) の損失の重み。画像なしの記録では無関係。推奨 1.5 (r6_vision_qad.md)")
    t.add_argument("--img-guard-tol", type=float, default=0.0,
                   help="画像だけの検証誤差が v1 の (1+tol) 倍を超える expert / 層は採らない")
    t.add_argument("--min-val-img", type=int, default=32, help="画像の守りを掛ける最小の検証対数 (expert ごと)")
    t.add_argument("--val-frac-img", type=float, default=0.25,
                   help="画像を含む系列の検証割合 (画像なしの記録では無関係)。0.1 だと 24 本中 2 本で守りが効かない")
    t.add_argument("--lang-ja-value", type=int, default=1)
    t.add_argument("--lr", type=float, default=5e-4, help="δ (相対変化) の Adam lr。QuIP# §F.6 の su/sv と同じ")
    t.add_argument("--l2", type=float, default=0.1, help="mean(δ²) の係数 (v1 からのずれ)")
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--batch", type=int, default=1024, help="expert mode の 1 step の (token, expert) 数")
    t.add_argument("--layer-batch", type=int, default=16384, help="layer mode の 1 step のトークン数")
    t.add_argument("--eval-batch", type=int, default=16384)
    t.add_argument("--min-steps", type=int, default=50)
    t.add_argument("--max-steps", type=int, default=400)
    t.add_argument("--eval-every", type=int, default=10)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--es-tol", type=float, default=1e-4)
    t.add_argument("--val-frac", type=float, default=0.1)
    t.add_argument("--min-tokens", type=int, default=256, help="学習用トークンがこれ未満の expert は学習しない (cold)")
    t.add_argument("--min-val", type=int, default=16)
    t.add_argument("--ja-expert-ratio", type=float, default=4.0, help="日本語比率が層平均の何倍以上なら日本語誤差で早期終了")
    t.add_argument("--min-val-ja", type=int, default=32)
    t.add_argument("--n-experts", type=int, default=384)
    t.add_argument("--swiglu-limit", type=float, default=None, help="既定は orig-ckpt/inference/config.json")
    t.add_argument("--no-teacher-act-quant", action="store_true", help="教師の活性 FP8 量子化を再現しない")
    t.add_argument("--init-diff", default="", help="layer mode の初期値 (expert mode の出力)。{L} 置換可")
    t.add_argument("--no-layer-report", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--skip-existing", action="store_true")
    t.add_argument("--verbose", action="store_true")
    a2 = sub.add_parser("apply")
    a2.add_argument("--pack", required=True)
    a2.add_argument("--diff", required=True)
    a2.add_argument("--out", required=True)
    a2.add_argument("--mode", choices=("overlay", "shards"), default="overlay")
    a2.add_argument("--force-base", action="store_true", help="差分の base_sha256 が pack と合わなくても書く")
    es = sub.add_parser("estimate")
    es.add_argument("--tokens", type=int, default=600000)
    es.add_argument("--loss", choices=("expert", "layer"), default="expert")
    es.add_argument("--counts-from", default="", help="教師 .pt (counts) から expert ごとの偏りを入れる")
    cd = sub.add_parser("check-decomp")
    cd.add_argument("--pack", required=True)
    cd.add_argument("--layer", type=int, default=0)
    cd.add_argument("--experts", default="0,7")
    ct = sub.add_parser("check-teacher")
    ct.add_argument("--orig-ckpt", required=True)
    ct.add_argument("--capture", action="append", required=True, help="記録ディレクトリ (か .pt)")
    ct.add_argument("--layer", type=int, default=0)
    ct.add_argument("--experts", default="0,7")
    a = ap.parse_args(argv)

    if a.cmd == "estimate":
        counts = None
        if a.counts_from:
            c = torch.load(a.counts_from, weights_only=False)["counts"]
            arr = torch.stack([torch.as_tensor(v, dtype=torch.float64) for v in c.values()]).mean(0)
            counts = (arr / arr.sum() * a.tokens * 6).tolist()
        print(json.dumps(estimate(a.tokens, counts=counts, loss=a.loss), indent=2))
        return
    if a.cmd == "apply":
        apply_pack(a.pack, a.diff, a.out, a.mode, force_base=a.force_base)
        return
    if a.cmd == "check-decomp":
        check_decomp(a.pack, a.layer, [int(v) for v in a.experts.split(",")])
        return
    if a.cmd == "check-teacher":
        check_teacher(a.orig_ckpt, a.capture, a.layer, [int(v) for v in a.experts.split(",")])
        return

    cfg = Cfg()
    limit = a.swiglu_limit
    if limit is None:
        limit = json.load(open(os.path.join(a.orig_ckpt, "inference", "config.json"))).get("swiglu_limit", 0.0)
    for k, v in dict(loss=a.loss, ja_weight=a.ja_weight, img_weight=a.img_weight, img_guard_tol=a.img_guard_tol,
                     min_val_img=a.min_val_img, val_frac_img=a.val_frac_img, lr=a.lr, l2=a.l2, epochs=a.epochs, batch=a.batch,
                     layer_batch=a.layer_batch, eval_batch=a.eval_batch, min_steps=a.min_steps, max_steps=a.max_steps,
                     eval_every=a.eval_every, patience=a.patience, es_tol=a.es_tol, val_frac=a.val_frac,
                     min_tokens=a.min_tokens, min_val=a.min_val, ja_expert_ratio=a.ja_expert_ratio,
                     min_val_ja=a.min_val_ja, n_experts=a.n_experts, limit=float(limit),
                     teacher_act_quant=not a.no_teacher_act_quant, layer_report=not a.no_layer_report, seed=a.seed,
                     verbose=a.verbose, init_diff="", cdt=torch.bfloat16 if a.device.startswith("cuda") else torch.float32).items():
        setattr(cfg, k, v)
    os.makedirs(a.out, exist_ok=True)
    teacher = TeacherFP4(a.orig_ckpt)
    pack = PackEXL3(a.pack)
    for L in parse_layers(a.layers):
        outp = os.path.join(a.out, f"qad_l2_L{L:02d}.safetensors")
        if a.skip_existing and os.path.exists(outp):
            log(f"[L{L}] skip (既存)")
            continue
        cfg.init_diff = a.init_diff.format(L=L) if a.init_diff else ""
        if a.capture:
            data = load_capture(a.capture, L, use_kp=not a.no_kp_weight)
        elif a.layer_file:
            data = load_layer_files([p.format(L=L) for p in a.layer_file], a.lang_ja_value)
        else:
            raise SystemExit("--capture か --layer-file が要る")
        if any(not s.startswith("student") for s in data["stream"]):
            log(f"[L{L}] 注意: 入力 stream が {data['stream']} (方式 A は学生 stream を推奨、SPEC §2.5)")
        new, rep = train_layer(L, data, teacher, pack, cfg, a.device)
        md = {"qad_l2": json.dumps({k: v for k, v in rep.items() if k != "experts"}, ensure_ascii=False),
              "pack": os.path.abspath(a.pack),
              "base_sha256": base_digest(lambda k: _tensor_bytes(pack.t.load(k)), new)}
        st_save(outp, new, md)
        with open(os.path.join(a.out, f"qad_l2_L{L:02d}.json"), "w") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False, default=str)
        log(f"[L{L}] -> {outp} ({len(new)} tensors, {rep['seconds']:.0f}s)")
        del data
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
