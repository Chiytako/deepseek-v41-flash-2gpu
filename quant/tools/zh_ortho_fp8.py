"""重みの直交化 (W' = W - r r^T W) を FP8 (e4m3、32x32 ブロック、UE8M0 scale) に戻したとき、
方向 r の成分がどれだけ残るかを CPU で測る。torch 不要 (numpy のみ)。

方向 r の候補 (GPU なしで作れる代理):
  head : 出力側の「中国語 − 日本語」方向。γ ⊙ (mean head[簡体字トークン] − mean head[日本語で使うトークン])
  embed: 入力側の同じ差。mean embed[簡体字] − mean embed[日本語で使う]
  rand : 乱数方向 (対照)
本番の r は harness/steer_collect.py で取る残差の平均差になる。ここでは「FP8 に戻すと直交化が消えるか」
という形式側の性質だけを見る。

usage: tools/.venv/bin/python -I tools/zh_ortho_fp8.py --vocab work/r5/zh_vocab.json --layers 5,20,35  (--vocab は zh_vocab_r5.py の出力)
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import json
import os
import time

import numpy as np

CKPT = os.environ.get("DSV41_CKPT", MODELS + "/DeepSeek-V4.1-Flash-1017")
FP8_MAX = 448.0


# ----------------------------------------------------------------- safetensors (memmap)
class ST:
    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.map = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
        self.hdr = {}

    def _h(self, fname):
        if fname not in self.hdr:
            with open(os.path.join(self.ckpt, fname), "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.hdr[fname] = (json.loads(f.read(n)), 8 + n)
        return self.hdr[fname]

    def raw(self, name):
        fname = self.map[name]
        h, base = self._h(fname)
        v = h[name]
        a, b = v["data_offsets"]
        dt = {"BF16": np.uint16, "F8_E4M3": np.uint8, "F8_E8M0": np.uint8, "F32": np.float32}[v["dtype"]]
        mm = np.memmap(os.path.join(self.ckpt, fname), dtype=np.uint8, mode="r", offset=base + a, shape=(b - a,))
        return mm.view(dt).reshape(v["shape"]), v["dtype"]


def bf16(u16):
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


def _e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 0xF, b & 7
        if (b & 0x7F) == 0x7F:
            t[b] = np.nan
            continue
        v = (m / 8.0) * 2.0**-6 if e == 0 else (1 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = -v if s else v
    return t


E4M3 = _e4m3_table()
GRID = np.unique(E4M3[np.isfinite(E4M3)])  # 昇順の表現可能値 (±0 は 1 個)


def deq_fp8(w_u8, s_u8, blk=32):
    w = E4M3[w_u8]
    sc = np.exp2(s_u8.astype(np.float32) - 127.0)
    return w * np.repeat(np.repeat(sc, blk, 0), blk, 1)[: w.shape[0], : w.shape[1]]


def block_scale(w, blk=32):
    o, i = w.shape
    a = np.abs(w).reshape(o // blk, blk, i // blk, blk).max(axis=(1, 3))
    e = np.ceil(np.log2(np.maximum(a, 1e-30) / FP8_MAX))
    return np.exp2(e).astype(np.float32)


def rtn(x):
    """x を e4m3 の格子に最近接丸め。返り値 (丸め値, もう一方の隣接値)。"""
    x = np.clip(x, -FP8_MAX, FP8_MAX)
    k = np.clip(np.searchsorted(GRID, x), 1, len(GRID) - 1)
    lo, hi = GRID[k - 1], GRID[k]
    near = np.where(x - lo <= hi - x, lo, hi)
    other = np.where(near == lo, hi, lo)
    return near, other


def requant(w, blk=32, r=None, iters=0):
    """RTN で FP8 に戻す。r と iters>0 を渡すと、列ごとに r^T w_q が 0 に近づくよう一部の丸め方向を反転する。"""
    sc = block_scale(w, blk)
    S = np.repeat(np.repeat(sc, blk, 0), blk, 1)
    x = w / S
    q, other = rtn(x)
    if r is None or iters == 0:
        return q * S
    rr = r[:, None].astype(np.float32)
    for _ in range(iters):
        wq = q * S
        e = (rr * (wq - w)).sum(0)  # 列ごとの r 成分の誤差 (目標は 0、w は直交化済み)
        d = (other - q) * S  # 反転したときの値の変化
        gain = -rr * d * np.sign(e)[None, :]  # 正なら |e| を減らす
        cost = ((other * S - w) ** 2 - (q * S - w) ** 2)
        ok = gain > 0
        ratio = np.where(ok, cost / np.maximum(gain, 1e-30), np.inf)
        order = np.argsort(ratio, axis=0)
        g_sorted = np.take_along_axis(np.where(ok, gain, 0.0), order, 0)
        cum = np.cumsum(g_sorted, 0)
        take_sorted = (cum - g_sorted) < np.abs(e)[None, :]
        take = np.zeros_like(take_sorted)
        np.put_along_axis(take, order, take_sorted & np.take_along_axis(ok, order, 0), 0)
        q_new = np.where(take, other, q)
        other = np.where(take, q, other)
        q = q_new
    return q * S


def unit(v):
    v = v.astype(np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--layers", default="5,20,35")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    st = ST(CKPT)
    voc = json.load(open(a.vocab))
    zh = np.array(voc["ids"]["simpl"])
    # 日本語側: 仮名トークン全部 + 日本語回答で使われた漢字トークン (kanji_unseen_multi の補集合のうち頻出側)
    kanji = np.array(voc["ids"]["kanji"])
    unseen = set(voc.get("kanji_unseen_multi", []))
    ja = np.concatenate([np.array(voc["ids"]["kana"]), np.array([i for i in kanji if i not in unseen])])
    t0 = time.time()
    head, _ = st.raw("head.weight")
    emb, _ = st.raw("embed.weight")
    gam = bf16(st.raw("norm.weight")[0])
    hz, hj = bf16(head[np.sort(zh)]).mean(0), bf16(head[np.sort(ja)]).mean(0)
    ez, ej = bf16(emb[np.sort(zh)]).mean(0), bf16(emb[np.sort(ja)]).mean(0)
    dirs = {"head": unit(gam * (hz - hj)), "embed": unit(ez - ej),
            "rand": unit(np.random.default_rng(0).standard_normal(5120))}
    rep = {"n_zh": int(len(zh)), "n_ja": int(len(ja)),
           "cos_head_embed": float(dirs["head"] @ dirs["embed"]),
           "cos_head_raw_vs_gamma": float(unit(hz - hj) @ dirs["head"]), "layers": {}}
    print(f"[dirs] {time.time() - t0:.1f}s zh={len(zh)} ja={len(ja)} cos(head,embed)={rep['cos_head_embed']:.3f}", flush=True)
    for L in [int(x) for x in a.layers.split(",")]:
        for tname in (f"layers.{L}.attn.wo_b", f"layers.{L}.ffn.shared_experts.w2"):
            wu, _ = st.raw(tname + ".weight")
            su, _ = st.raw(tname + ".scale")
            W = deq_fp8(np.asarray(wu), np.asarray(su))
            # 検算: 元の W を同じ規則で FP8 に戻すと元に一致するか
            same = float(np.mean(requant(W) == W))
            res = {"requant_identity": same}
            for dn, r in dirs.items():
                p = r @ W  # [in]
                coln = np.linalg.norm(W, axis=0)
                cos = np.abs(p) / np.maximum(coln, 1e-30)
                Wp = W - np.outer(r, p)
                q0 = requant(Wp)
                q1 = requant(Wp, r=r, iters=a.iters)
                base = np.linalg.norm(p)
                res[dn] = {
                    "col_cos_mean": float(cos.mean()), "col_cos_p99": float(np.quantile(cos, 0.99)),
                    "proj_rel_frob": float(base / np.linalg.norm(W)),
                    "retained_rtn": float(np.linalg.norm(r @ q0) / base),
                    "retained_fix": float(np.linalg.norm(r @ q1) / base),
                    "err_rtn_vs_W'": float(np.linalg.norm(q0 - Wp) / np.linalg.norm(Wp)),
                    "err_fix_vs_W'": float(np.linalg.norm(q1 - Wp) / np.linalg.norm(Wp)),
                    "delta_rel": float(np.linalg.norm(Wp - W) / np.linalg.norm(W)),
                }
            rep["layers"][tname] = res
            print(tname, json.dumps(res), f"{time.time() - t0:.0f}s", flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
