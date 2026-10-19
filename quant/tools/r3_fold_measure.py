#!/usr/bin/env python3
"""r3_fold_measure.py: 1017 の routed experts に「折りたたみ」(専門家間の共有) の余地がどれだけあるかを CPU で測る。

GPU は使わない。1 層ぶんの packed FP4 (約 7.2GB) を RAM に置き、float32 には少しずつ戻して計算する (ピーク約 12GB)。

測るもの (層ごと、w1 / w3 / w2 それぞれ):
  A. 共有成分: 全 expert 平均 M を引いた残差のエネルギー比 (Σ||W_e-M||² / Σ||W_e||²)、
     expert ごとに最適スケール α_e を掛けた M を引いた場合も。
  B. 平坦化した expert 間 cos 類似度の分布 (384x384 Gram)。
  C. hidden 側 (w1/w3 は入力 5120、w2 は出力 5120) の共有共分散 C = Σ_e (hidden 側の外積)。
     固有値スペクトル、top-r の累積エネルギー、ガウス近似での変換符号化利得
     (対角スケーリングのみ / 128 ブロック KLT / 512 ブロック KLT / 全 5120 KLT)。
  D. expert 単体の特異値 (残差 W_e - α_e M の低ランク性。D2-MoE 型の delta 低ランク化の可否)。
  E. ニューロン単位 (w1 行 | w3 行 | w2 列 の 15360 次元) の最近傍 |cos|。
     ランダムに選んだ問い合わせニューロンについて、他 expert の全 2304x383 ニューロンの中での最大 |cos|。
  F. 量子化シミュレーション (スカラー Lloyd-Max、EXL3 と同じ前処理 = 行列両側の RMS スケール + 128 Hadamard):
     F1 直接 2/3bit、F2 平均を引いた残差を 2bit、F3 共有 KLT + 整数ビット配分 (平均 2.0bit)、F4 128 ブロック KLT + 配分。

使い方:
  OPENBLAS_NUM_THREADS=8 uv run --no-project --with numpy python -I r3_fold_measure.py --layer 20 --out out.json
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse, json, math, os, struct, sys, time
import numpy as np

CKPT = MODELS + "/DeepSeek-V4.1-Flash-1017"
HEADERS = DSV41 + "/ref/headers.json"
E = 384
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)
# 1 byte -> 2 値 (低ニブルが偶数要素)。convert.py:31-34 と同じ
LUT2 = np.stack([FP4[np.arange(256) & 15], FP4[np.arange(256) >> 4]], 1).astype(np.float32)  # [256,2]
E8M0 = np.array([2.0 ** (i - 127) if i < 255 else np.nan for i in range(256)], np.float32)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- 読み込み
def locate(layer):
    h = json.load(open(HEADERS))
    for f, d in h.items():
        if f"layers.{layer}.ffn.experts.0.w1.weight" in d:
            return f, d
    raise SystemExit(f"layer {layer} not found")


def load_layer(layer):
    """packed codes と scale を [E, N, K/2] uint8 / [E, N, K/32] uint8 で返す。"""
    fname, hd = locate(layer)
    path = os.path.join(CKPT, fname)
    with open(path, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for part in ("w1", "w3", "w2"):
        k0 = f"layers.{layer}.ffn.experts.0.{part}"
        N, Kh = hd[k0 + ".weight"]["shape"]
        Ks = hd[k0 + ".scale"]["shape"][1]
        codes = np.empty((E, N, Kh), np.uint8)
        scales = np.empty((E, N, Ks), np.uint8)
        for e in range(E):
            for kind, arr in (("weight", codes), ("scale", scales)):
                m = hd[f"layers.{layer}.ffn.experts.{e}.{part}.{kind}"]
                a, b = m["data_offsets"]
                arr[e] = np.asarray(mm[base + a: base + b]).reshape(arr.shape[1:])
        out[part] = (codes, scales)
        log(f"loaded {part} {codes.shape} from {fname}")
    return fname, out


def deq(codes, scales):
    """codes [..., N, K/2], scales [..., N, K/32] -> float32 [..., N, K]"""
    v = LUT2[codes].reshape(*codes.shape[:-1], codes.shape[-1] * 2)
    s = E8M0[scales]
    v = v.reshape(*scales.shape, 32) * s[..., None]
    return v.reshape(*codes.shape[:-1], codes.shape[-1] * 2)


def expert(L, part, e):
    c, s = L[part]
    return deq(c[e], s[e])


def hidden_major(W, part):
    """hidden(5120) 側を列にした [rows, 5120] を返す。w1/w3 は [2304,5120] のまま、w2 [5120,2304] は転置。"""
    return W if part in ("w1", "w3") else W.T


# ---------------------------------------------------------------- 符号化利得
def logdet_psd(M):
    w = np.linalg.eigvalsh(M.astype(np.float64))
    w = np.clip(w, 1e-30, None)
    return float(np.log2(w).sum()), w


def coding_gain_bits(C):
    """ガウス近似の高レート変換符号化利得 (bit/重み)。対角正規化を基準に、ブロック KLT で何 bit 得するか。
    利得 = 0.5 * mean(log2 diag) - 0.5 * logdet(block)/n。"""
    n = C.shape[0]
    d = np.diag(C).astype(np.float64)
    res = {}
    am = d.mean()
    res["diag_vs_flat"] = 0.5 * (math.log2(am) - np.log2(d).mean())  # 入力チャネルの分散のばらつき (EXL3 の su で既に取れる分)
    for bs in (32, 128, 512, n):
        tot = 0.0
        for i in range(0, n, bs):
            ld, _ = logdet_psd(C[i:i + bs, i:i + bs])
            tot += ld
        res[f"klt{bs}_vs_diag"] = 0.5 * (np.log2(d).sum() - tot) / n
    return res


def reverse_waterfill(lam, R):
    """平均 R bit/成分 のガウス逆注水。lam は分散。D(R)/mean(lam) を返す。"""
    lam = np.asarray(lam, np.float64)
    lo, hi = 1e-30, lam.max()
    for _ in range(200):
        th = math.sqrt(lo * hi)
        r = np.maximum(0, 0.5 * np.log2(lam / th)).mean()
        if r > R:
            lo = th
        else:
            hi = th
    D = np.minimum(lam, th).mean()
    return D / lam.mean()


# ---------------------------------------------------------------- 量子化器 (スカラー Lloyd-Max + EXL3 風前処理)
def lloyd_max(b, iters=300):
    if b == 0:
        return np.zeros(1), 1.0
    x = np.linspace(-8, 8, 160001)
    p = np.exp(-x * x / 2); p /= p.sum()
    k = 2 ** b
    c = np.quantile(np.random.default_rng(0).standard_normal(200000), (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        t = (c[1:] + c[:-1]) / 2
        idx = np.searchsorted(t, x)
        c = np.bincount(idx, x * p, k) / np.maximum(np.bincount(idx, p, k), 1e-300)
    t = (c[1:] + c[:-1]) / 2
    idx = np.searchsorted(t, x)
    D = ((x - c[idx]) ** 2 * p).sum()
    return c, D


LM = {b: lloyd_max(b) for b in range(0, 7)}


def q_scalar(X, b):
    """単位分散を仮定したスカラー量子化 (b=0 は 0)。"""
    if b == 0:
        return np.zeros_like(X)
    c, _ = LM[b]
    t = (c[1:] + c[:-1]) / 2
    return c[np.searchsorted(t, X)].astype(np.float32)


def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / math.sqrt(n)).astype(np.float32)


H128 = hadamard(128)
RNG = np.random.default_rng(1234)


def had_cols(X, sign):
    """列方向 (axis=1) に 128 ブロックの符号付き Hadamard"""
    r, n = X.shape
    return ((X * sign).reshape(r, n // 128, 128) @ H128).reshape(r, n)


def had_cols_inv(X, sign):
    r, n = X.shape
    return ((X.reshape(r, n // 128, 128) @ H128.T).reshape(r, n)) * sign


def exl3_like(W, b, g_search=True):
    """EXL3 regularize と同じ流れ (出力/入力チャネル RMS、両側 128 Hadamard) にスカラー量子化を当てる。"""
    r, n = W.shape
    so = np.sqrt((W ** 2).mean(1, keepdims=True)) + 1e-30
    X = W / so
    sr = RNG.choice([-1.0, 1.0], r).astype(np.float32)
    X = had_cols(X.T, sr).T  # 行側 Hadamard
    si = np.sqrt((X ** 2).mean(0, keepdims=True)) + 1e-30
    X = X / si
    sc = RNG.choice([-1.0, 1.0], n).astype(np.float32)
    X = had_cols(X, sc)
    best = None
    for g in ([0.85, 0.9, 0.95, 1.0, 1.05, 1.1] if g_search else [1.0]):
        Q = q_scalar(X / g, b) * g
        err = ((Q - X) ** 2).mean()
        if best is None or err < best[0]:
            best = (err, Q)
    Q = best[1]
    Q = had_cols_inv(Q, sc) * si
    Q = had_cols_inv(Q.T, sr).T * so
    return Q


def allocate_bits(var, R, bmax=6):
    """var: 成分 (列) ごとの分散。平均 R bit の整数配分を、Lloyd-Max の歪み表で貪欲に決める。"""
    n = len(var)
    b = np.zeros(n, int)
    D = np.array([LM[k][1] for k in range(bmax + 1)])
    budget = int(round(R * n))
    gain = var * (D[0] - D[1])
    import heapq
    h = [(-gain[i], i) for i in range(n)]
    heapq.heapify(h)
    for _ in range(budget):
        g, i = heapq.heappop(h)
        b[i] += 1
        if b[i] < bmax:
            heapq.heappush(h, (-(var[i] * (D[b[i]] - D[b[i] + 1])), i))
    return b


def klt_alloc_quant(Wh, V, R):
    """Wh: [rows, 5120] (hidden 側が列)。V: [5120,5120] 直交 (列が基底)。
    Y = Wh V の列ごとに分散を見て整数ビット配分 → 行側 Hadamard でガウス化 → スカラー量子化。"""
    r, n = Wh.shape
    so = np.sqrt((Wh ** 2).mean(1, keepdims=True)) + 1e-30
    Y = (Wh / so) @ V
    sr = RNG.choice([-1.0, 1.0], r).astype(np.float32)
    Y = had_cols(Y.T, sr).T
    var = (Y ** 2).mean(0)
    bits = allocate_bits(var, R)
    sd = np.sqrt(var) + 1e-30
    Z = Y / sd
    Q = np.empty_like(Z)
    for k in np.unique(bits):
        m = bits == k
        Q[:, m] = q_scalar(Z[:, m], int(k))
    Q = Q * sd
    Q = had_cols_inv(Q.T, sr).T
    Q = (Q @ V.T) * so
    return Q, np.bincount(bits, minlength=7).tolist()


def relerr(A, B):
    return float(np.sqrt(((A - B) ** 2).sum() / (B ** 2).sum()))


# ---------------------------------------------------------------- 本体
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nq", type=int, default=768, help="ニューロン最近傍の問い合わせ数")
    ap.add_argument("--nsim", type=int, default=6, help="量子化シミュレーションの expert 数")
    ap.add_argument("--cov-stride", type=int, default=1, help="共分散に使う expert の間引き")
    a = ap.parse_args()
    t0 = time.time()
    fname, L = load_layer(a.layer)
    R = {"layer": a.layer, "shard": fname, "parts": {}}
    rng = np.random.default_rng(a.layer)
    sim_ids = sorted(rng.choice(E, a.nsim, replace=False).tolist())
    Cs = {}
    means = {}
    for part in ("w1", "w3", "w2"):
        codes, scales = L[part]
        N, K = codes.shape[1], codes.shape[2] * 2
        P = {}
        # ---- A/B: Gram (行ブロックごと)
        G = np.zeros((E, E), np.float64)
        Msum = np.zeros((N, K), np.float64)
        rb = 64
        for i in range(0, N, rb):
            X = deq(codes[:, i:i + rb], scales[:, i:i + rb]).reshape(E, -1)
            G += (X @ X.T).astype(np.float64)
            Msum[i:i + rb] = X.reshape(E, -1, K).sum(0)
        M = (Msum / E).astype(np.float32)
        means[part] = M
        nrm2 = np.diag(G).copy()
        tot = nrm2.sum()
        Mn2 = float((M.astype(np.float64) ** 2).sum())
        P["energy_mean_component"] = E * Mn2 / tot
        P["resid_ratio_mean"] = (tot - E * Mn2) / tot
        # α_e 最適: <W_e, M> = (G 1)_e / E
        dotM = G.sum(1) / E
        P["resid_ratio_scaled_mean"] = float((tot - (dotM ** 2 / Mn2).sum()) / tot)
        cos = G / np.sqrt(np.outer(nrm2, nrm2))
        iu = np.triu_indices(E, 1)
        cv = cos[iu]
        P["cos_flat"] = {"mean": float(cv.mean()), "std": float(cv.std()), "p01": float(np.quantile(cv, 0.01)),
                         "p50": float(np.quantile(cv, 0.5)), "p99": float(np.quantile(cv, 0.99)),
                         "max": float(cv.max()), "min": float(cv.min()),
                         "frac_gt_0.1": float((cv > 0.1).mean()), "frac_gt_0.3": float((cv > 0.3).mean())}
        top = np.argsort(-cv)[:5]
        P["cos_flat_top_pairs"] = [[int(iu[0][j]), int(iu[1][j]), float(cv[j])] for j in top]
        rel_norm = np.sqrt(nrm2 / nrm2.mean())
        P["expert_norm_rel"] = {"min": float(rel_norm.min()), "p50": float(np.median(rel_norm)),
                                "max": float(rel_norm.max()), "argmax": int(rel_norm.argmax())}
        log(part, "A/B done", P["resid_ratio_mean"], P["cos_flat"]["p50"], P["cos_flat"]["max"])
        # ---- C: hidden 側共分散
        C = np.zeros((5120, 5120), np.float64)
        for e in range(0, E, a.cov_stride):
            Wh = hidden_major(expert(L, part, e), part)
            C += (Wh.T @ Wh).astype(np.float64)
        Cs[part] = C
        lam = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
        cum = np.cumsum(lam) / lam.sum()
        P["pooled_spectrum"] = {f"top{r}": float(cum[r - 1]) for r in (16, 64, 256, 512, 1024, 2048, 3072, 4096)}
        pr = lam / lam.sum()
        P["pooled_eff_rank_entropy"] = float(np.exp(-(pr[pr > 0] * np.log(pr[pr > 0])).sum()))
        P["pooled_lam_max_over_mean"] = float(lam[0] / lam.mean())
        P["coding_gain_bits"] = coding_gain_bits(C)
        lamn = lam / lam.mean()
        P["waterfill_D_at_2bit"] = {"flat": 2 ** -4, "pooled_klt": reverse_waterfill(lamn, 2.0)}
        P["waterfill_D_at_3bit"] = {"flat": 2 ** -6, "pooled_klt": reverse_waterfill(lamn, 3.0)}
        log(part, "C done", P["coding_gain_bits"], P["pooled_spectrum"]["top256"])
        # ---- D: expert 単体 (残差) の特異値
        dsv = []
        for e in sim_ids:
            W = expert(L, part, e)
            al = dotM[e] / Mn2
            Rm = (W - al * M).astype(np.float32)
            small = Rm @ Rm.T if Rm.shape[0] <= Rm.shape[1] else Rm.T @ Rm
            s2 = np.clip(np.linalg.eigvalsh(small.astype(np.float64))[::-1], 0, None)
            c2 = np.cumsum(s2) / s2.sum()
            dsv.append({f"top{r}": float(c2[r - 1]) for r in (64, 256, 512, 1024)})
        P["resid_lowrank_energy"] = {k: float(np.mean([d[k] for d in dsv])) for k in dsv[0]}
        R["parts"][part] = P
        log(part, "D done", P["resid_lowrank_energy"])
    # ---- w1/w3 は同じ入力 x を共有する。合算した入力側共分散でも利得を出す
    Cin = Cs["w1"] + Cs["w3"]
    R["w13_input_coding_gain_bits"] = coding_gain_bits(Cin)
    # ---- E: ニューロン最近傍
    def neurons(e):
        a1 = expert(L, "w1", e); a3 = expert(L, "w3", e); a2 = expert(L, "w2", e).T
        X = np.concatenate([a1, a3, a2], 1)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    qe = rng.choice(E, a.nq); qi = rng.choice(2304, a.nq)
    Q = np.empty((a.nq, 15360), np.float32)
    for e in np.unique(qe):
        Ne = neurons(e)
        Q[qe == e] = Ne[qi[qe == e]]
    best_x = np.zeros(a.nq, np.float32); best_s = np.zeros(a.nq, np.float32)
    for f in range(E):
        S = np.abs(Q @ neurons(f).T)
        same = qe == f
        if same.any():
            Ss = S[same].copy(); Ss[np.arange(same.sum()), qi[same]] = 0
            best_s[same] = np.maximum(best_s[same], Ss.max(1))
        Sx = S[~same].max(1)
        best_x[~same] = np.maximum(best_x[~same], Sx)
    # 基準: 同じ共分散構造を持つ「独立な」ニューロンでの期待値は解析で近似 (等方なら sqrt(2 ln(2N)/d))
    iso = math.sqrt(2 * math.log(2 * 2304 * 383) / 15360)
    def qs(v):
        return {"p50": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "p99": float(np.quantile(v, 0.99)),
                "max": float(v.max()), "frac_gt_0.5": float((v > 0.5).mean()), "frac_gt_0.8": float((v > 0.8).mean())}
    R["neuron_nn_cross_expert"] = qs(best_x)
    R["neuron_nn_same_expert"] = qs(best_s)
    R["neuron_nn_iso_baseline"] = iso
    R["neuron_nn_resid_energy_if_merged"] = float(np.mean(1 - best_x ** 2))
    log("E done", R["neuron_nn_cross_expert"])
    # ---- F: 量子化シミュレーション
    sim = {}
    for part in ("w1", "w3", "w2"):
        C = Cs[part]
        lam, V = np.linalg.eigh(C)
        V = V[:, ::-1].astype(np.float32).copy()
        Vb = np.zeros((5120, 5120), np.float32)
        for i in range(0, 5120, 128):
            _, vb = np.linalg.eigh(C[i:i + 128, i:i + 128])
            Vb[i:i + 128, i:i + 128] = vb.astype(np.float32)
        M = means[part]
        rows = []
        for e in sim_ids:
            W = expert(L, part, e)
            Wh = hidden_major(W, part)
            Mh = hidden_major(M, part)
            al = float((Wh * Mh).sum() / (Mh ** 2).sum())
            r = {"e": e}
            for b in (2, 3):
                r[f"direct_{b}b"] = relerr(exl3_like(Wh, b), Wh)
            Rm = Wh - al * Mh
            Mq = exl3_like(Mh, 4)
            r["meanfold_2b"] = relerr(exl3_like(Wh - al * Mq, 2) + al * Mq, Wh)
            Qk, hist = klt_alloc_quant(Wh, V, 2.0)
            r["klt5120_alloc_2b"] = relerr(Qk, Wh); r["klt5120_bits_hist"] = hist
            Qb, hist2 = klt_alloc_quant(Wh, Vb, 2.0)
            r["klt128_alloc_2b"] = relerr(Qb, Wh); r["klt128_bits_hist"] = hist2
            Qi, _ = klt_alloc_quant(Wh, np.eye(5120, dtype=np.float32), 2.0)
            r["diag_alloc_2b"] = relerr(Qi, Wh)
            rows.append(r)
            log("F", part, r)
        keys = [k for k in rows[0] if k != "e" and not k.endswith("hist")]
        sim[part] = {"per_expert": rows, "mean": {k: float(np.mean([x[k] for x in rows])) for k in keys}}
    R["quant_sim"] = sim
    R["lloyd_max_D"] = {b: LM[b][1] for b in LM}
    R["elapsed_s"] = time.time() - t0
    json.dump(R, open(a.out, "w"), indent=1)
    log("wrote", a.out, f"{R['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
