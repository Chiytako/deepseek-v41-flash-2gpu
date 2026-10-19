"""層 0 の routing を CPU で近似し、画像トークンとテキストの専門家の使い方を比べる (GPU 不要)。

近似: 層 0 の MoE 入力 = ffn_norm(hc_pre(X0 + attn 寄与)) のうち、attention の寄与を落として ffn_norm(e) とする。
  X0 は埋め込み e の hc_mult 複製で、hc_pre と comb (Sinkhorn で行和 1) は複製を同じ向きのまま保つので、
  RMSNorm の後は ffn_norm(e) に一致する (model.py:959-996、attention を除く)。
  - テキスト: e = embed[token]。agent24 / ja16 の全トークンで数え、ハーネスの実測 counts (層 0) と比べて近似の妥当性を見る。
  - 画像: e = aligner(ViT(patches)) (IMAGE) と学習済みの image_start / image_end / image_newline。bias_vl で選ぶ
    (model.py:820-822)。比較のため、同じ入力をテキスト用 bias で選んだ反実仮想も数える。
ViT は BF16 原重みを fp32 で CPU 実行する (vision.py)。前処理は ref/image_processor.py の load_image そのまま。

usage (dsv41-quant イメージ、CPU だけ):
  python3 r6_vl_proxy0.py --images /w/img --jsonl agent=/t/agent24.jsonl --jsonl ja=/t/ja16.jsonl \
     --counts agent=/t/agent24_teacher.pt --counts ja=/t/ja16_teacher.pt --bias-json /w/r6_vl_bias.json --out /w/r6_vl_proxy0.json
"""
import argparse
import glob
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

REF = os.environ.get("DSV41_REF", "/ref")
CKPT = os.environ.get("DSV41_CKPT", "/model")
sys.path.insert(0, REF)
N_EXP, TOPK = 384, 6


class Weights:
    def __init__(self, ckpt):
        self.ckpt = ckpt
        self.wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
        self.h = {}

    def get(self, name):
        fn = self.wm[name]
        if fn not in self.h:
            self.h[fn] = safe_open(os.path.join(self.ckpt, fn), "pt")
        return self.h[fn].get_tensor(name)


def model_args():
    cfg = json.load(open(os.path.join(CKPT, "inference", "config.json")))
    a = types.SimpleNamespace(**cfg)
    defaults = dict(vision_rope_theta=10000.0, vision_max_wh_ratio=None, gate_temp=1.0, norm_eps=1e-20)
    for k, v in defaults.items():
        if not hasattr(a, k):
            setattr(a, k, v)
    a.vision_enabled = a.vision_n_layers > 0
    return a


def build_vision(W, args):
    from vision import Aligner, ViT

    vit, al = ViT(args).float(), Aligner(args).float()
    for prefix, mod in (("vision.", vit), ("aligner.", al)):
        sd = mod.state_dict()
        for k in sd:
            sd[k] = W.get(prefix + k).float()
        mod.load_state_dict(sd)
    return vit.eval(), al.eval()


def rmsnorm(x, w, eps):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w.float()


def route(x, gw, bias, gate_temp):
    s = F.softplus((x @ gw.T) / gate_temp).sqrt()
    idx = (s + bias).topk(TOPK, dim=-1)[1]
    return torch.bincount(idx.flatten(), minlength=N_EXP).double().numpy()


def stats(c):
    s = c / c.sum()
    srt = np.sort(s)[::-1]
    asc = np.sort(s)
    n = len(s)
    gini = float(((2 * np.arange(1, n + 1) - n - 1) * asc).sum() / (n * asc.sum()))
    return {"top25": float(srt[:96].sum()), "bot30": float(srt[-115:].sum()), "gini": gini, "zero": int((c == 0).sum())}


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def topset(c, k):
    return set(np.argsort(-c)[:k].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="画像のディレクトリ (jpg/jpeg/png)")
    ap.add_argument("--jsonl", action="append", default=[], help="label=path (ids を持つ jsonl)")
    ap.add_argument("--counts", action="append", default=[], help="label=path (teacher .pt の counts)")
    ap.add_argument("--bias-json", default="", help="r6_vl_bias.py の出力 (up/down 集合を使う)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--k3-frac", type=float, default=0.193)
    ap.add_argument("--mix", type=float, default=0.3)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    args = model_args()
    W = Weights(CKPT)
    L = a.layer
    assert L == 0, "attention を落とす近似は層 0 でしか成り立たない"
    gw = W.get(f"layers.{L}.ffn.gate.weight").float()
    b = W.get(f"layers.{L}.ffn.gate.bias").float()
    bv = W.get(f"layers.{L}.ffn.gate.bias_vl").float()
    fw = W.get(f"layers.{L}.ffn_norm.weight").float()
    eps = args.norm_eps
    rep = {"layer": L, "approx": "ffn_norm(embedding), attention skipped", "text": {}, "images": {}}

    # ---- テキスト: 近似 vs 実測
    emb = W.get("embed.weight")  # bf16 [V, D]
    measured = {}
    for spec in a.counts:
        lab, p = spec.split("=", 1)
        measured[lab] = np.asarray(torch.load(p, weights_only=False, map_location="cpu")["counts"][L], dtype=np.float64)
    proxy_text = {}
    for spec in a.jsonl:
        lab, p = spec.split("=", 1)
        ids = [i for l in open(p) for i in json.loads(l)["ids"]]
        c = np.zeros(N_EXP)
        for s0 in range(0, len(ids), 8192):
            x = rmsnorm(emb[torch.tensor(ids[s0:s0 + 8192])], fw, eps)
            c += route(x, gw, b, args.gate_temp)
        proxy_text[lab] = c
        r = {"ntok": len(ids), "proxy": stats(c)}
        if lab in measured:
            m = measured[lab]
            r["measured"] = stats(m)
            r["cos_proxy_measured"] = cos(c, m)
            r["spearman_proxy_measured"] = spearman(c, m)
            r["top96_overlap"] = len(topset(c, 96) & topset(m, 96))
        rep["text"][lab] = r
        print(f"[text] {lab}: {json.dumps(r)}", flush=True)
    if len(measured) >= 2:
        labs = list(measured)
        rep["text"]["measured_cos_" + "_".join(labs)] = cos(measured[labs[0]], measured[labs[1]])

    # ---- 画像
    from image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, image_token_types, load_image

    vit, al = build_vision(W, args)
    delim = {IMAGE_START: W.get("image_start").float(), IMAGE_END: W.get("image_end").float(),
             IMAGE_NEW_LINE: W.get("image_newline").float()}
    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    img_vl = np.zeros(N_EXP)
    img_txtbias = np.zeros(N_EXP)
    per_img = {}
    for f in files:
        t0 = time.time()
        patches, nh, nw, lh, lw = load_image({"url": f}, args)
        with torch.inference_mode():
            feats = al(vit(patches.float(), nh, nw), nh, nw)  # [lh*lw, D]
        tt = image_token_types(lh, lw)
        span = torch.empty(len(tt), feats.size(1))
        span[tt == IMAGE] = feats
        for k, v in delim.items():
            span[tt == k] = v
        x = rmsnorm(span, fw, eps)
        cv = route(x, gw, bv, args.gate_temp)
        ct = route(x, gw, b, args.gate_temp)
        img_vl += cv
        img_txtbias += ct
        per_img[os.path.basename(f)] = {"tokens": len(tt), "grid": [lh, lw], "vl": stats(cv),
                                        "cos_vl_vs_textbias": cos(cv, ct), "seconds": round(time.time() - t0, 1)}
        print(f"[image] {os.path.basename(f)} {len(tt)} tok grid {lh}x{lw} {per_img[os.path.basename(f)]}", flush=True)
    rep["images"]["per_image"] = per_img
    rep["images"]["all_vl"] = stats(img_vl)
    rep["images"]["all_textbias"] = stats(img_txtbias)
    rep["images"]["cos_vl_vs_textbias"] = cos(img_vl, img_txtbias)
    pv = img_vl / img_vl.sum()
    uni = np.full(N_EXP, 1 / N_EXP)
    rep["images"]["cos_vl_vs_uniform"] = cos(pv, uni)
    # 画像どうしの一致 (1 枚ごとの分布が似ているか)
    # テキストとの比較
    cmp = {}
    for lab, m in measured.items():
        s = m / m.sum()
        cold = set(np.argsort(s)[:115].tolist())
        cmp[lab] = {
            "cos_img_vs_text_measured": cos(pv, s),
            "cos_img_vs_text_proxy": cos(pv, proxy_text[lab] / proxy_text[lab].sum()) if lab in proxy_text else None,
            "img_top96_in_text_top96": len(topset(pv, 96) & topset(s, 96)),
            "img_share_on_text_cold115": float(pv[list(cold)].sum()),
            "text_share_on_text_cold115": float(s[list(cold)].sum()),
            "img_share_on_text_top96": float(pv[list(topset(s, 96))].sum()),
        }
    if "agent" in measured and "ja" in measured:
        sa, sj = measured["agent"] / measured["agent"].sum(), measured["ja"] / measured["ja"].sum()
        k = int(round(a.k3_frac * N_EXP))

        def mask(imp):
            imp = (1 - a.mix) * imp + a.mix / N_EXP
            mk = np.zeros(N_EXP, bool)
            mk[np.argsort(-imp)[:k]] = True
            return mk

        for name, imp in {"text(agent.7,ja.3)": 0.7 * sa + 0.3 * sj,
                          "text+img(.5,.3,.2)": 0.5 * sa + 0.3 * sj + 0.2 * pv}.items():
            mk = mask(imp)
            cmp.setdefault("k3_cover", {})[name] = {"agent": float(sa[mk].sum()), "ja": float(sj[mk].sum()),
                                                    "img": float(pv[mk].sum())}
    if a.bias_json:
        bj = json.load(open(a.bias_json))["layers"][str(L)]
        up, down = bj["up"], bj["down"]
        n = len(up)
        cmp["updown"] = {"n": n, "uniform_share": n / N_EXP,
                         "img_vl_share_up": float(pv[up].sum()), "img_vl_share_down": float(pv[down].sum()),
                         "img_textbias_share_up": float((img_txtbias / img_txtbias.sum())[up].sum()),
                         "img_textbias_share_down": float((img_txtbias / img_txtbias.sum())[down].sum())}
    rep["images"]["vs_text"] = cmp
    rep["images"]["top_experts_vl"] = [[int(e), float(pv[e])] for e in np.argsort(-pv)[:20]]
    print("[images]", json.dumps({k: v for k, v in rep["images"].items() if k != "per_image"}, indent=1), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
