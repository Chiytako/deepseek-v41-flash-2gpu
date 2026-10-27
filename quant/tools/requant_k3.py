"""選んだ routed expert を 1017 原本 (MXFP4) から K3 (3.0bpw、MCG) で符号化し直す。

符号化は build/qrecipe/tools/requant_full.py の encode と同じ (exllamav3 1.5.1 の Viterbi + refit_scales(H=I))。
K だけ 3 にする。結果は層ごとに OUT/layer_LL.safetensors (w1/w3/w2 の trellis/suh/svh/mcg) と
OUT/layer_LL.json (テンソルごとの相対誤差。比較用に abl の K2 の誤差も入れる) に書き、層単位で再開できる。
パックへの組み込みは tools/assemble_k3.py。

入力 --select: tools/vl/select_k3.py の出力 ({"ranked": {"<layer>": [expert, ...]}, ...})。各層の先頭 --per-layer 個を符号化する。
dsv41-quant:151 の中で動かす (build/ocr_k3.sh quant)。
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/recipe/tools")

K = 3
CODEBOOK = "mcg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="1017 原本 (MXFP4)")
    ap.add_argument("--base", required=True, help="比較に使う K2 パック (abl の snapshot)")
    ap.add_argument("--select", required=True)
    ap.add_argument("--per-layer", type=int, default=40)
    ap.add_argument("--layers", default="", help="例 0-19 (既定は全層)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-chunk", type=int, default=4608)
    a = ap.parse_args()

    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from safetensors import safe_open
    from safetensors.torch import save_file

    from quantize_experts_exl3 import _dequant_t, _load_index, _quantize_fast
    from requant_full import tensor_seed
    from requant_probe import refit_identity, relerr

    dev = torch.device("cuda:0")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ranked = {int(k): v for k, v in json.load(open(a.select))["ranked"].items()}
    layers = sorted(ranked)
    if a.layers:
        lo, hi = map(int, a.layers.split("-"))
        layers = [l for l in layers if lo <= l <= hi]
    src_map = _load_index(Path(a.src))["weight_map"]
    base_map = json.load(open(Path(a.base) / "model.safetensors.index.json"))["weight_map"]
    h_cache: dict = {}

    def recon(trellis, suh, svh, shape, k):
        o = torch.empty(shape, dtype=torch.half, device=dev)
        ext.reconstruct_had_slice(o, trellis.to(dev), suh.to(dev).half(), svh.to(dev).half(), k, True, False, 0)
        return o

    for layer in layers:
        done = out / f"layer_{layer:02d}.json"
        if done.is_file():
            print(f"skip layer {layer}", flush=True)
            continue
        t0 = time.time()
        tensors, rows = {}, []
        for e in ranked[layer][: a.per_layer]:
            for part in ("w1", "w3", "w2"):
                stem = f"layers.{layer}.ffn.experts.{e}.{part}"
                with safe_open(str(Path(a.src) / src_map[stem + ".weight"]), framework="pt") as fh:
                    w = _dequant_t(fh.get_tensor(stem + ".weight"), fh.get_tensor(stem + ".scale"), "cuda:0").to(dev)
                torch.manual_seed(tensor_seed(stem))
                enc = _quantize_fast([w.clone()], K, "cuda:0", h_cache, codebook=CODEBOOK, tile_chunk=a.tile_chunk)[0]
                trellis, su, sv = enc["trellis"], enc["suh"].to(dev), enc["svh"].to(dev)
                q = recon(trellis, su, sv, w.shape, K)
                e_vit = relerr(q, w)
                _, r, c = refit_identity(w, q.float())
                su_f, sv_f = (su.float() * r).half(), (sv.float() * c).half()
                e_fit = relerr(recon(trellis, su_f, sv_f, w.shape, K), w)
                keep = math.isfinite(e_fit) and e_fit <= e_vit
                if keep:
                    su, sv = su_f, sv_f
                with safe_open(str(Path(a.base) / base_map[stem + ".trellis"]), framework="pt") as bf:
                    b = {x: bf.get_tensor(f"{stem}.{x}") for x in ("trellis", "suh", "svh")}
                e_base = relerr(recon(b["trellis"], b["suh"], b["svh"], w.shape, 2), w)
                tensors[stem + ".trellis"] = trellis.contiguous()
                tensors[stem + ".suh"] = su.cpu().contiguous()
                tensors[stem + ".svh"] = sv.cpu().contiguous()
                tensors[stem + "." + CODEBOOK] = enc[CODEBOOK]
                rows.append({"tensor": stem, "k3": e_fit if keep else e_vit, "k2_base": e_base, "refit": keep})
                del w, q, enc
        tmp = out / f".layer_{layer:02d}.safetensors.tmp"
        save_file(tensors, str(tmp))
        os.replace(tmp, out / f"layer_{layer:02d}.safetensors")
        k3 = sum(r["k3"] for r in rows) / len(rows)
        k2 = sum(r["k2_base"] for r in rows) / len(rows)
        done.write_text(json.dumps({"layer": layer, "experts": ranked[layer][: a.per_layer], "k3_mean": k3, "k2_base_mean": k2,
                                    "seconds": time.time() - t0, "rows": rows}, indent=1) + "\n")
        print(f"layer {layer}: {len(rows)} tensors, relerr K3 {k3:.4f} (abl K2 {k2:.4f}), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
