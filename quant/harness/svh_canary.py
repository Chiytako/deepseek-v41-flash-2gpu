#!/usr/bin/env python3
"""ds41rt で routed expert の svh が clamp の前に掛かっているかを確かめる「カナリア差分」を作る (SPEC §7-2、QAD_PLAN.md §1.4)。

層 L の全 expert で up (w3) の svh を c 倍、down (w2) の suh を 1/c 倍にした差分 qad_l2_L{LL}.safetensors を書く。
  - 正しい順 (up = clamp((x⊙su)·Q̃·sv, ±10)、h = silu(gate)·up·w、down の入力に suh): up が c 倍されてから clamp に
    掛かるので、|up| > 10/c の位置で出力が変わる。c=64 なら層 L の出力ははっきり変わる。
  - 誤った順 (svh を clamp の後で掛ける、または down の suh と相殺される位置で掛ける): c と 1/c が打ち消し合い、
    出力は v1 と fp16 の丸め程度しか変わらない。
差分は qad_l2.py apply --mode shards|overlay でそのまま当てられる (base_sha256 付き)。運用には絶対に使わない。

使い方 (CPU で可、torch が要る):
  python3 svh_canary.py --pack /pack --layer 0 --c 64 --out /w/canary_L00
  python3 qad_l2.py apply --pack /pack --diff /w/canary_L00 --out /models/v1-canary --mode shards
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qad_l2 as Q  # noqa: E402


def make_canary(pack, layer, c, n_experts=None):
    wm = json.load(open(os.path.join(pack, "model.safetensors.index.json")))["weight_map"]
    read = Q._pack_reader(pack, wm)
    n = n_experts or 1 + max(int(k.split(".")[4]) for k in wm if k.startswith(f"layers.{layer}.ffn.experts."))
    new = {}
    for e in range(n):
        for part, kind, f in (("w3", "svh", c), ("w2", "suh", 1.0 / c)):
            k = Q.ename(layer, e, part, kind)
            raw = bytearray(read(k))
            t = torch.frombuffer(raw, dtype=torch.float16).clone()
            v = (t.float() * f).half()
            if not torch.isfinite(v.float()).all():
                raise SystemExit(f"{k}: c={c} で fp16 が溢れる")
            nz = t != 0
            if (v[nz] == 0).any():
                raise SystemExit(f"{k}: c={c} で 0 に潰れる値がある (c を小さくする)")
            new[k] = v
    md = {"svh_canary": json.dumps({"layer": layer, "c": c, "note": "ds41rt の svh 位置の検査専用。運用に使わない"}),
          "pack": os.path.abspath(pack),
          "base_sha256": Q.base_digest(read, new)}
    return new, md


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--c", type=float, default=64.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} が空でない")
    os.makedirs(a.out, exist_ok=True)
    new, md = make_canary(a.pack, a.layer, a.c)
    p = os.path.join(a.out, f"qad_l2_L{a.layer:02d}.safetensors")
    Q.st_save(p, new, md)
    print(f"[canary] {len(new)} tensors (layer {a.layer}, c={a.c}) -> {p}")


if __name__ == "__main__":
    main()
