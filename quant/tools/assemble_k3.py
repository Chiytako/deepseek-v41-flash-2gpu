"""K2 のパック (abl) に、tools/requant_k3.py で作った K3 の専門家を組み込んだ snapshot を作る (標準ライブラリのみ)。

- 専門家を差し替える shard だけ書き直す (テンソルの順と名前は元のまま、差し替えた trellis だけ形が [.., .., 48] になる)。
  ほかのファイルは hardlink。index は名前も shard 割り当ても変わらないので hardlink のまま (ds41rt は shard ごとのテンソル数まで照合する)。
- config.json の quantization_config に expert_bits {"<layer>.<expert>": 3} を足す (ds41rt の raw EXL3 読み込みが解釈する)。
- --per-layer で各層の先頭 n 個だけ使う (requant_k3.py はそれ以上作っておけば、量を変えて組み直せる)。

使い方 (root 所有のファイルを hardlink するので sudo):
  sudo python3 tools/assemble_k3.py --base <abl snapshot> --k3 <requant_k3 の OUT> --per-layer 32 --out <新 snapshot>
"""
import argparse
import json
import os
import struct
from pathlib import Path

CHUNK = 1 << 26


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def copy_range(src, dst, off, length):
    src.seek(off)
    while length:
        b = src.read(min(CHUNK, length))
        dst.write(b)
        length -= len(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--k3", required=True)
    ap.add_argument("--per-layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, k3dir, out = Path(a.base), Path(a.k3), Path(a.out)
    assert not out.exists(), f"{out} は既にある"

    # 差し替える専門家と、その K3 テンソルの在りか
    chosen, k3_src = {}, {}
    for j in sorted(k3dir.glob("layer_*.json")):
        rec = json.load(open(j))
        layer, experts = rec["layer"], rec["experts"][: a.per_layer]
        assert len(experts) == a.per_layer, f"layer {layer}: K3 は {len(rec['experts'])} 個しか無い"
        chosen[layer] = experts
        f = k3dir / f"layer_{layer:02d}.safetensors"
        hdr, data0 = read_header(f)
        for e in experts:
            for part in ("w1", "w3", "w2"):
                for kind in ("trellis", "suh", "svh", "mcg"):
                    name = f"layers.{layer}.ffn.experts.{e}.{part}.{kind}"
                    assert name in hdr, f"{name} が {f} に無い"
                    k3_src[name] = (f, data0, hdr[name])

    index = json.load(open(base / "model.safetensors.index.json"))
    shards = sorted({index["weight_map"][n] for n in k3_src})
    out.mkdir(parents=True)
    for p in sorted(base.iterdir()):
        if p.name in shards or p.name == "config.json":
            continue
        if p.is_dir():
            os.system(f"cp -al '{p}' '{out / p.name}'")
        else:
            os.link(p, out / p.name)

    total_delta = 0
    for s in shards:
        hdr, data0 = read_header(base / s)
        meta = hdr.pop("__metadata__", None)
        names = sorted(hdr, key=lambda n: hdr[n]["data_offsets"][0])
        new_hdr, off, plan = {}, 0, []
        for n in names:
            if n in k3_src:
                f, d0, h = k3_src[n]
                assert h["dtype"] == hdr[n]["dtype"], n
                ln = h["data_offsets"][1] - h["data_offsets"][0]
                plan.append((f, d0 + h["data_offsets"][0], ln))
                new_hdr[n] = {"dtype": h["dtype"], "shape": h["shape"], "data_offsets": [off, off + ln]}
            else:
                h = hdr[n]
                ln = h["data_offsets"][1] - h["data_offsets"][0]
                plan.append((base / s, data0 + h["data_offsets"][0], ln))
                new_hdr[n] = {"dtype": h["dtype"], "shape": h["shape"], "data_offsets": [off, off + ln]}
            off += ln
        if meta is not None:
            new_hdr["__metadata__"] = meta
        hb = json.dumps(new_hdr, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        tmp = out / (s + ".tmp")
        handles = {}
        with open(tmp, "wb") as dst:
            dst.write(struct.pack("<Q", len(hb)) + hb)
            for f, o, ln in plan:
                if f not in handles:
                    handles[f] = open(f, "rb")
                copy_range(handles[f], dst, o, ln)
        for h in handles.values():
            h.close()
        chk, d0 = read_header(tmp)
        chk.pop("__metadata__", None)
        assert sorted(chk) == sorted(names) and os.path.getsize(tmp) == d0 + off, f"{s}: 書き込みの検査に失敗"
        os.replace(tmp, out / s)
        delta = os.path.getsize(out / s) - os.path.getsize(base / s)
        total_delta += delta
        print(f"{s}: +{delta / 2**20:.0f} MiB", flush=True)

    cfg = json.load(open(base / "config.json"))
    cfg["quantization_config"]["expert_bits"] = {f"{l}.{e}": 3 for l in sorted(chosen) for e in chosen[l]}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    (out / ".ocr_k3.json").write_text(json.dumps({"base": str(base), "k3": str(k3dir), "per_layer": a.per_layer,
                                                   "experts": len(cfg["quantization_config"]["expert_bits"]),
                                                   "added_bytes": total_delta}, indent=1) + "\n")
    print(f"{len(shards)} shard を書き直し、K3 の専門家 {sum(map(len, chosen.values()))} 個、増分 {total_delta / 2**30:.2f} GiB → {out}")


if __name__ == "__main__":
    main()
