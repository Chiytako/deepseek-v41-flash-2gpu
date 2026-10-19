"""量子化のどこが中国語方向の確率を増やすかを、層 (と expert) の部分差し替えで切り分ける (r5_zh_mixing.md §4)。

stream_ref.py の ExpertStore._is_exl3 を包み、EXL3 パックの experts を「指定した層 (と expert) だけ」使う。
それ以外の層は原重み (MXFP4) のまま。teacher() を学生採点 (--teacher-ref) で回すので、出力 .pt は
stream_ref.py --teacher-out と同じ形式になり、tools/zh_teacher_mass.py と KL の集計にそのまま使える。

GPU 版 (LIL イメージ内):
  python3 harness/steer_zhbisect.py --exl3-pack /packs/v0 --layers 0-9 \
      --teacher-in work/teacher/ja16.jsonl --teacher-ref work/teacher/ja16_teacher.pt \
      --teacher-out work/zh/bisect/ja16_v0_L0-9.pt
  # expert 単位: --experts-json '{"12": [3, 77], "13": "all"}'  (層 -> expert 番号の並びか "all")
CPU 版 (指定の解釈だけを確かめる): tools/.venv/bin/python -I harness/steer_zhbisect.py --self-test --layers 0-9,36-39
所要の目安: ja16 は 1 回あたり教師と同程度 (15〜25 分、v0 の採点は experts の読み込みが 705 秒)。
"""
import argparse
import json
import os
import sys


def parse_layers(spec):
    out = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def parse_experts(js):
    if not js:
        return None
    d = json.loads(js)
    return {int(k): (None if v == "all" else set(int(x) for x in v)) for k, v in d.items()}


def make_filter(layers, experts):
    def ok(layer_id, e):
        if experts is not None and layer_id in experts:
            s = experts[layer_id]
            return s is None or e in s
        return layer_id in layers
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exl3-pack", default="")
    ap.add_argument("--layers", default="", help="EXL3 を使う層 (例: 0-9,36-39)。空なら --experts-json だけ")
    ap.add_argument("--experts-json", default="")
    ap.add_argument("--teacher-in", default="")
    ap.add_argument("--teacher-ref", default="")
    ap.add_argument("--teacher-out", default="")
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--expert-budget-gb", type=float, default=60)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    layers = parse_layers(a.layers)
    experts = parse_experts(a.experts_json)
    ok = make_filter(layers, experts)
    if a.self_test:
        sel = [(l, e) for l in range(40) for e in range(384) if ok(l, e)]
        print(f"[self-test] layers={sorted(layers)} experts={ {k: (sorted(v) if v else 'all') for k, v in (experts or {}).items()} } "
              f"-> EXL3 を使う (layer, expert) {len(sel)} 個 / 15360")
        return 0
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stream_ref as S

    orig = S.ExpertStore._is_exl3

    def _is_exl3(self, layer_id, e):
        return orig(self, layer_id, e) and ok(layer_id, e)

    S.ExpertStore._is_exl3 = _is_exl3
    model, tok, args = S.build({"max_batch_size": 1, "max_seq_len": a.max_seq_len, "temperature": 0.0},
                               a.expert_budget_gb, exl3_pack=a.exl3_pack)
    S._patch_head()
    seqs = [json.loads(l) for l in open(a.teacher_in)]
    ref = torch.load(a.teacher_ref, weights_only=False)
    os.makedirs(os.path.dirname(a.teacher_out) or ".", exist_ok=True)
    print(f"[bisect] EXL3 layers={sorted(layers)} experts_json={a.experts_json or '-'}", flush=True)
    S.teacher(model, seqs, a.teacher_out, ref=ref)
    print("[experts]", S.STORE.stats, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
