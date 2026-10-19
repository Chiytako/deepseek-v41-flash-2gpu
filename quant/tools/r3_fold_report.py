#!/usr/bin/env python3
"""r3_fold_report.py: r3_fold_measure.py の JSON (層ごと) を Markdown の表にまとめる。

使い方: python3 -I r3_fold_report.py work/r3fold/layer*.json
標準ライブラリだけで動く。
"""
import json, math, sys


def bits_from_mse_ratio(r):
    """同じレートで MSE が r 倍になったときの、ガウス近似での等価ビット差 (正 = 得)。"""
    return 0.5 * math.log2(1 / r) if r > 0 else float("nan")


def main(paths):
    Ls = [json.load(open(p)) for p in paths]
    Ls.sort(key=lambda d: d["layer"])
    E = 384
    print("### A/B: 共有成分と expert 間 cos (平坦化)\n")
    print("| 層 | part | 平均成分のエネルギー比 | 残差比 (平均を引く) | 残差比 (α_e·平均) | cos p50 | cos p99 | cos max | ノルム比 min / max |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in Ls:
        for p, P in d["parts"].items():
            c = P["cos_flat"]; n = P["expert_norm_rel"]
            print(f"| {d['layer']} | {p} | {P['energy_mean_component']:.4f} | {P['resid_ratio_mean']:.4f} | "
                  f"{P['resid_ratio_scaled_mean']:.4f} | {c['p50']:.4f} | {c['p99']:.4f} | {c['max']:.4f} | "
                  f"{n['min']:.2f} / {n['max']:.2f} (e{n['argmax']}) |")
    print(f"\n独立なら平均成分のエネルギー比は 1/{E} = {1/E:.4f}、残差比は {1-1/E:.4f}。\n")
    print("### C: hidden 側 (5120) の共有共分散\n")
    print("| 層 | part | top64 | top256 | top1024 | top2048 | 有効ランク | KLT128 利得 | KLT512 利得 | KLT5120 利得 | 逆注水 D(2bit)/平坦 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in Ls:
        for p, P in d["parts"].items():
            s = P["pooled_spectrum"]; g = P["coding_gain_bits"]; w = P["waterfill_D_at_2bit"]
            print(f"| {d['layer']} | {p} | {s['top64']:.3f} | {s['top256']:.3f} | {s['top1024']:.3f} | {s['top2048']:.3f} | "
                  f"{P['pooled_eff_rank_entropy']:.0f} | {g['klt128_vs_diag']:.4f} | {g['klt512_vs_diag']:.4f} | "
                  f"{g['klt5120_vs_diag']:.4f} | {w['pooled_klt']/w['flat']:.3f} |")
    print("\n等方なら topR は R/5120 (top256 = 0.050, top1024 = 0.200)、有効ランクは 5120。利得の単位は bit/重み。\n")
    print("### D: expert 単体 (α_e·平均を引いた残差) の特異値エネルギー\n")
    print("| 層 | part | top64 | top256 | top512 | top1024 |")
    print("|---|---|---|---|---|---|")
    for d in Ls:
        for p, P in d["parts"].items():
            r = P["resid_lowrank_energy"]
            print(f"| {d['layer']} | {p} | {r['top64']:.3f} | {r['top256']:.3f} | {r['top512']:.3f} | {r['top1024']:.3f} |")
    print("\n参考: 同じ形 (2304x5120) の i.i.d. ガウス行列は 0.072 / 0.255 / 0.450 / 0.726。\n")
    print("### E: ニューロン単位 (w1 行 | w3 行 | w2 列、15360 次元) の最近傍 |cos|\n")
    print("| 層 | 他 expert 内 p50 | p99 | max | >0.5 の割合 | 同じ expert 内 p50 | max | 等方基準 | 統合時の残差エネルギー |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in Ls:
        x = d["neuron_nn_cross_expert"]; s = d["neuron_nn_same_expert"]
        print(f"| {d['layer']} | {x['p50']:.3f} | {x['p99']:.3f} | {x['max']:.3f} | {x['frac_gt_0.5']:.3f} | "
              f"{s['p50']:.3f} | {s['max']:.3f} | {d['neuron_nn_iso_baseline']:.3f} | {d['neuron_nn_resid_energy_if_merged']:.3f} |")
    print("\n### F: 量子化シミュレーション (スカラー Lloyd-Max、相対誤差 = ||Q-W||/||W||、6 expert 平均)\n")
    print("| 層 | part | 直接 2bit | 直接 3bit | 平均 fold + 残差 2bit | 共有 KLT5120 + 配分 | KLT128 + 配分 | 対角 + 配分 | KLT5120 の等価ビット得 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in Ls:
        for p, S in d["quant_sim"].items():
            m = S["mean"]
            gain = bits_from_mse_ratio((m["klt5120_alloc_2b"] / m["direct_2b"]) ** 2)
            print(f"| {d['layer']} | {p} | {m['direct_2b']:.4f} | {m['direct_3b']:.4f} | {m['meanfold_2b']:.4f} | "
                  f"{m['klt5120_alloc_2b']:.4f} | {m['klt128_alloc_2b']:.4f} | {m['diag_alloc_2b']:.4f} | {gain:+.4f} |")
    lm = Ls[0]["lloyd_max_D"]
    print(f"\nLloyd-Max (ガウス) の MSE: 2bit {float(lm['2']):.4f} (相対誤差 {math.sqrt(float(lm['2'])):.4f})、"
          f"3bit {float(lm['3']):.4f} ({math.sqrt(float(lm['3'])):.4f})。ガウスの R(D) 限界は 2bit で 0.0625 (0.25)。")


if __name__ == "__main__":
    main(sys.argv[1:])
