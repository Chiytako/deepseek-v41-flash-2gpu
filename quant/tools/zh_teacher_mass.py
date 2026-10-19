"""teacher-forced の top-32 logprob から、中国語専用トークンに置かれる確率質量を測る (CPU、torch 不要)。

入力: harness/stream_ref.py --teacher-out / --teacher-ref が書いた .pt (recs: ids, score_from, top_v, top_i)。
  教師 (原重み 1017) の .pt と、学生 (v0 など) の .pt を並べて渡す。学生の .pt の top_v / top_i は学生自身の
  top-32 (stream_ref.py:449-466)。同じ系列なので位置ごとに対にして比べられる。

測るもの (位置ごと、領域別: think / answer / tool、answer は code と本文を分ける):
  m1    : T=1 の top-32 のうち zh_only (zs ∪ zt) トークンの確率の和。top-32 の外は分からない (residual として別に出す)
  m06   : T=0.6 で top-32 を再正規化したときの和 (GLM 実トラフィックの 65% が T=0.6)
  m06p  : T=0.6 + top_p 0.95 (top-32 内で nucleus を切る)
  argmax: 最尤トークンが zh_only の位置の割合 (greedy で混入が起きる位置)
  P(>=1): 回答 1 本あたり、サンプリングで zh_only を 1 回以上引く確率 = 1 - Π(1 - m)
  zw    : 漢字だけの中国語機能語トークン (可以 没有 …) の質量 (字は日本語にもあるので別に出す)
  frag  : CJK のバイト断片トークンの質量 (日本語の字の書き出しにも使うので、混入の指標にはしない)
  counterpart: 目標トークンが「複数トークンに分かれる常用漢字」(例: 拠 勧 奨 遅) の 1 個目のとき、その字の
     中国語の字体 (Unihan の kSimplifiedVariant / kTraditionalVariant / kJapaneseOldVariant / kZVariant / kSemanticVariant
     で引いた ZS / ZT 字) で始まるトークンに置かれた質量

usage: zh_teacher_mass.py --teacher T.pt [--student NAME=S.pt ...] [--out out.json] [--examples 15]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zh_detect as Z  # noqa: E402
import zh_vocab_r5 as V  # noqa: E402  (語彙走査。tools/zh_vocab_r5.py)
from zh_ptload import load_pt  # noqa: E402

UNIHAN = DSV41 + "/work/zh/data/unihan/Unihan_Variants.txt"
THINK_END = 128822
DSML = 128825


def load_variants():
    var = defaultdict(set)
    simp = defaultdict(set)
    try:
        for line in open(UNIHAN, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            cp, field, val = line.rstrip("\n").split("\t", 2)
            c = chr(int(cp[2:], 16))
            vs = [chr(int(v.split("<")[0][2:], 16)) for v in val.split() if v.startswith("U+")]
            if field == "kSimplifiedVariant":
                simp[c].update(vs)
            if field in ("kSimplifiedVariant", "kTraditionalVariant", "kJapaneseOldVariant", "kZVariant",
                         "kSemanticVariant", "kSpecializedSemanticVariant", "kJapaneseVariant"):
                var[c].update(vs)
    except OSError:
        pass
    return var, simp


def zh_counterparts(c, var, simp):
    cand = set(simp.get(c, ()))
    for v in var.get(c, ()):
        cand.add(v)
        cand.update(simp.get(v, ()))
    return {x for x in cand if x != c and Z.char_class(x) in ("ZS", "ZT")}


class Vocab:
    def __init__(self):
        tj = json.load(open(V.TOK, encoding="utf-8"))
        self.tb = V.token_bytes(tj)
        d = json.load(open(f"{HERE}/zh_only_tokens.json", encoding="utf-8"))
        n = len(self.tb)
        self.is_zh = np.zeros(n, bool)
        self.is_zh[d["zh_only"]] = True
        self.is_zw = np.zeros(n, bool)
        self.is_zw[d["sets"]["zw"]] = True
        self.is_frag = np.zeros(n, bool)
        self.is_frag[d["sets"]["frag"]] = True
        # 先頭の文字 -> トークン id (counterpart 用)
        self.first_char = defaultdict(list)
        for i, b in enumerate(self.tb):
            if b:
                t = b.decode("utf-8", errors="ignore")
                if t:
                    self.first_char[t[0]].append(i)
        self.text = lambda i: (self.tb[i] or b"").decode("utf-8", errors="replace")
        self.str2id = {}
        for i, b in enumerate(self.tb):
            if b:
                self.str2id.setdefault(b, i)
        self._twin = {}

    def twins(self, t, var, simp):
        """J 漢字を含むトークン t の中国語字体の双子トークン (各字を ZS/ZT の異体字に置き換えた文字列がトークンにあるもの)。"""
        if t in self._twin:
            return self._twin[t]
        out = set()
        b = self.tb[t]
        if b:
            s = b.decode("utf-8", errors="ignore")
            if s and any(Z.is_han(ord(c)) for c in s) and not self.is_zh[t]:
                import itertools
                opts = []
                for c in s:
                    cs = zh_counterparts(c, var, simp) if Z.is_han(ord(c)) else set()
                    opts.append([c] + sorted(cs)[:3])
                if sum(len(o) > 1 for o in opts) and len(opts) <= 6:
                    for combo in itertools.product(*opts):
                        w = "".join(combo)
                        if w != s:
                            i = self.str2id.get(w.encode("utf-8"))
                            if i is not None and self.is_zh[i]:
                                out.add(i)
        self._twin[t] = out
        return out

    def ttype(self, t):
        b = self.tb[t]
        if b is None:
            return "special"
        if self.is_zh[t]:
            return "zh"
        s = b.decode("utf-8", errors="replace")
        if self.is_frag[t]:
            return "frag"
        if any(Z.is_kana(ord(c)) for c in s):
            return "kana"
        if any(Z.is_han(ord(c)) for c in s):
            return "kanji"
        if all(ord(c) < 128 for c in s):
            return "ascii"
        return "other"


def positions(rec, voc, var, simp):
    """位置ごとの属性: region, in_code, ttype, counterpart 候補 id 集合 (無ければ None)。"""
    ids = rec["ids"]
    sf = rec["score_from"]
    # 領域: prefix の最後の </think> より前なら think
    think = THINK_END not in ids[:sf] or (max(i for i, t in enumerate(ids[:sf]) if t == THINK_END) < max([i for i, t in enumerate(ids[:sf]) if t == 128821] or [-1]))
    region = "think" if think else "answer"
    fence = 0
    buf = b""
    out = []
    # バイト位置と文字境界
    allb = b"".join((voc.tb[t] or b"") for t in ids[sf:])
    starts = {}
    k = 0
    txt = allb.decode("utf-8", errors="replace")
    # 文字の開始バイト位置 -> (文字, バイト長)
    o = 0
    while o < len(allb):
        b0 = allb[o]
        L = 1 if b0 < 0x80 else 2 if b0 >> 5 == 6 else 3 if b0 >> 4 == 14 else 4 if b0 >> 3 == 30 else 1
        try:
            ch = allb[o:o + L].decode("utf-8")
        except UnicodeDecodeError:
            ch, L = "�", 1
        starts[o] = (ch, L)
        o += L
    o = 0
    for j, t in enumerate(ids[sf:]):
        if t == THINK_END:
            out.append(("think_end", False, "special", None))
            region = "answer"
            o += len(voc.tb[t] or b"")
            continue
        if t == DSML:
            region = "tool"
        b = voc.tb[t] or b""
        tt = voc.ttype(t)
        cp = None
        if o in starts:
            ch, L = starts[o]
            if L > len(b) and Z.char_class(ch) == "J":
                cs = zh_counterparts(ch, var, simp)
                cp = (ch, cs)
        out.append((region, fence % 2 == 1, tt, cp))
        buf = (buf + b)[-8:]
        fence += b.count(b"```")
        o += len(b)
    return out


def analyze(recs, voc, var, simp, label, examples=12):
    acc = defaultdict(lambda: defaultdict(float))
    per_seq = []
    ex = []
    pairs = Counter()
    cp_acc = defaultdict(float)
    for r in recs:
        tv = r["top_v"].astype(np.float32)
        ti = r["top_i"].astype(np.int64)
        p = np.exp(tv)
        zh = voc.is_zh[ti]
        zw = voc.is_zw[ti]
        fr = voc.is_frag[ti]
        m1 = (p * zh).sum(1)
        mzw = (p * zw).sum(1)
        mfr = (p * fr).sum(1)
        resid = np.clip(1 - p.sum(1), 0, 1)
        # T=0.6
        q = np.exp((tv - tv[:, :1]) / 0.6)
        q /= q.sum(1, keepdims=True)
        m06 = (q * zh).sum(1)
        # T=0.6 + top_p 0.95
        cq = np.cumsum(q, 1)
        keep = (cq - q) < 0.95
        qp = q * keep
        qp /= qp.sum(1, keepdims=True)
        m06p = (qp * zh).sum(1)
        am = zh[:, 0]
        attrs = positions(r, voc, var, simp)
        assert len(attrs) == len(m1), (len(attrs), len(m1))
        seq = defaultdict(lambda: defaultdict(float))
        tgt = r["ids"][r["score_from"]:]
        for j, (reg, incode, tt, cp) in enumerate(attrs):
            if reg == "think_end":
                continue
            key = reg if reg != "answer" else ("answer_code" if incode else "answer_text")
            for kk in (key, f"{key}|tgt={tt}"):
                a = acc[kk]
                a["n"] += 1
                a["m1"] += m1[j]
                a["m06"] += m06[j]
                a["m06p"] += m06p[j]
                a["argmax"] += am[j]
                a["zw"] += mzw[j]
                a["frag"] += mfr[j]
                a["resid"] += resid[j]
                a["gt1e2"] += m1[j] > 1e-2
                a["gt1e1"] += m1[j] > 1e-1
                a["tgt_zh"] += voc.is_zh[tgt[j]]
                a["log1m_m1"] += np.log1p(-min(m1[j], 0.999999))
                a["log1m_m06p"] += np.log1p(-min(m06p[j], 0.999999))
            s = seq[key]
            s["n"] += 1
            s["m1"] += m1[j]
            s["m06p"] += m06p[j]
            s["argmax"] += am[j]
            s["log1m_m1"] += float(np.log1p(-min(m1[j], 0.999999)))
            s["log1m_m06p"] += float(np.log1p(-min(m06p[j], 0.999999)))
            # 寄与の大きい (目標, 中国語トークン) の対
            if m1[j] > 1e-3:
                for k in np.nonzero(zh[j])[0]:
                    pairs[(voc.text(tgt[j]), voc.text(int(ti[j, k])))] += float(p[j, k])
            if m1[j] > 0.05 and len(ex) < 400:
                ctx = b"".join((voc.tb[t] or b"") for t in r["ids"][max(0, r["score_from"] + j - 12): r["score_from"] + j]).decode("utf-8", "replace")
                top = [(voc.text(int(ti[j, k])), round(float(p[j, k]), 3)) for k in np.nonzero(zh[j])[0][:3]]
                ex.append({"seq": r["i"], "pos": j, "region": key, "ctx": ctx[-24:], "target": voc.text(tgt[j]), "m1": round(float(m1[j]), 3), "zh_top": top})
            tw = voc.twins(int(tgt[j]), var, simp)
            if tw:
                tk = "twin|" + key
                a = acc[tk]
                a["n"] += 1
                mt = float(sum(p[j, k] for k in range(ti.shape[1]) if int(ti[j, k]) in tw))
                a["m1"] += mt
                a["argmax"] += int(ti[j, 0]) in tw
                a["log1m_m1"] += np.log1p(-min(mt, 0.999999))
            if cp is not None:
                ch, cs = cp
                cp_acc["n"] += 1
                if cs:
                    cp_acc["n_with_cp"] += 1
                    ids_cp = set()
                    for c in cs:
                        ids_cp.update(voc.first_char.get(c, ()))
                    mm = float(sum(p[j, k] for k in range(ti.shape[1]) if int(ti[j, k]) in ids_cp))
                    cp_acc["mass_cp"] += mm
                    cp_acc["argmax_cp"] += int(ti[j, 0]) in ids_cp
                    cp_acc["m1_at_multi"] += m1[j]
        per_seq.append({"i": r["i"], **{k: dict(v) for k, v in seq.items()}})
    return acc, per_seq, ex, pairs, cp_acc


def summarize(acc):
    out = {}
    for k, a in sorted(acc.items()):
        n = a["n"]
        if not n:
            continue
        out[k] = {"n": int(n), "m1_per_1k": 1e3 * a["m1"] / n, "m06_per_1k": 1e3 * a["m06"] / n, "m06p_per_1k": 1e3 * a["m06p"] / n,
                  "argmax_per_1k": 1e3 * a["argmax"] / n, "zw_per_1k": 1e3 * a["zw"] / n, "frag_per_1k": 1e3 * a["frag"] / n,
                  "resid_mean": a["resid"] / n, "pos_gt1e-2": int(a["gt1e2"]), "pos_gt1e-1": int(a["gt1e1"]), "tgt_zh": int(a["tgt_zh"]),
                  "P_ge1_T1": 1 - float(np.exp(a["log1m_m1"])), "P_ge1_T06p": 1 - float(np.exp(a["log1m_m06p"]))}
    return out


def boot_ratio(ps_t, ps_s, key, field, nb=2000, seed=0):
    """系列単位のブートストラップで Σstudent/Σteacher の 95% 区間。"""
    rnd = random.Random(seed)
    a = [(t.get(key, {}).get(field, 0.0), s.get(key, {}).get(field, 0.0)) for t, s in zip(ps_t, ps_s)]
    if not a or sum(x for x, _ in a) == 0:
        return None
    rs = []
    for _ in range(nb):
        smp = [a[rnd.randrange(len(a))] for _ in a]
        st = sum(x for x, _ in smp)
        if st > 0:
            rs.append(sum(y for _, y in smp) / st)
    rs.sort()
    return [sum(y for _, y in a) / sum(x for x, _ in a), rs[int(0.025 * len(rs))], rs[int(0.975 * len(rs)) - 1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", action="append", default=[], help="NAME=path.pt")
    ap.add_argument("--out", default="")
    ap.add_argument("--examples", type=int, default=12)
    a = ap.parse_args()
    voc = Vocab()
    var, simp = load_variants()
    runs = [("teacher", a.teacher)] + [tuple(s.split("=", 1)) for s in a.student]
    res = {}
    seqs = {}
    for name, path in runs:
        d = load_pt(path)
        acc, ps, ex, pairs, cp = analyze(d["recs"], voc, var, simp, name)
        sm = summarize(acc)
        res[name] = {"path": path, "summary": sm, "counterpart": dict(cp), "examples": ex[:200],
                     "top_pairs": [[t, z, round(m, 4)] for (t, z), m in pairs.most_common(40)]}
        seqs[name] = ps
        print(f"\n===== {name}: {path} ({len(d['recs'])} 系列)")
        print("  region                n     m1/1k  m06/1k  m06p/1k argmax/1k  zw/1k  frag/1k resid  >1e-2 >1e-1 tgt_zh  P>=1(T1) P>=1(T.6p)")
        for k, s in sm.items():
            if "|" in k and not k.endswith(("tgt=kanji", "tgt=frag", "tgt=kana", "tgt=ascii")) and not k.startswith("twin|"):
                continue
            print(f"  {k:22s}{s['n']:6d} {s['m1_per_1k']:8.3f} {s['m06_per_1k']:7.3f} {s['m06p_per_1k']:8.3f} {s['argmax_per_1k']:8.3f} {s['zw_per_1k']:7.3f} {s['frag_per_1k']:7.2f} {s['resid_mean']:.3f} {s['pos_gt1e-2']:5d} {s['pos_gt1e-1']:5d} {s['tgt_zh']:5d}  {s['P_ge1_T1']:.3f}  {s['P_ge1_T06p']:.3f}")
        if cp.get("n"):
            print(f"  counterpart: 複数トークンの常用漢字の書き出し {int(cp['n'])} 位置 (中国語字体あり {int(cp.get('n_with_cp', 0))})、"
                  f"中国語字体への質量 平均 {cp.get('mass_cp', 0) / max(1, cp.get('n_with_cp', 1)):.4f}、argmax が中国語字体 {int(cp.get('argmax_cp', 0))}、"
                  f"その位置の zh_only 質量 平均 {cp.get('m1_at_multi', 0) / max(1, cp.get('n_with_cp', 1)):.4f}")
        for key in ("answer_text", "think"):
            v1 = [1 - np.exp(q[key]["log1m_m1"]) for q in ps if key in q]
            v6 = [1 - np.exp(q[key]["log1m_m06p"]) for q in ps if key in q]
            if v1:
                res[name].setdefault("per_seq_P_ge1", {})[key] = {"T1": float(np.mean(v1)), "T06p": float(np.mean(v6)), "n_seq": len(v1)}
                print(f"  1 系列 (回答 1 本) あたり zh_only を 1 回以上引く確率の平均 [{key}]: T=1 {np.mean(v1):.4f}、T=0.6+top_p0.95 {np.mean(v6):.4f} (系列 {len(v1)})")
        print("  寄与の大きい (目標トークン -> 中国語トークン) 対:", " ".join(f"{t!r}->{z!r}:{m:.3f}" for t, z, m in res[name]["top_pairs"][:15]))
        for e in ex[: a.examples]:
            print("   例", json.dumps(e, ensure_ascii=False))
    if len(runs) > 1:
        print("\n===== 学生 / 教師 (系列ブートストラップ 95% 区間)")
        for name, _ in runs[1:]:
            for key in ("answer_text", "answer_code", "think", "tool"):
                for field in ("m1", "m06p", "argmax"):
                    br = boot_ratio(seqs["teacher"], seqs[name], key, field)
                    if br:
                        print(f"  {name} {key:12s} {field:6s} 比 {br[0]:.2f} [{br[1]:.2f}, {br[2]:.2f}]")
                        res[name].setdefault("ratio_vs_teacher", {})[f"{key}.{field}"] = br
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
        print(f"\n[write] {a.out}")


if __name__ == "__main__":
    main()
