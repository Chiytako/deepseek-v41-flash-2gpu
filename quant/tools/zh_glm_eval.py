"""GLM 実トラフィックの GLM 出力に zh_detect を掛ける (参考値、CPU)。

入力: work/x2/req/*.meta.json と *.u32 (V4.1 でトークン化した prompt + GLM 出力)。
出力は </think> で思考と回答に分け、回答は DSML のツール呼び出しの前までを本文とする。
回答が日本語 (仮名 10 字以上) のものについて level と内訳を数える。混入が見つかった要求だけ prompt を復号し、
prompt にある字・語・句 (copied) とモデルが新たに出したもの (novel) を分ける。level は novel だけで決まる。
思考 (多くは英語) も同じ判定器に掛け、漢字の簡体字 (ZS) と中国語の句を数える。

usage: zh_glm_eval.py [req_dir] [--out work/zh/glm_eval.json] [--workers 24]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zh_detect as Z  # noqa: E402

TOK = MODELS + "/DeepSeek-V4.1-Flash-1017/tokenizer.json"
THINK_END = 128822
_tok = None


def _init():
    global _tok
    from tokenizers import Tokenizer
    _tok = Tokenizer.from_file(TOK)


def proc(mf):
    metas = json.load(open(mf))
    if not metas:
        return Counter(), [], []
    u = np.fromfile(mf[:-10] + ".u32", dtype=np.uint32)
    off = 0
    S = Counter()
    ex = []
    rows = []
    for m in metas:
        p, o = m["plen"], m["olen"]
        out = u[off + p: off + p + o]
        pr = (off, p)
        off += p + o
        idx = np.where(out == THINK_END)[0]
        think_ids = out[: idx[0]] if len(idx) else out[:0]
        ans_ids = out[idx[0] + 1:] if len(idx) else out
        ans = _tok.decode(ans_ids.tolist(), skip_special_tokens=False)
        j = ans.find("<｜DSML｜")
        body = ans if j < 0 else ans[:j]
        th = _tok.decode(think_ids.tolist(), skip_special_tokens=False) if len(think_ids) else ""
        S["req"] += 1
        res_a = Z.detect(body)
        res_t = Z.detect(th) if th else None
        hit_a = res_a["is_ja"] and res_a["level"] >= 1
        hit_t = res_t is not None and res_t["level"] >= 1
        ctx = None
        if hit_a or hit_t:
            ctx = _tok.decode(u[pr[0]: pr[0] + pr[1]].tolist(), skip_special_tokens=False)
            if hit_a:
                res_a = Z.detect(body, context=ctx)
            if hit_t:
                res_t = Z.detect(th, context=ctx)
        if res_a["is_ja"]:
            S["ja_resp"] += 1
            S["ja_chars"] += res_a["chars"]
            S[f"ja_level{res_a['level']}"] += 1
            if res_a["zs"]:
                S["ja_zs_resp"] += 1
                S["ja_zs_chars"] += len(res_a["zs"])
                if len(res_a["zs"]) >= 3:
                    S["ja_zs3_resp"] += 1
            if res_a["zs_copied"]:
                S["ja_zs_copied_resp"] += 1
            if res_a["strong_words"]:
                S["ja_strong_resp"] += 1
            if res_a["clauses"]:
                S["ja_clause_resp"] += 1
            if res_a["zt"]:
                S["ja_zt_resp"] += 1
            if res_a["pinyin"]:
                S["ja_pinyin_resp"] += 1
            if res_a["zh_punct"]:
                S["ja_punct_resp"] += 1
            # 旧方式 (r3ja_zhmix: EUC-JP に無く GB2312 にある字、コードブロックのみ除外、copied の区別なし)
            old = [c for c in Z._RE_FENCE.sub("", body) if 0x4E00 <= ord(c) <= 0x9FFF and not Z._enc(c, "euc_jp") and Z._enc(c, "gb2312")]
            if old:
                S["ja_old_ge1"] += 1
            if len(old) >= 3:
                S["ja_old_ge3"] += 1
            if res_a["level"] >= 2:
                ex.append({"file": m["file"], "rid": m.get("rid"), "level": res_a["level"], "zs": res_a["zs"][:20],
                           "strong": res_a["strong_words"][:6], "clauses": [c["text"][:60] for c in res_a["clauses"][:2]],
                           "zs_copied": res_a["zs_copied"], "chars": res_a["chars"]})
            rows.append({"rid": m.get("rid"), "ts": m.get("ts"), "level": res_a["level"], "chars": res_a["chars"], "think_tok": int(len(think_ids)), "ans_tok": int(len(ans_ids))})
        elif res_a["chars"] >= 20:
            S["nonja_resp"] += 1
            if res_a["level"] >= 3 and res_a["han"] > res_a["chars"] * 0.3:
                S["zh_resp"] += 1
        if res_t is not None:
            S["think"] += 1
            S["think_chars"] += res_t["chars"]
            if res_t["is_ja"]:
                S["think_ja"] += 1
            S[f"think_level{res_t['level']}"] += 1
            if res_t["zs"]:
                S["think_zs"] += 1
            if res_t["clauses"]:
                S["think_clause"] += 1
            if res_t["level"] >= 2 and len(ex) < 2000:
                ex.append({"file": m["file"], "rid": m.get("rid"), "where": "think", "level": res_t["level"], "zs": res_t["zs"][:20],
                           "strong": res_t["strong_words"][:6], "clauses": [c["text"][:60] for c in res_t["clauses"][:2]], "zs_copied": res_t["zs_copied"]})
    return S, ex, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("req_dir", nargs="?", default=DSV41 + "/work/x2/req")
    ap.add_argument("--out", default=DSV41 + "/work/zh/glm_eval.json")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    files = sorted(glob.glob(a.req_dir + "/*.meta.json"))
    S = Counter()
    EX = []
    ROWS = []
    with mp.Pool(a.workers, initializer=_init) as pool:
        for s, ex, rows in pool.imap_unordered(proc, files):
            S.update(s)
            EX.extend(ex)
            ROWS.extend(rows)
    n = max(1, S["ja_resp"])
    print(f"[req] {S['req']} 件、日本語回答 {S['ja_resp']} 件 ({S['ja_chars']/1e6:.2f}M 字)、非日本語回答 {S['nonja_resp']} (うち中国語の回答 {S['zh_resp']})")
    print(f"[ja] level>=1 {(n - S['ja_level0'])/n:.3%}  level>=2 {(S['ja_level2'] + S['ja_level3'])/n:.3%}  level=3 {S['ja_level3']/n:.3%}")
    print(f"     ZS 字 (novel) あり {S['ja_zs_resp']/n:.3%} (3 字以上 {S['ja_zs3_resp']/n:.3%})、ZS が prompt から copied {S['ja_zs_copied_resp']/n:.3%}、"
          f"strong 機能語 {S['ja_strong_resp']/n:.3%}、中国語の句 {S['ja_clause_resp']/n:.3%}、ZT {S['ja_zt_resp']/n:.3%}、拼音 {S['ja_pinyin_resp']/n:.3%}、句読点 {S['ja_punct_resp']/n:.3%}")
    print(f"     旧方式 (r3ja_zhmix 相当) 1 字以上 {S['ja_old_ge1']/n:.3%}、3 字以上 {S['ja_old_ge3']/n:.3%}")
    print(f"     1 万字あたりの novel ZS 字 {S['ja_zs_chars']/max(1, S['ja_chars'])*1e4:.3f}")
    t = max(1, S["think"])
    print(f"[think] {S['think']} 件 ({S['think_chars']/1e6:.1f}M 字)、日本語の思考 {S['think_ja']/t:.2%}、level>=2 {(S['think_level2'] + S['think_level3'])/t:.3%}、ZS {S['think_zs']/t:.3%}、句 {S['think_clause']/t:.3%}")
    zc = Counter()
    for e in EX:
        if e.get("where") != "think":
            zc.update(e["zs"])
    print("[ja novel ZS 上位]", " ".join(f"{c}{k}" for c, k in zc.most_common(40)))
    for e in [e for e in EX if e.get("where") != "think"][:30]:
        print("  例", json.dumps(e, ensure_ascii=False))
    for e in [e for e in EX if e.get("where") == "think"][:10]:
        print("  思考の例", json.dumps(e, ensure_ascii=False))
    json.dump({"stats": dict(S), "examples": EX[:3000], "rows": ROWS}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"[write] {a.out}  (examples / rows は要求ログ由来。外に出さない)")


if __name__ == "__main__":
    main()
