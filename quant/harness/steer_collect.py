"""言語方向の推定用に、参照ハーネス (stream_ref.py) で残差と routing を層・位置区分ごとに集める。

目的 (research/SPEC_STEER_JA.md §2):
  - 日本語と中国語 (および英語推論・コード) の系列を prefill し、各 MoE 層で
      * MoE 入力 x (ffn_norm 後, [n,5120]) の区分ごとの和・二乗和・件数  →  平均差が「言語方向」の候補
      * ブロック出力の残差 h (hc コピーを平均, [n,5120]) の同じ統計          →  残差に直接足す/射影する方向
      * routing の選択回数 counts[e] を区分ごとに                              →  zh_experts.py で言語偏重 expert を特定
  - 位置区分 (group) は系列ごとに渡す。既定は <think> の前後と DSML 以降で think/answer/tool に分ける。
    言語別の方向が欲しいときは、日本語系列と中国語系列を別ファイルにして collect し、後段で平均差を取る。

GPU 版 (本番): LIL イメージ内で
  python3 harness/steer_collect.py --in work/r5/seqs_ja.jsonl --out work/r5/collect_ja.pt
CPU 版 (このスクリプトの import テスト、モデルは作らない):
  tools/.venv/bin/python -I harness/steer_collect.py --self-test

入力 jsonl: 1 行 = {"ids": [...], "score_from": int, "src": str,
                    "groups": {"answer": [[lo,hi],...], "think": [...], ...}  # 省略時は自動分割}
出力 .pt: {"layers": [...], "dim": 5120, "groups": [...],
           "moe_in":  {group: {layer: {"sum": f32[5120], "sqsum": f32[5120], "n": int}}},
           "resid":   {group: {layer: {...}}},
           "routing": {group: {layer: int64[384]}},
           "ntok": {group: int}, "meta": {...}}
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THINK_END = 128822   # </think>
DSML = 128825        # ｜DSML｜ (ツール呼び出し)

# --------------------------------------------------------------------------- 位置区分


def auto_groups(ids, score_from):
    """<think> 前 = think、以降で DSML までが answer、DSML 以降が tool。採点区間 (score_from 以降) だけ。"""
    n = len(ids)
    te = next((i for i in range(n) if ids[i] == THINK_END), None)
    dm = next((i for i in range(n) if ids[i] == DSML), None)
    think_end = (te + 1) if te is not None else score_from
    ans_end = dm if (dm is not None and dm >= think_end) else n
    g = {"think": [], "answer": [], "tool": []}
    g["think"].append([score_from, max(score_from, min(think_end, n))])
    g["answer"].append([max(score_from, think_end), max(score_from, ans_end)])
    if ans_end < n:
        g["tool"].append([max(score_from, ans_end), n])
    return {k: [[a, b] for a, b in v if b > a] for k, v in g.items() if any(b > a for a, b in v)}


def group_vec(ids_len, groups, names):
    """各位置 -> group index (無しは -1)。重なりは最初の group を採る。"""
    gv = np.full(ids_len, -1, dtype=np.int64)
    for gi, name in enumerate(names):
        for lo, hi in groups.get(name, []):
            sel = gv[lo:hi] < 0
            gv[lo:hi][sel] = gi
    return gv


# --------------------------------------------------------------------------- 集計器


class Accum:
    def __init__(self, names, dim, n_experts):
        self.names, self.dim, self.ne = names, dim, n_experts
        self.moe_in = {g: {} for g in names}
        self.resid = {g: {} for g in names}
        self.routing = {g: {} for g in names}
        self.ntok = {g: 0 for g in names}
        self.cur = None  # 現在の系列の group vector (np.int64[seqlen])

    def _add(self, store, layer, gi, X):
        import torch

        d = store[self.names[gi]].setdefault(layer, {"sum": None, "sqsum": None, "n": 0})
        xf = X.float()
        s, sq = xf.sum(0).cpu(), (xf * xf).sum(0).cpu()
        d["sum"] = s if d["sum"] is None else d["sum"] + s
        d["sqsum"] = sq if d["sqsum"] is None else d["sqsum"] + sq
        d["n"] += int(X.shape[0])

    def feed(self, store, layer, X):
        """X: [n, dim] (この系列の全位置、行の並びは ids 順)。cur で区分に振り分ける。"""
        import torch

        if self.cur is None:
            return
        gv = torch.as_tensor(self.cur, device=X.device)
        for gi in range(len(self.names)):
            m = gv == gi
            if m.any():
                self._add(store, layer, gi, X[m])

    def feed_routing(self, layer, indices):
        """indices: [n, topk] (expert 番号)。cur で区分ごとに bincount。"""
        if self.cur is None:
            return
        import torch

        gv = self.cur
        flat_g = np.repeat(gv, indices.shape[1])
        flat_e = indices.reshape(-1).cpu().numpy()
        for gi, name in enumerate(self.names):
            m = flat_g == gi
            if m.any():
                c = np.bincount(flat_e[m], minlength=self.ne).astype(np.int64)
                r = self.routing[name].setdefault(layer, np.zeros(self.ne, dtype=np.int64))
                r += c

    def finalize_ntok(self):
        for g in self.names:
            anyl = next(iter(self.moe_in[g].values()), None)
            self.ntok[g] = anyl["n"] if anyl else 0


ACC: Accum | None = None


# --------------------------------------------------------------------------- フック


def install_hooks(M, acc):
    """stream_ref.build() の後に呼ぶ。MoE 入力・ブロック残差・routing を捕捉する。"""
    import torch

    base_block_forward = M.Block.forward
    base_moe_forward = M.MoE.forward  # stream_ref が差し替えた moe_forward

    def moe_forward(self, x, image_mask=None):
        shape = x.size()
        xf = x.view(-1, self.dim)
        if ACC is not None:
            ACC.feed(ACC.moe_in, self.layer_id, xf)
            weights, indices = self.gate(xf, None if image_mask is None else image_mask.flatten())
            ACC.feed_routing(self.layer_id, indices)
        return base_moe_forward(self, x, image_mask)

    def block_forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
        out, ffn_pre = base_block_forward(self, x, start_pos, pre_mix, image_mask, *attn_args)
        if ACC is not None:
            # out: [b,s,hc,dim] -> 残差を hc 平均で [n,dim]
            h = out.mean(dim=2).reshape(-1, out.shape[-1])
            ACC.feed(ACC.resid, self.layer_id, h)
        return out, ffn_pre

    M.MoE.forward = moe_forward
    M.Block.forward = block_forward


# --------------------------------------------------------------------------- 実行


def run(args):
    import torch

    sys.path.insert(0, HERE)
    import stream_ref as SR

    model, tok, margs = SR.build({"max_batch_size": 1, "max_seq_len": args.max_seq_len},
                                 args.expert_budget_gb, exl3_pack=args.exl3_pack)
    names = args.groups.split(",")
    global ACC
    ACC = Accum(names, margs.dim, margs.n_routed_experts)
    install_hooks(SR.M, ACC)
    SR._patch_head()

    seqs = [json.loads(l) for l in open(args.inp)]
    for i, sq in enumerate(seqs):
        ids, sf = sq["ids"], sq.get("score_from", 0)
        grp = sq.get("groups") or auto_groups(ids, sf)
        ACC.cur = group_vec(len(ids), grp, names)
        with torch.inference_mode():
            SR.FULL_LOGITS = False
            model.forward(torch.tensor([ids], dtype=torch.long), 0)
        print(f"[collect] {i}: {len(ids)} tok, groups "
              + " ".join(f"{n}={int((ACC.cur==gi).sum())}" for gi, n in enumerate(names)), flush=True)
    ACC.cur = None
    ACC.finalize_ntok()
    save(args.out, ACC, margs, names)
    print(f"[write] {args.out}  ntok {ACC.ntok}", flush=True)


def save(path, acc, margs, names):
    import torch

    def pack(store):
        return {g: {L: {"sum": d["sum"], "sqsum": d["sqsum"], "n": d["n"]} for L, d in store[g].items()} for g in names}

    torch.save({"layers": sorted({L for g in names for L in acc.moe_in[g]}), "dim": margs.dim,
                "groups": names, "n_experts": margs.n_routed_experts,
                "moe_in": pack(acc.moe_in), "resid": pack(acc.resid),
                "routing": {g: {L: v for L, v in acc.routing[g].items()} for g in names},
                "ntok": acc.ntok}, path)


def self_test():
    """torch/モデル無しで、位置区分と集計のロジックだけ確かめる (numpy)。"""
    ids = [0, 128803, 10, 11, THINK_END, 20, 21, 22, DSML, 30, 31]
    g = auto_groups(ids, score_from=2)
    assert g["think"] == [[2, 5]], g
    assert g["answer"] == [[5, 8]], g
    assert g["tool"] == [[8, 11]], g
    names = ["think", "answer", "tool"]
    gv = group_vec(len(ids), g, names)
    assert list(gv) == [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2], list(gv)
    # 件数が区分の長さに一致
    for gi, n in enumerate(names):
        assert int((gv == gi).sum()) == sum(b - a for a, b in g[n])
    # 自動分割で think のみ (DSML/think 無し) のケース
    g2 = auto_groups([0, 1, 2, 3], score_from=1)
    assert g2 == {"answer": [[1, 4]]}, g2
    print("[self-test] OK: auto_groups, group_vec, counts consistent")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", default="")
    ap.add_argument("--groups", default="think,answer,tool")
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--expert-budget-gb", type=float, default=60)
    ap.add_argument("--exl3-pack", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    assert a.inp and a.out, "--in と --out が要る (または --self-test)"
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
