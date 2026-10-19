"""stream_ref.py の画像入り系列版 (教師 logits / 学生採点 / 教師の greedy 生成)。

stream_ref.py は変えない。build() (専門家ストリーミング、Engram の pread、torch 版 sparse_attn) を
そのまま使い、入力の作り方と forward の呼び方 (images, token_types) だけを足す。

入力 jsonl (1 行 1 問):
  {"id": "...", "messages": [OpenAI 形式。image_url の url はローカルパス (jsonl からの相対可) /
                              file:// / data:...;base64 / http(s)],
   "answer": "回答本文" | {"content": ..., "reasoning_content": ...}   # 任意
   "thinking_mode": "chat"|"thinking", "reasoning_effort": ..., "tools": [...], "max_new": 512}  # 任意
  - answer があれば: ids = プロンプト + 回答 (assistant の書式、EOS 付き)、score_from = 回答の先頭。
  - answer が無ければ: 教師モードでは教師が画像込みで greedy 生成し、その出力を回答として採点する。
  - 既存形式 {"ids": [...], "score_from": n} の行もそのまま通る (画像なし、stream_ref.teacher と同じ扱い)。
プロンプトは ref/generate.py の prepare_case と同じ (encoding.encode_case → image_processor.prepare_vl_inputs)。

出力 .pt (stream_ref.teacher と同じキー + 画像の情報):
  recs[i]: i, src, ids, score_from, top_v, top_i, tgt_lp, argmax_match  (ここまで既存と同じ)
           + id, prompt_len, answer_source ("given"|"teacher_greedy"|"ids"|"ref"), answer_text, truncated,
             images [{start, n_tokens, n_llm_h, n_llm_w, n_vit_h, n_vit_w, sha256}], n_image_tokens,
             counts_image (np.int32 [n_layers, n_experts]、この系列の prefill の画像区間)
  counts       : 全トークン (テキスト + 画像) の選択回数。既存と同じ意味 (qad_l2 --counts-from で読める)
  counts_text  : テキスト区間 (教師の生成に使った prefill / decode は含めない。採点の prefill だけ)
  counts_image : 画像区間 (START/IMAGE/NEW_LINE/END。gate が bias_vl を使う位置)
  kl, agree    : 学生モードのとき位置ごとの KL / top-1 一致 (既存と同じ)

使い方 (dsv41-quant:151 の中。マウントは QAD_PLAN.md §1.0 の qrun と同じ + 画像のディレクトリ):
  # 教師 (answer の無い問題は教師の greedy 生成を回答にする。生成結果は --gen-out にも書く)
  python3 stream_ref_vl.py --vl-in /v/vl32.jsonl --teacher-out /t/vl32_teacher.pt --gen-out /t/vl32_gen.jsonl \
      --max-new 512 --max-seq-len 8192 --expert-budget-gb 20
  # 学生 (ids は教師 .pt から取る。プロンプトと画像の一致を検査する)
  python3 stream_ref_vl.py --vl-in /v/vl32.jsonl --teacher-ref /t/vl32_teacher.pt --teacher-out /t/vl32_v1.pt \
      --exl3-pack $V1 --max-seq-len 8192 --expert-budget-gb 20
  # 生成だけ (logits を取らない)
  python3 stream_ref_vl.py --vl-in /v/vl32.jsonl --gen-out /t/vl32_gen_v1.jsonl --exl3-pack $V1
  # GPU なしの下見 (トークン数、画像区間の位置、上限の検査だけ)
  python3 stream_ref_vl.py --vl-in /v/vl32.jsonl --plan-only
  # ds41rt で同じ問題を生成させる要求本体 (画像は data: URL に埋める。ds41rt はローカルパスを受けない)
  python3 stream_ref_vl.py --vl-in /v/vl32.jsonl --export-openai /t/vl32_ds41rt.jsonl --max-new 512
"""
import argparse
import base64
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_ref as SR  # noqa: E402  (REF と CKPT/encoding を sys.path に入れる)

M = SR.M
from encoding import IMAGE_PLACEHOLDER, encode_case  # noqa: E402
from image_processor import (IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START, TEXT,  # noqa: E402
                             load_image_bytes, prepare_vl_inputs)

MAX_IMAGES = 16  # ds41rt の上限に合わせた警告用


# --------------------------------------------------------------------------- 入力の準備 (CPU)
def load_model_args(over=None):
    cfg = json.load(open(os.path.join(SR.CKPT, "inference", "config.json")))
    cfg.update(over or {})
    return M.ModelArgs(**cfg)


def _resolve_url(url, base_dir):
    if not isinstance(url, str) or url.startswith(("data:", "http://", "https://")):
        return url
    if url.startswith("file://"):
        url = url[len("file://"):]
    if not os.path.isabs(url):
        url = os.path.join(base_dir, url)
    if not os.path.exists(url):
        raise FileNotFoundError(url)
    return url


def resolve_image_paths(messages, base_dir):
    """image_url / image の url をローカルの絶対パスへ (data: と http(s) はそのまま)。元は変えない。"""
    messages = copy.deepcopy(messages)

    def walk(blocks):
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "image_url":
                iu = b.get("image_url")
                if isinstance(iu, str):
                    b["image_url"] = _resolve_url(iu, base_dir)
                elif isinstance(iu, dict):
                    iu["url"] = _resolve_url(iu.get("url", ""), base_dir)
            elif b.get("type") == "image" and isinstance(b.get("url"), str):
                b["url"] = _resolve_url(b["url"], base_dir)
            if isinstance(b.get("content"), list):
                walk(b["content"])

    for m in messages:
        if isinstance(m.get("content"), list):
            walk(m["content"])
    return messages


def image_sha256(record):
    return hashlib.sha256(load_image_bytes(record)).hexdigest()


def _case(rec, messages):
    case = {"messages": copy.deepcopy(messages), "thinking_mode": rec.get("thinking_mode"),
            "reasoning_effort": rec.get("reasoning_effort")}
    if rec.get("tools"):
        case["messages"][0]["tools"] = rec["tools"]
    if rec.get("context"):
        raise ValueError("context (KV の事前充填) は未対応 (ref/generate.py の prepare_case と同じ)")
    return case


def _answer_message(ans):
    if isinstance(ans, str):
        return {"role": "assistant", "content": ans}
    if isinstance(ans, dict):
        m = {"role": "assistant"}
        m.update(ans)
        return m
    raise TypeError(f"answer は str か dict: {type(ans)}")


def image_spans(c):
    return [{"start": img.start, "n_tokens": int(img.types.numel()), "n_vit_h": img.n_vit_h, "n_vit_w": img.n_vit_w,
             "n_llm_h": int((img.types == IMAGE_NEW_LINE).sum()),
             "n_llm_w": int((img.types == IMAGE).sum()) // max(1, int((img.types == IMAGE_NEW_LINE).sum()))}
            for img in (c["images"] or [])]


def check_spans(c, image_token_id, downsample=3, limit=None):
    """画像区間の整合: 位置・型の並び・id・プレフィル内・重なりなし・aligner の行数。違反は AssertionError。
    limit: 画像区間が収まるべき終わり (既定 prompt_len。採点だけの行 (回答が決まっている) は系列全体が 1 回の
    prefill なので len(ids) でよい)。"""
    ids, types = c["ids"], c["token_types"]
    sf = c["prompt_len"] if limit is None else limit
    assert len(ids) == len(types)
    covered = np.zeros(len(ids), dtype=bool)
    prev_end = 0
    for img, sp in zip(c["images"] or [], image_spans(c)):
        s, n = sp["start"], sp["n_tokens"]
        assert s >= prev_end, "画像区間が重なっている / 順序が逆"
        assert s + n <= sf, "画像区間がプロンプト (prefill の 1 チャンク) からはみ出す"
        t = types[s:s + n]
        h, w = sp["n_llm_h"], sp["n_llm_w"]
        assert t == [IMAGE_START] + ([IMAGE] * w + [IMAGE_NEW_LINE]) * h + [IMAGE_END], "型の並び"
        assert all(x == image_token_id for x in ids[s:s + n]), "画像区間の id"
        assert img.patches.shape[0] == sp["n_vit_h"] * sp["n_vit_w"], "パッチ数"
        # Aligner は patch 格子を downsample の倍数へ pad して unfold する → ceil(n_vit/r) 行 × 列
        assert (-(-sp["n_vit_h"] // downsample), -(-sp["n_vit_w"] // downsample)) == (h, w), "aligner の格子"
        covered[s:s + n] = True
        prev_end = s + n
    for i in np.flatnonzero(~covered):
        assert types[i] == TEXT and ids[i] != image_token_id, f"画像区間の外に画像トークン (pos {i})"
    return True


def prepare_vl_case(rec, tok, args, thinking="chat", base_dir="."):
    """1 問を ids / token_types / images / score_from に。answer が無ければ score_from=None (生成待ち)。"""
    if "ids" in rec and "messages" not in rec:  # 既存の {ids, score_from}
        ids = list(rec["ids"])
        return {"id": rec.get("id"), "src": rec.get("src"), "ids": ids, "token_types": [TEXT] * len(ids),
                "images": None, "image_sha": [], "prompt_len": int(rec["score_from"]),
                "score_from": int(rec["score_from"]), "answer_source": "ids", "max_new": rec.get("max_new")}
    if "vl_ids" in rec:  # tools/vl/build_sets.py の行 (vl_kl / vl_calib)。回答込みの会話と採点位置が決まっている
        return _prepare_rows_case(rec, tok, args, thinking, base_dir)
    messages = resolve_image_paths(rec["messages"], base_dir)
    if rec.get("answer") is None and messages and messages[-1].get("role") == "assistant":
        # 最後が assistant の会話は、その発話を回答として採点する (プロンプトに入れて続きを生成させない)
        last = messages[-1]
        rec = dict(rec, answer={k: v for k, v in last.items() if k != "role"})
        messages = messages[:-1]
    case = _case(rec, messages)
    prompt, image_records = encode_case(case, thinking)
    tokens, types, images = prepare_vl_inputs(prompt, image_records, tok, args)
    out = {"id": rec.get("id"), "src": rec.get("src"), "prompt_len": len(tokens), "images": images,
           "image_sha": [image_sha256(r) for r in image_records], "max_new": rec.get("max_new")}
    out.update({k: rec[k] for k in ("lang", "cat", "sub") if k in rec})
    if rec.get("answer") is not None:
        full_case = _case(rec, messages + [_answer_message(rec["answer"])])
        full_prompt, full_records = encode_case(full_case, thinking)
        if not full_prompt.startswith(prompt) or len(full_records) != len(image_records):
            raise ValueError(f"{rec.get('id')}: 回答を足した書式がプロンプトの延長になっていない")
        suffix = full_prompt[len(prompt):]
        if IMAGE_PLACEHOLDER in suffix:
            raise ValueError("回答に画像は入れられない")
        ans_ids = tok.encode(suffix, add_special_tokens=False)
        out.update(ids=tokens + ans_ids, token_types=types + [TEXT] * len(ans_ids), score_from=len(tokens),
                   answer_source="given", answer_text=suffix)
    else:
        out.update(ids=list(tokens), token_types=list(types), score_from=None, answer_source="teacher_greedy")
    check_spans(out, args.image_token_id, args.vision_downsample_ratio)
    return out


def _prepare_rows_case(rec, tok, args, thinking, base_dir):
    """build_sets.py の行: messages (最後の assistant まで) を公式経路で描画し直し、行の vl_ids / token_types /
    image_spans と照合する。採点位置は行の score_from (build_calib の規則: <｜Assistant｜> の次から。chat では
    </think> の予測も含む)。spans / q_spans (絶対位置) は rec に残し、区間別の集計に使う。"""
    messages = resolve_image_paths(rec["messages"], base_dir)
    case = _case(rec, messages)
    prompt, image_records = encode_case(case, thinking)
    tokens, types, images = prepare_vl_inputs(prompt, image_records, tok, args)
    rid = rec.get("id")
    if tokens != [int(t) for t in rec["vl_ids"]]:
        raise ValueError(f"{rid}: vl_ids が公式経路 (encode_case → prepare_vl_inputs) の描画と違う")
    if rec.get("token_types") is not None and types != [int(t) for t in rec["token_types"]]:
        raise ValueError(f"{rid}: token_types が公式経路の描画と違う")
    got = [(im.start, int(im.types.numel()), im.n_vit_h, im.n_vit_w) for im in (images or [])]
    want = [(int(s["start"]), int(s["len"]), int(s["n_vit_h"]), int(s["n_vit_w"])) for s in rec.get("image_spans") or []]
    if want and got != want:
        raise ValueError(f"{rid}: image_spans が公式経路の描画と違う ({got} vs {want})")
    sf = int(rec["score_from"])
    # score_from は最初の assistant の発話の位置: その前までの会話を描画したもの + 1 (<｜Assistant｜>) と一致すること
    k = next((j for j, m in enumerate(messages) if m.get("role") == "assistant"), None)
    if k is None:
        raise ValueError(f"{rid}: assistant の発話が無い (採点する回答が無い)")
    p_prompt, p_recs = encode_case(_case(rec, messages[:k]), thinking)
    p_tokens, _, _ = prepare_vl_inputs(p_prompt, p_recs, tok, args)
    a_id = tok.convert_tokens_to_ids("<｜Assistant｜>")
    if tokens[:len(p_tokens)] != p_tokens:
        raise ValueError(f"{rid}: 最初の回答の前までの描画が全体の描画の接頭辞になっていない")
    j = len(p_tokens) - 1
    while j >= 0 and tokens[j] != a_id:
        j -= 1
    if j < 0 or sf != j + 1:
        raise ValueError(f"{rid}: score_from {sf} が最初の <｜Assistant｜> の次 ({j + 1}) と違う")
    if not 0 < sf < len(tokens):
        raise ValueError(f"{rid}: score_from {sf} が範囲外")
    out = {"id": rid, "src": rec.get("src") or rec.get("ref"), "ids": tokens, "token_types": types, "images": images,
           "image_sha": [image_sha256(r) for r in image_records], "prompt_len": int(rec.get("prompt_len", sf)),
           "score_from": sf, "answer_source": "given", "answer_text": tok.decode(tokens[sf:]),
           "max_new": rec.get("max_new"), "spans": [list(map(int, s)) for s in rec.get("spans") or []],
           "q_spans": [list(map(int, s)) for s in rec.get("q_spans") or []]}
    out.update({k_: rec[k_] for k_ in ("lang", "cat", "sub") if k_ in rec})
    # 回答込みの系列全体を 1 回の prefill で採点するので、画像区間は系列内にあればよい
    check_spans(out, args.image_token_id, args.vision_downsample_ratio, limit=len(tokens))
    return out


def apply_score_q(c):
    """--score-q: q_spans (画像の直後の問い) があれば、採点をその先頭から始める (answer_from に元の位置を残す)。"""
    if c.get("q_spans") and c["score_from"] is not None:
        c["answer_from"] = c["score_from"]
        c["score_from"] = min(c["score_from"], min(s[0] for s in c["q_spans"]))
    return c


def load_vl_cases(path, tok, args, thinking="chat", image_root=""):
    base = image_root or os.path.dirname(os.path.abspath(path))
    return [prepare_vl_case(json.loads(l), tok, args, thinking, base) for l in open(path) if l.strip()]


# --------------------------------------------------------------------------- routing counts (text / image)
class RouteCounter:
    """MoE.forward / Gate.forward を包み、画像区間 (image_mask=True) の選択回数を層ごとに数える。
    STORE.counts (全トークン) はそのまま。テキスト区間 = 全体 − 画像。出力は変えない。"""

    def __init__(self):
        self.image = {}       # layer_id -> np.int64[n_experts] (累積)
        self.seq = {}         # 系列ごと (reset_seq で空にする)
        self.n_image_tokens = 0
        self._slot = None

    def install(self, Mmod=M):
        rc = self
        orig_moe, orig_gate = Mmod.MoE.forward, Mmod.Gate.forward

        def gate_forward(gself, x, image_mask=None):
            w, i = orig_gate(gself, x, image_mask)
            s = rc._slot
            if s is not None and "seen" not in s:
                s["seen"] = True
                if image_mask is not None and bool(image_mask.any()):
                    sel = i[image_mask.to(i.device)]
                    s["c"] = torch.bincount(sel.flatten(), minlength=gself.weight.shape[0]).cpu().numpy()
                    s["n"] = int(sel.shape[0])
            return w, i

        def moe_forward(mself, x, image_mask=None):
            rc._slot = {}
            try:
                y = orig_moe(mself, x, image_mask)
                s = rc._slot
            finally:
                rc._slot = None
            if "c" in s:
                for d in (rc.image, rc.seq):
                    acc = d.setdefault(mself.layer_id, np.zeros(mself.n_routed_experts, dtype=np.int64))
                    acc += s["c"].astype(np.int64)
                if mself.layer_id == 0:
                    rc.n_image_tokens += s["n"]
            return y

        gate_forward._vl_orig = orig_gate
        moe_forward._vl_orig = orig_moe
        Mmod.Gate.forward = gate_forward
        Mmod.MoE.forward = moe_forward
        self.M = Mmod

    def uninstall(self):
        self.M.Gate.forward = self.M.Gate.forward._vl_orig
        self.M.MoE.forward = self.M.MoE.forward._vl_orig

    def reset_seq(self):
        self.seq = {}

    def seq_array(self, n_layers, n_experts):
        a = np.zeros((n_layers, n_experts), dtype=np.int32)
        for l, v in self.seq.items():
            if l < n_layers:
                a[l] = v
        return a

    def split(self, total):
        """total (STORE.counts) から (text, image) を作る。"""
        img = {k: v.copy() for k, v in self.image.items()}
        txt = {k: v - img.get(k, 0) for k, v in total.items()}
        assert all((v >= 0).all() for v in txt.values()), "画像の回数が全体を超えた (数え方の不整合)"
        return txt, img


COUNTER: RouteCounter | None = None


# --------------------------------------------------------------------------- forward
def _vl_kwargs(types, images):
    if not images:
        return {}  # テキストだけなら stream_ref.teacher と同じ呼び方 (image_mask=None)
    return {"images": [images], "token_types": torch.tensor([types], dtype=torch.long)}


@torch.inference_mode()
def prefill_logits(model, ids, types, images):
    SR.FULL_LOGITS = True
    try:
        _, logits, _ = model.forward(torch.tensor([ids], dtype=torch.long), 0, **_vl_kwargs(types, images))
    finally:
        SR.FULL_LOGITS = False
    return logits


@torch.inference_mode()
def greedy_vl(model, ids, types, images, max_new, eos_id):
    """画像込みの prefill 1 回 + decode を 1 トークンずつ (stream_ref.greedy と同じ流れ)。EOS は残す。"""
    t0 = time.time()
    nxt, _, _ = model.forward(torch.tensor([ids], dtype=torch.long), 0, **_vl_kwargs(types, images))
    t_pre = time.time() - t0
    out = [int(nxt.flatten()[0])]
    pos = len(ids)
    t1 = time.time()
    while len(out) < max_new and out[-1] != eos_id:
        nxt, _, _ = model.forward(torch.tensor([[out[-1]]], dtype=torch.long), pos)
        out.append(int(nxt.flatten()[0]))
        pos += 1
    return out, t_pre, time.time() - t1


def score(logits, ids, sf, topk=32, ref_rec=None):
    """stream_ref.teacher と同じ式。返り値 (rec の数値部分, kl, agree)。"""
    lg = logits[0, sf - 1: len(ids) - 1].float()
    lp = torch.log_softmax(lg, -1)
    tgt = torch.tensor(ids[sf:], device=lp.device)
    tlp = lp.gather(1, tgt[:, None])[:, 0]
    v, ix = lp.topk(topk, -1)
    kl = agree = None
    extra = {}
    if ref_rec is not None:
        tv = ref_rec["top_v"].to(lp.device).float()
        ti = ref_rec["top_i"].to(lp.device).long()
        pt = tv.exp()
        qs = lp.gather(1, ti)
        kl = (pt * (tv - qs)).sum(-1)  # 教師 top-32 上の KL(P||Q) 近似
        agree = lp.argmax(-1) == ti[:, 0]
        extra = {"mass": pt.sum(-1).mean().item(), "nll_teacher": -ref_rec["tgt_lp"].mean().item()}
    rec = {"top_v": v.half().cpu(), "top_i": ix.int().cpu(), "tgt_lp": tlp.cpu(),
           "argmax_match": (ix[:, 0] == tgt).float().mean().item()}
    return rec, (None if kl is None else kl.cpu()), (None if agree is None else agree.cpu()), extra


def _check_ref(c, r, i):
    """学生: 教師 .pt の rec と、この入力から作ったプロンプト・画像が一致するか。ids は教師から取る。"""
    pl = r.get("prompt_len", r["score_from"])
    assert pl == c["prompt_len"], f"{i}: プロンプト長が教師と違う ({c['prompt_len']} vs {pl})"
    assert r["ids"][:pl] == c["ids"][:pl], f"{i}: プロンプトの ids が教師と違う"
    assert r.get("image_sha", c["image_sha"]) == c["image_sha"], f"{i}: 画像が教師と違う"
    if c["score_from"] is not None:
        assert r["ids"] == c["ids"] and r["score_from"] == c["score_from"], f"{i}: 回答が教師と違う"
    return list(r["ids"]), int(r["score_from"])


def run(model, tok, args, cases, out_path="", ref=None, topk=32, max_new=512, gen_out="", teacher_logits=True):
    eos = tok.eos_token_id
    n_layers, n_exp = args.n_layers, args.n_routed_experts
    recs, kls, ags = [], [], []
    gf = open(gen_out, "w") if gen_out else None
    for i, c in enumerate(cases):
        t0 = time.time()
        COUNTER.reset_seq()
        types, images = c["token_types"], c["images"]
        ids, sf, src, gen = c["ids"], c["score_from"], c["answer_source"], None
        truncated = False
        if ref is not None:
            ids, sf = _check_ref(c, ref["recs"][i], i)
            src = ref["recs"][i].get("answer_source", "ref")
        elif sf is None:
            mn = int(c.get("max_new") or max_new)
            assert len(ids) + mn <= args.max_seq_len, f"{i}: max_seq_len 不足 ({len(ids)} + {mn})"
            snap = ({k: v.copy() for k, v in SR.STORE.counts.items()}, {k: v.copy() for k, v in COUNTER.image.items()},
                    COUNTER.n_image_tokens)
            gen, tp, td = greedy_vl(model, ids, types, images, mn, eos)
            if teacher_logits:  # counts は採点の prefill だけ (stream_ref.teacher と同じ意味)。生成の分は戻す
                SR.STORE.counts.clear(); SR.STORE.counts.update(snap[0])
                COUNTER.image.clear(); COUNTER.image.update(snap[1])
                COUNTER.n_image_tokens = snap[2]
            truncated = gen[-1] != eos
            print(f"[gen] {i}: prompt {len(ids)} tok (image {sum(s['n_tokens'] for s in image_spans(c))}), "
                  f"prefill {tp:.1f}s, decode {len(gen)} tok {td:.1f}s{' (truncated)' if truncated else ''}", flush=True)
            ids, sf = ids + gen, len(ids)
        types = types + [TEXT] * (len(ids) - len(types))
        assert len(ids) <= args.max_seq_len, f"{i}: {len(ids)} > max_seq_len {args.max_seq_len}"
        text = tok.decode(ids[sf:])
        if gf is not None:
            gf.write(json.dumps({"i": i, "id": c["id"], "answer_source": src, "prompt_len": sf, "answer_ids": ids[sf:],
                                 "answer_text": text, "truncated": truncated}, ensure_ascii=False) + "\n")
            gf.flush()
        if not teacher_logits:
            continue
        COUNTER.reset_seq()  # 生成の decode は数え直さない (画像区間は prefill だけ)
        logits = prefill_logits(model, ids, types, images)
        rr = ref["recs"][i] if ref is not None else None
        rec, kl, ag, extra = score(logits, ids, sf, topk, rr)
        del logits
        sp = image_spans(c)
        for s, h in zip(sp, c["image_sha"]):
            s["sha256"] = h
        rec.update({"i": i, "id": c["id"], "src": c.get("src"), "ids": ids, "score_from": sf, "prompt_len": c["prompt_len"],
                    "answer_source": src, "answer_text": text, "truncated": truncated, "images": sp,
                    "image_sha": c["image_sha"], "n_image_tokens": sum(s["n_tokens"] for s in sp),
                    "counts_image": COUNTER.seq_array(n_layers, n_exp)})
        # build_sets の行の区間 (絶対位置。KL / tgt_lp の添字は 位置 − score_from) と分類
        rec.update({k: c[k] for k in ("spans", "q_spans", "answer_from", "lang", "cat", "sub") if c.get(k) is not None})
        recs.append(rec)
        kls.append(kl)
        ags.append(ag)
        dt = time.time() - t0
        if kl is not None:
            print(f"[student] {i}: KL {kl.mean().item():.4f} (p50 {kl.median().item():.4f}, p99 {kl.quantile(0.99).item():.3f}) "
                  f"top1-agree {ag.float().mean().item():.3f} nll {-rec['tgt_lp'].mean().item():.4f} vs teacher "
                  f"{extra['nll_teacher']:.4f} mass {extra['mass']:.3f} images {len(sp)}", flush=True)
        print(f"[teacher] {i}: {len(ids)} tok (image {rec['n_image_tokens']}), scored {len(ids) - sf}, "
              f"nll {-rec['tgt_lp'].mean().item():.4f}, top1 {rec['argmax_match']:.3f}, {dt:.1f}s", flush=True)
        total = {k: v.copy() for k, v in SR.STORE.counts.items()}
        ctext, cimg = COUNTER.split(total)
        if out_path:
            torch.save({"recs": recs, "counts": total, "counts_text": ctext, "counts_image": cimg,
                        "n_image_tokens": COUNTER.n_image_tokens, "vl": True, "kl": kls, "agree": ags}, out_path)
    if gf is not None:
        gf.close()
    if ref is not None and recs:
        kl = torch.cat(kls)
        ag = torch.cat(ags).float()
        print(f"[student] ALL: KL mean {kl.mean().item():.4f} median {kl.median().item():.4f} p99 {kl.quantile(0.99).item():.3f} "
              f"top1-agree {ag.mean().item():.4f} (n={kl.numel()})", flush=True)
    return recs


def to_openai_request(rec, base_dir, model="deepseek-ai/DeepSeek-V4.1-Flash", max_tokens=512, thinking="chat"):
    """ds41rt 用の /v1/chat/completions 本体。ds41rt は data: と http(s) しか受けないので
    ローカルの画像は data:<mime>;base64 に埋める (ds41rt-api native_v41/images.rs)。"""
    msgs = resolve_image_paths(rec["messages"], base_dir)
    mime = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp", ".gif": "gif"}

    def emb(url):
        if not isinstance(url, str) or url.startswith(("data:", "http://", "https://")):
            return url
        kind = mime.get(os.path.splitext(url)[1].lower(), "png")
        return f"data:image/{kind};base64," + base64.b64encode(open(url, "rb").read()).decode()

    if msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]  # 回答 (参照) は送らない。生成させる
    for m in msgs:
        for b in m.get("content") if isinstance(m.get("content"), list) else []:
            if isinstance(b, dict) and b.get("type") == "image_url":
                iu = b["image_url"]
                b["image_url"] = {"url": emb(iu if isinstance(iu, str) else iu.get("url", ""))}
    body = {"model": model, "messages": msgs, "temperature": 0, "max_tokens": int(rec.get("max_new") or max_tokens)}
    # ds41rt は thinking を省くと思考ありで答える (ds41rt-api native_v41.rs:261 with_default_thinking_mode(true))。
    # ハーネスと同じ書式にするため、行 (無ければ CLI) の thinking_mode を明示する
    tm = rec.get("thinking_mode") or thinking
    body["thinking"] = {"type": "enabled" if tm == "thinking" else "disabled"}
    if tm == "thinking" and rec.get("reasoning_effort"):
        body["reasoning_effort"] = rec["reasoning_effort"]
    if rec.get("tools"):
        body["tools"] = rec["tools"]
    return body


# --------------------------------------------------------------------------- main
def plan_summary(cases, args, max_new):
    for i, c in enumerate(cases):
        sp = image_spans(c)
        need = len(c["ids"]) + (int(c.get("max_new") or max_new) if c["score_from"] is None else 0)
        warn = []
        if len(sp) > MAX_IMAGES:
            warn.append(f"images {len(sp)} > {MAX_IMAGES} (ds41rt の上限)")
        if need > args.max_seq_len:
            warn.append(f"need {need} > max_seq_len {args.max_seq_len}")
        print(f"[plan] {i} {c['id']}: ids {len(c['ids'])} prompt {c['prompt_len']} answer "
              f"{'gen' if c['score_from'] is None else len(c['ids']) - c['score_from']} "
              f"images {[(s['start'], s['n_tokens'], s['n_llm_h'], s['n_llm_w']) for s in sp]} {' '.join(warn)}", flush=True)


def main(argv=None):
    global COUNTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl-in", required=True, help="jsonl {messages, answer?} (既存の {ids, score_from} も可)")
    ap.add_argument("--image-root", default="", help="相対パスの基準 (既定は jsonl のディレクトリ)")
    ap.add_argument("--thinking", default="chat", choices=["chat", "thinking"])
    ap.add_argument("--teacher-out", default="", help="出力 .pt (教師 / 学生)")
    ap.add_argument("--teacher-ref", default="", help="教師の .pt (指定すると学生として採点)")
    ap.add_argument("--exl3-pack", default="")
    ap.add_argument("--gen-out", default="", help="回答 (生成 / 与えられたもの) の jsonl")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=16384)
    ap.add_argument("--expert-budget-gb", type=float, default=60)
    ap.add_argument("--plan-only", action="store_true", help="GPU を使わず入力の検査だけ")
    ap.add_argument("--score-q", action="store_true",
                    help="build_sets の行で q_spans (画像の直後の問い) からも採点する (教師と学生で揃えること)")
    ap.add_argument("--export-openai", default="", help="ds41rt 用の要求本体 jsonl を書いて終わる (画像は data: URL)")
    a = ap.parse_args(argv)
    if a.export_openai:
        base = a.image_root or os.path.dirname(os.path.abspath(a.vl_in))
        with open(a.export_openai, "w") as f:
            for l in open(a.vl_in):
                if l.strip() and "messages" in (r := json.loads(l)):
                    f.write(json.dumps({"id": r.get("id"), "body": to_openai_request(r, base, max_tokens=a.max_new,
                                                                                         thinking=a.thinking)},
                                       ensure_ascii=False) + "\n")
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SR.CKPT)
    over = {"max_batch_size": 1, "max_seq_len": a.max_seq_len, "temperature": 0.0}
    margs = load_model_args(over)
    cases = load_vl_cases(a.vl_in, tok, margs, a.thinking, a.image_root)  # build の前に検査 (失敗を早く)
    if a.score_q:
        cases = [apply_score_q(c) for c in cases]
    plan_summary(cases, margs, a.max_new)
    if a.plan_only:
        return
    ref = torch.load(a.teacher_ref, weights_only=False) if a.teacher_ref else None
    if ref is not None:
        assert len(ref["recs"]) >= len(cases), "教師 .pt の系列数が足りない"
    if not a.teacher_out and not a.gen_out:
        raise SystemExit("--teacher-out か --gen-out が要る")
    model, tok, args = SR.build(over, a.expert_budget_gb, exl3_pack=a.exl3_pack)
    assert model.vision is not None, "vision tower が無い (config の vision_n_layers)"
    SR._patch_head()
    COUNTER = RouteCounter()
    COUNTER.install(M)
    SR.STORE.counts_image = COUNTER.image  # STORE からも引けるように (counts はそのまま全体)
    run(model, tok, args, cases, a.teacher_out, ref=ref, max_new=a.max_new, gen_out=a.gen_out,
        teacher_logits=bool(a.teacher_out))
    print("[experts]", SR.STORE.stats, flush=True)
    print("[route] image tokens (layer 0):", COUNTER.n_image_tokens, flush=True)


if __name__ == "__main__":
    main()
