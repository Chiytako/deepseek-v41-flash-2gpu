"""参照ハーネスでのサンプリング生成と、on-policy の中国語質量の記録 (r5_zh_mixing.md §5)。

stream_ref.py の build() で原重み (または --exl3-pack で EXL3 パック) を組み、プロンプト集
(tools/zh_promptset.py の prompts.jsonl、"harness": true の 20 本) を V4.1 の encoding.py で描画して生成する。
各ステップで、全語彙の softmax から zh_only (tools/zh_only_tokens.json の zs ∪ zt) に置かれた確率を記録する。
teacher-forced (zh_teacher_mass.py) と違って top-32 で切らず、モデル自身の出力の上での値 (on-policy) になる。

出力 jsonl は tools/zh_gen_eval.py score でそのまま採点できる形 (reasoning / content / tool_calls 無し) に、
"trace" (ステップごとの zh 質量の要約) を足したもの。

GPU 版 (LIL イメージ内、stream_ref.py と同じ環境変数 DSV41_REF / DSV41_CKPT):
  python3 harness/steer_zhgen.py --prompts work/zh/promptset/prompts.jsonl --only-harness \
      --temp 1.0 --top-p 1.0 --seed 0 --max-new 1024 --out work/zh/gen/ref_t1.0.jsonl [--exl3-pack /packs/v0]
CPU 版 (描画とトークン数の確認だけ。torch もモデルも使わない):
  tools/.venv/bin/python -I harness/steer_zhgen.py --self-test
速度の目安: 1.4〜1.75 tok/s (r3_japanese §0)。20 本 × 1,024 トークンで 3.3〜4 時間 + prefill。
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import importlib.util
import json
import os
import sys
import time

ROOT = DSV41
CKPT = os.environ.get("DSV41_CKPT", "/model")
LOCAL_MODEL = MODELS + "/DeepSeek-V4.1-Flash-1017"
THINK_END = 128822
EFFORT = {"low": "low", "minimal": "low", "none": None, "medium": "high", "high": "high", "max": "max"}


def _encoding():
    for base in (os.path.join(CKPT, "encoding"), os.path.join(LOCAL_MODEL, "encoding")):
        p = os.path.join(base, "encoding.py")
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("v41_encoding", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise FileNotFoundError("encoding.py が見つからない")


def norm_messages(messages, tools):
    """tools/x2_extract.py の norm_messages と同じ規則 (画像は除き、tools は system に付ける)。"""
    out = []
    for m in messages:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            txt = "\n\n".join((b.get("text") or "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            m["content"] = txt
        elif c is None:
            m["content"] = ""
        if m.get("role") == "developer":
            m["role"] = "system"
        tc = m.get("tool_calls")
        if tc:
            fixed = []
            for x in tc:
                f = dict(x.get("function") or {})
                if not isinstance(f.get("arguments"), str):
                    f["arguments"] = json.dumps(f.get("arguments") or {}, ensure_ascii=False)
                fixed.append({"type": "function", "function": f})
            m["tool_calls"] = fixed
        out.append(m)
    if tools:
        if out and out[0].get("role") == "system":
            out[0] = dict(out[0], tools=tools)
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tools})
    return out


def render(enc, p, thinking_override=None):
    th = p.get("thinking", True) if thinking_override is None else thinking_override
    mode = "thinking" if th else "chat"
    eff = EFFORT.get(str(p.get("reasoning_effort") or "high"), "high") if th else None
    return enc.encode_messages(norm_messages(p["messages"], p.get("tools")), thinking_mode=mode, reasoning_effort=eff)


def load_prompts(path, only_harness, ids):
    P = [json.loads(l) for l in open(path, encoding="utf-8")]
    if only_harness:
        P = [p for p in P if p.get("harness")]
    if ids:
        want = set(ids.split(","))
        P = [p for p in P if p["id"] in want]
    return P


def self_test(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(LOCAL_MODEL, "tokenizer.json"))
    enc = _encoding()
    P = load_prompts(a.prompts, True, a.ids)
    zh = json.load(open(f"{ROOT}/tools/zh_only_tokens.json"))["zh_only"]
    print(f"[self-test] prompts {len(P)}、zh_only {len(zh)} トークン")
    tot = 0
    for p in P:
        s = render(enc, p)
        n = len(tok.encode(s, add_special_tokens=False).ids)
        tot += n
        print(f"  {p['id']:14s} {n:6d} tok (x2 plen {p.get('plen')})  末尾 {s[-40:]!r}")
    print(f"[self-test] prefill 合計 {tot} トークン。prefill 60 tok/s なら {tot / 60 / 60:.1f} 分")
    return 0


def run(a):
    import numpy as np
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import stream_ref as S

    enc = _encoding()
    P = load_prompts(a.prompts, a.only_harness, a.ids)
    model, tok, args = S.build({"max_batch_size": 1, "max_seq_len": a.max_seq_len, "temperature": 0.0},
                               a.expert_budget_gb, exl3_pack=a.exl3_pack)
    zh_ids = json.load(open(f"{ROOT}/tools/zh_only_tokens.json"))["zh_only"]
    V = None
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out, encoding="utf-8")}
    f = open(a.out, "a", encoding="utf-8")
    for p in P:
        if p["id"] in done:
            continue
        text = render(enc, p, None if a.thinking == "auto" else a.thinking == "on")
        ids = tok.encode(text, add_special_tokens=False)
        g = torch.Generator(device="cpu").manual_seed(a.seed * 100003 + hash(p["id"]) % 100003)
        t0 = time.time()
        with torch.inference_mode():
            _, logits, _ = model.forward(torch.tensor([ids], dtype=torch.long), 0)
            t_pre = time.time() - t0
            out, trace = [], []
            pos = len(ids)
            region = "think" if text.endswith("<think>") else "answer"
            while len(out) < a.max_new:
                lg = logits[0].float().cpu()
                if V is None:
                    V = lg.numel()
                    mask = torch.zeros(V, dtype=torch.bool)
                    mask[torch.tensor([i for i in zh_ids if i < V])] = True
                p1 = torch.softmax(lg, -1)
                m1 = float(p1[mask].sum())
                if a.temp <= 0:
                    q = torch.zeros_like(p1)
                    q[int(lg.argmax())] = 1.0
                else:
                    q = torch.softmax(lg / a.temp, -1)
                    if a.top_p < 1.0:
                        sv, si = torch.sort(q, descending=True)
                        keep = (torch.cumsum(sv, 0) - sv) < a.top_p
                        q2 = torch.zeros_like(q)
                        q2[si[keep]] = sv[keep]
                        q = q2 / q2.sum()
                ms = float(q[mask].sum())
                t = int(torch.multinomial(q, 1, generator=g))
                trace.append((region, round(m1, 6), round(ms, 6), int(mask[t])))
                out.append(t)
                if t == THINK_END:
                    region = "answer"
                if t == tok.eos_token_id:
                    break
                _, logits, _ = model.forward(torch.tensor([[t]], dtype=torch.long), pos)
                pos += 1
        dt = time.time() - t0
        if THINK_END in out:
            k = out.index(THINK_END)
            reasoning, content = tok.decode(out[:k]), tok.decode(out[k + 1:], skip_special_tokens=True)
        else:
            reasoning, content = ("", tok.decode(out, skip_special_tokens=True)) if region == "answer" else (tok.decode(out), "")
        summ = {}
        for reg in ("think", "answer"):
            tr = [x for x in trace if x[0] == reg]
            if tr:
                summ[reg] = {"n": len(tr), "m1_sum": sum(x[1] for x in tr), "msample_sum": sum(x[2] for x in tr),
                             "zh_sampled": sum(x[3] for x in tr),
                             "P_ge1_sample": 1 - float(np.exp(sum(np.log1p(-min(x[2], 0.999999)) for x in tr)))}
        rec = {"id": p["id"], "cat": p["cat"], "seed": a.seed, "label": a.label, "temp": a.temp, "top_p": a.top_p,
               "reasoning": reasoning, "content": content, "tool_calls": [],
               "finish": "stop" if out and out[-1] == tok.eos_token_id else "length", "plen": len(ids),
               "prefill_s": t_pre, "sec": dt, "trace": summ}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        print(f"[gen] {p['id']} prompt {len(ids)} tok prefill {t_pre:.0f}s, 生成 {len(out)} tok {dt - t_pre:.0f}s, trace {summ}", flush=True)
    print("[experts]", S.STORE.stats, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=f"{ROOT}/work/zh/promptset/prompts.jsonl")
    ap.add_argument("--only-harness", action="store_true")
    ap.add_argument("--ids", default="")
    ap.add_argument("--label", default="ref")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--max-seq-len", type=int, default=40960)
    ap.add_argument("--thinking", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--expert-budget-gb", type=float, default=60)
    ap.add_argument("--exl3-pack", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test(a)
    if not a.out:
        a.out = f"{ROOT}/work/zh/gen/{a.label}_harness_t{a.temp}_p{a.top_p}_s{a.seed}.jsonl"
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
