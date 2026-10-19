"""Build Japanese-weighted calibration / distillation sets and a JA-KL eval set for DeepSeek-V4.1-Flash-1017.

CPU only. No GPU, no docker. Inputs are local except the public corpora fetched beforehand with
`fetch_ja_corpus.py` (HF datasets-server rows API, licences in <src_dir>/_sources.json).

Outputs (<out_dir>):
  calib64k.jsonl   ~64k tokens, a strict subset (the first rows of every category) of calib256k
  calib256k.jsonl  ~256k tokens
  jakl32.jsonl     32 Japanese eval rows, disjoint from the calib sets and from work/teacher/{agent24,ja16}
  calib_meta.json  per-file counts / token totals / language ratios / exclusions / script sha256

Row format (one JSON object per line; the harness `stream_ref.py --teacher-in` reads ids/score_from/src):
  ids         token ids (<= seq_len), every packed segment starts with <｜begin▁of▁sentence｜>
  score_from  first scored position (harness scores [score_from, len))
  lang        ja | agent | code | en | multi   (category of the row)
  src         ";"-joined provenance of the packed segments
  sub         sub-category (ja-agent, ja-assist, ja-short, ja-chat, ja-wiki, ja-tech, ja-aozora, req-agent,
              sud-agentic, req-code, sud-code, sud-en, sud-multi)
  spans       [[a, b), ...] assistant / output token spans (precise loss region for packed rows)
  ja_frac     Japanese character share of the row text (kana+kanji over non-space chars, 0 if no kana)
  ja_tok      share of tokens whose text has kana or kanji (0 if the row has no kana); a per-token mask for the
              L2 Japanese weight (c_t = 2.0) is CJK_TOK[ids] restricted to rows with ja_tok > 0

Composition: SPEC_QUANT_JA.md §2.1 (r3_japanese.md §7), with the 5% on-policy share replaced by rendered
Japanese instruction -> Japanese answer conversations (V4.1 encoding.py) and the 2% image share moved to
agent (the harness has no vision path for these rows).

Hold-out: request log rows with ts <= --split-ts are calib candidates, later ones JA-KL candidates; task
hashes (first user utterance, r3ja rows.jsonl `task_h`) used by JA-KL are excluded from calib; anything
that shares content-anchored 16-gram shingles with agent24/ja16 (or with JA-KL, for calib) is rejected.

usage: build_calib.py [--out-dir DIR] [--src-dir DIR] [--seed N] [--dry]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import glob
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = DSV41
MODEL = MODELS + "/DeepSeek-V4.1-Flash-1017"
REQ = ROOT + "/work/x2/req"
ROWS = ROOT + "/work/r3ja/rows.jsonl"
JOIN = ROOT + "/tools/r3ja/r3ja_join.py"
SUD = ROOT + "/ext/sudoingx_dsv41/data/caldata"
TEACHER = ROOT + "/work/teacher"
TECH_GLOBS = [ROOT + "/research/*.md"] + [g for g in _os.environ.get("EXTRA_TECH_GLOBS", "").split(":") if g]  # publish: the private GLM-era research/*.md was used here too

BOS = 0
CJK_TOK = None  # bool[vocab], set by Env
# SPEC_QUANT_JA.md §2.1 table (ratios of total tokens). on-policy 5% -> ja-chat, vl 2% -> agent.
MIX = {
    "ja-agent": ("ja", 0.14), "ja-assist": ("ja", 0.12), "ja-short": ("ja", 0.03), "ja-chat": ("ja", 0.05),
    "ja-wiki": ("ja", 0.03), "ja-tech": ("ja", 0.02), "ja-aozora": ("ja", 0.01),
    "req-agent": ("agent", 0.26), "sud-agentic": ("agent", 0.08),
    "req-code": ("code", 0.07), "sud-code": ("code", 0.06),
    "sud-en": ("en", 0.08), "sud-multi": ("multi", 0.05),
}
JAKL = {"ja-agent": 8, "ja-assist": 8, "ja-short": 2, "ja-chat": 4, "ja-wiki": 4, "ja-tech": 4, "ja-aozora": 2}
REQ_SUB = {"A0_agent_impl": "agent", "A1_agent_review": "agent", "A2_agent_verify": "agent", "F_subagent": "agent",
           "B_assistant": "assist", "D_desktop": "assist", "C3_nosys_other": "assist",
           "H_other": "assist", "E_cli_en": "assist",
           "C1_status_summary": "short", "C2_plan_json": "short", "G_memory_title_util": "short"}
SUD_EN = {"chat", "knowledge", "writing", "math", "instruct", "long_context"}


# ----------------------------------------------------------------------------------------------- text stats
def charstats(text):
    """(non-space chars, kana, kanji, code symbols) — same character classes as x2_prep.classify."""
    n = kana = kanji = sym = 0
    for ch in text:
        if ch.isspace():
            continue
        n += 1
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or 0xFF66 <= o <= 0xFF9D:
            kana += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            kanji += 1
        elif ch in "{}[]();=<>_/\\|$#&*+`%^~":
            sym += 1
    return n, kana, kanji, sym


def ja_frac(st):
    """Japanese share: (kana+kanji)/n, but 0 unless kana is present (separates zh, which has no kana)."""
    n, kana, kanji, _ = st
    if n == 0 or kana < 3 or kana < 0.15 * (kana + kanji):
        return 0.0
    return (kana + kanji) / n


def classify(st):
    """ja if Japanese >= 25% of the chars (SPEC §2.1 / JA §7), else code if symbols >= 6%, else other."""
    n, _, _, sym = st
    if ja_frac(st) >= 0.25:
        return "ja"
    if n and sym / n >= 0.06:
        return "code"
    return "other"


SPECIAL_RE = re.compile(r"<｜[^｜]{1,40}｜>")


def strip_special(text):
    return SPECIAL_RE.sub(" ", text)


# ------------------------------------------------------------------------------------------------- shingles
_P = np.uint64(0x100000001B3)


def shingles(ids, k=16, keep_mod=8):
    """Content-anchored k-gram hashes (keep hash % keep_mod == 0): offset-independent near-dup detection."""
    a = np.asarray(ids, dtype=np.uint64)
    if a.size < k:
        return set()
    m = a.size - k + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * _P + (a[j:j + m] + np.uint64(1))
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    h = h[(h % np.uint64(keep_mod)) == 0]
    return set(h.tolist())


USER_ID = 128803  # <｜User｜> (checked in Env.__init__)


def content_shingles(ids):
    """Shingles of the conversation content: skips everything before the first <｜User｜> (system prompt and the
    encoder's fixed tool/DSML instructions are shared boilerplate and would make unrelated items look duplicated)."""
    try:
        i = ids.index(USER_ID)
    except ValueError:
        i = 0
    return shingles(ids[i:])


def overlap(s, ref):
    if not s:
        return 0.0
    return len(s & ref) / len(s)


# ------------------------------------------------------------------------------------------ tokenizer etc.
class Env:
    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(MODEL + "/tokenizer.json")
        spec = importlib.util.spec_from_file_location("v41_encoding", MODEL + "/encoding/encoding.py")
        self.enc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.enc)
        tid = self.tok.token_to_id
        self.sp = {k: tid(k) for k in ("<｜Assistant｜>", "<｜User｜>", "<｜System｜>", "<｜end▁of▁sentence｜>",
                                       "<｜begin▁of▁sentence｜>")}
        assert self.sp["<｜begin▁of▁sentence｜>"] == BOS, self.sp
        assert self.sp["<｜User｜>"] == USER_ID, self.sp
        n_vocab = self.tok.get_vocab_size()
        flag = np.zeros(n_vocab, dtype=bool)  # token text contains kana or a CJK ideograph
        for i in range(n_vocab):
            t = self.tok.decode([i], skip_special_tokens=False)
            flag[i] = any(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF for c in t)
        self.cjk_tok = flag
        global CJK_TOK
        CJK_TOK = flag
        self.nl_ids = {i for t, i in self.tok.get_vocab().items() if t.endswith("Ċ")}  # tokens ending in "\n"
        self._u32 = {}

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode([int(x) for x in ids], skip_special_tokens=False)

    def u32(self, name):
        if name not in self._u32:
            self._u32[name] = np.memmap(os.path.join(REQ, name), dtype=np.uint32, mode="r")
        return self._u32[name]

    def render(self, messages, tools=None, thinking=False, effort=None):
        """V4.1 encoding.py render -> (ids, assistant spans [a, b)); tools go on the system message."""
        msgs = [dict(m) for m in messages]
        if tools:
            if not msgs or msgs[0]["role"] != "system":
                msgs.insert(0, {"role": "system", "content": ""})
            msgs[0]["tools"] = tools
        mode = "thinking" if thinking else "chat"
        text = self.enc.encode_messages(msgs, thinking_mode=mode, reasoning_effort=effort if thinking else None)
        ids = self.encode(text)
        spans, inside, a = [], False, 0
        A, U, S, E = (self.sp[k] for k in ("<｜Assistant｜>", "<｜User｜>", "<｜System｜>", "<｜end▁of▁sentence｜>"))
        for i, t in enumerate(ids):
            if t in (U, S) and inside:
                spans.append([a, i])
                inside = False
            if t == A:
                inside, a = True, i + 1
            elif t == E and inside:
                spans.append([a, i + 1])
                inside = False
        if inside:
            spans.append([a, len(ids)])
        return ids, [s for s in spans if s[1] > s[0]]


# ------------------------------------------------------------------------------------------------- segments
class Seg:
    """One packable unit: ids (starts with BOS), output spans, provenance, text stats."""
    __slots__ = ("ids", "spans", "src", "sub", "st", "key", "sh", "jt")

    def __init__(self, ids, spans, src, sub, st, key):
        self.ids, self.spans, self.src, self.sub, self.st, self.key = ids, spans, src, sub, st, key
        self.sh = None
        self.jt = float(CJK_TOK[np.asarray(ids)].mean()) if CJK_TOK is not None and ids else 0.0


def load_requests(split_ts):
    rows = {}
    for line in open(ROWS):
        r = json.loads(line)
        rows[(r["file"], r["ts"])] = r
    src = open(JOIN).read()
    ns = {}
    exec(src[src.index("def cat"):src.index("def lang")], ns)  # r3ja_join.cat (task classes A0..H)
    cat = ns["cat"]
    reqs = []
    for mf in sorted(glob.glob(REQ + "/*.meta.json")):
        name = os.path.basename(mf)[:-len(".meta.json")] + ".u32"
        off = 0
        for m in json.load(open(mf)):
            L = m["plen"] + m["olen"]
            r = rows.get((m["file"], m["ts"]))
            if r is not None and m["n_img"] == 0 and m["olen"] >= 8 and m["finish"] != "abort":
                reqs.append({"src": name, "off": off, "plen": m["plen"], "olen": m["olen"], "ts": m["ts"],
                             "cat": cat(r), "task_h": r["task_h"], "tools": m["tools"],
                             "half": "calib" if m["ts"] <= split_ts else "eval"})
            off += L
    return reqs


def req_window(env, r, seg_len, out_cap):
    """Tail window of one request: [BOS] + req[start:end], end = plen + min(olen, out_cap)."""
    arr = env.u32(r["src"])
    base = r["off"]
    end = r["plen"] + min(r["olen"], out_cap)
    start = max(0, end - (seg_len - 1))
    body = arr[base + start: base + end].tolist()
    if start == 0 and body and body[0] == BOS:
        ids = body
        sf = r["plen"]
    else:
        ids = [BOS] + body
        sf = r["plen"] - start + 1
    return ids, sf, start, end


def req_segment(env, r, seg_len, out_cap, sub_prefix=None):
    ids, sf, start, end = req_window(env, r, seg_len, out_cap)
    out_txt = strip_special(env.decode(ids[sf:]))
    st = charstats(out_txt)
    c = classify(st)
    group = REQ_SUB.get(r["cat"], "assist")
    if c == "ja":
        sub = "ja-" + group
    elif c == "code":
        sub = "req-code"
    else:
        sub = "req-agent" if (r["tools"] or group == "agent") else "req-en"
    wst = charstats(strip_special(env.decode(ids)))
    s = Seg(ids, [[sf, len(ids)]], f"{r['src']}@{r['off']}+[{start},{end})", sub, wst, r["task_h"])
    s.sh = content_shingles(ids)
    return s


def teacher_windows():
    out = []
    for f in ("agent24.jsonl", "ja16.jsonl"):
        p = os.path.join(TEACHER, f)
        if os.path.exists(p):
            out += [json.loads(line) for line in open(p)]
    return out


def doc_segment(env, text, seg_len, src, sub, key, rng, random_start=True):
    """Plain document: BOS + a chunk of <= seg_len-1 tokens starting at a paragraph boundary."""
    ids = env.encode(text)
    if len(ids) > seg_len - 1 and random_start:
        # candidate starts at newline tokens in the first part of the document
        nl = [i for i, t in enumerate(ids[: len(ids) - (seg_len - 1)]) if t in env.nl_ids]
        st = rng.choice(nl) + 1 if nl else 0
    else:
        st = 0
    chunk = [BOS] + ids[st: st + seg_len - 1]
    s = Seg(chunk, [[1, len(chunk)]], src + f"+[{st},{st + len(chunk) - 1})", sub,
            charstats(env.decode(chunk[1:])), key)
    s.sh = content_shingles(chunk)
    return s


def conv_segment(env, messages, seg_len, src, sub, key, tools=None, thinking=False, effort=None):
    try:
        ids, spans = env.render(messages, tools, thinking, effort)
    except Exception as e:  # noqa: BLE001
        print("render error", src, type(e).__name__, e, file=sys.stderr)
        return None
    if len(ids) > seg_len:
        ids = ids[:seg_len]
        spans = [[a, min(b, seg_len)] for a, b in spans if a < seg_len]
    spans = [s for s in spans if s[1] > s[0]]
    if not spans:
        return None
    s = Seg(ids, spans, src, sub, charstats(strip_special(env.decode(ids))), key)
    s.sh = content_shingles(ids)
    return s


# ------------------------------------------------------------------------------------------ public corpora
def load_jsonl(p):
    return [json.loads(line) for line in open(p)] if os.path.exists(p) else []


def h8(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def wiki_items(src_dir):
    out = []
    for r in load_jsonl(os.path.join(src_dir, "wiki_ja.jsonl")):
        t, x = r["title"], r["text"]
        if len(x) < 800 or "一覧" in t or "曖昧さ回避" in x[:300] or t.startswith(("Wikipedia:", "Template:")):
            continue
        out.append((f"wiki_ja:{r['id']}:{t}", t + "\n\n" + x, "wiki:" + r["id"]))
    return out


def aozora_items(src_dir):
    out = []
    for r in load_jsonl(os.path.join(src_dir, "aozora.jsonl")):
        m = r["meta"]
        if m.get("作品著作権フラグ") != "なし" or m.get("文字遣い種別") != "新字新仮名" or len(r["text"]) < 1500:
            continue
        head = f"{m.get('作品名', '')}\n{m.get('姓', '')}{m.get('名', '')}\n\n"
        out.append((f"aozora:{m.get('作品ID')}:{m.get('作品名')}", head + r["text"],
                    "aozora:" + str(m.get("人物ID")) + ":" + str(m.get("作品ID"))))
    return out


def tech_items():
    out = []
    for g in TECH_GLOBS:
        for p in sorted(glob.glob(g)):
            x = open(p, encoding="utf-8", errors="replace").read()
            if ja_frac(charstats(x)) < 0.2 or len(x) < 3000:
                continue
            out.append((_os.path.relpath(p, _os.path.dirname(ROOT)), x, "tech:" + p))
    return out


def chat_items(src_dir):
    """Japanese instruction -> Japanese answer conversations: (src, messages, key, tools, thinking, effort, native)."""
    out = []
    eff_map = {"low": 50, "high": 75}  # same mapping as sudoingX caldata/build.py
    for f in ("conversations.jsonl", "instruct_v3.jsonl"):
        for r in load_jsonl(os.path.join(SUD, f)):
            if r["domain"] == "multilingual_ja":
                out.append({"src": f"sudoingx/{f}:{r['id']}", "messages": r["messages"], "key": "sud:" + r["id"],
                            "tools": r.get("tools"), "thinking": r["thinking"], "effort": eff_map.get(r["effort"]),
                            "native": True, "split": r["split"]})
    for r in load_jsonl(os.path.join(src_dir, "oasst2_ja.jsonl")):
        conv = [{"role": c["role"], "content": c["content"]} for c in r["conversations"]
                if c["role"] in ("user", "assistant") and c["content"]]
        if len(conv) < 2 or conv[0]["role"] != "user":
            continue
        if ja_frac(charstats(" ".join(c["content"] for c in conv))) < 0.25:
            continue
        out.append({"src": f"oasst2_ja:{r['_row']}", "messages": conv, "key": "oasst:" + h8(conv[0]["content"]),
                    "native": False})
    for r in load_jsonl(os.path.join(src_dir, "dolly_ja.jsonl")):
        u = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        if not r.get("output") or ja_frac(charstats(u + r["output"])) < 0.25:
            continue
        out.append({"src": f"dolly_ja:{r.get('index', r['_row'])}", "key": "dolly:" + h8(r["instruction"]),
                    "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": r["output"]}],
                    "native": False})
    return out


def sud_items(domains):
    eff_map = {"low": 50, "high": 75}
    out = []
    for f in ("conversations.jsonl", "instruct_v3.jsonl"):
        for r in load_jsonl(os.path.join(SUD, f)):
            if r["split"] == "cal" and r["domain"] in domains:
                out.append({"src": f"sudoingx/{f}:{r['id']}", "messages": r["messages"], "key": "sud:" + r["id"],
                            "tools": r.get("tools"), "thinking": r["thinking"], "effort": eff_map.get(r["effort"]),
                            "domain": r["domain"]})
    return out


# ------------------------------------------------------------------------------------------------- packing
def make_row(segs, lang, sub):
    ids, spans, srcs = [], [], []
    for s in segs:
        o = len(ids)
        ids += s.ids
        spans += [[a + o, b + o] for a, b in s.spans]
        srcs.append(s.src)
    tot = [0, 0, 0, 0]
    for s in segs:
        tot = [x + y for x, y in zip(tot, s.st)]
    jf = ja_frac(tuple(tot))
    jt = float(CJK_TOK[np.asarray(ids)].mean()) if jf > 0 else 0.0
    return {"ids": ids, "score_from": spans[0][0] if spans else 1, "lang": lang, "src": ";".join(srcs),
            "sub": sub, "spans": spans, "ja_frac": round(jf, 4), "ja_tok": round(jt, 4)}


def pack(pool, budget, seq_len, trunc_ok):
    """Greedy first-fit packing of segments (in pool order) into rows until `budget` tokens.

    Consumes from `pool` (list, popped from the front). Returns rows (lists of Seg)."""
    rows, cur, used = [], [], 0
    while pool and used < budget:
        room = int(min(seq_len - sum(len(s.ids) for s in cur), budget - used))
        s = pool[0]
        if len(s.ids) <= room:
            cur.append(pool.pop(0))
            used += len(s.ids)
            continue
        if trunc_ok and room >= 256:
            s = pool.pop(0)
            s.ids = s.ids[:room]
            s.spans = [[a, min(b, room)] for a, b in s.spans if a < room]
            cur.append(s)
            used += room
            continue
        if not cur:
            # an over-long segment that does not fit the remaining budget: take it if it overshoots < half
            if len(s.ids) - room < len(s.ids) / 2:
                cur.append(pool.pop(0))
                used += len(s.ids)
            else:
                break
            continue
        # try a later segment that fits (keeps rows full), else close the row
        for j in range(1, min(len(pool), 64)):
            if len(pool[j].ids) <= room:
                cur.append(pool.pop(j))
                used += len(cur[-1].ids)
                break
        else:
            rows.append(cur)
            cur = []
            continue
        if seq_len - sum(len(x.ids) for x in cur) < 128:
            rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)
    return rows


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=ROOT + "/work/calib")
    ap.add_argument("--src-dir", default=ROOT + "/work/calib/src")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--seg-len", type=int, default=2048, help="max tokens of one packed calib segment")
    ap.add_argument("--out-cap", type=int, default=1024, help="max output tokens in a request window")
    ap.add_argument("--small", type=int, default=65536)
    ap.add_argument("--large", type=int, default=262144)
    ap.add_argument("--split-ts", default="2026-10-15T09:32")
    ap.add_argument("--task-cap", type=int, default=2, help="max calib segments per task hash")
    ap.add_argument("--dup-eval", type=float, default=0.02, help="max shingle overlap with eval windows")
    ap.add_argument("--jakl-dup-teacher", type=float, default=0.30,
                    help="JA-KL vs agent24/ja16 shingle overlap cap (two eval sets may share repo content; "
                         "this only blocks the same conversation)")
    ap.add_argument("--dup-self", type=float, default=0.30, help="max shingle overlap with chosen calib")
    ap.add_argument("--jakl-min-jatok", type=float, default=0.15,
                    help="JA-KL request windows: min share of Japanese tokens over the whole window")
    ap.add_argument("--jakl-min-jatok-agent", type=float, default=0.10,
                    help="same for ja-agent windows (tool/code context is denser)")
    ap.add_argument("--jakl-short-seg", type=int, default=384, help="window of one packed J-short item")
    ap.add_argument("--calib-min-jatok", type=float, default=0.08,
                    help="calib ja-* request windows: min share of Japanese tokens over the whole window")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    env = Env()
    os.makedirs(a.out_dir, exist_ok=True)

    # ---- exclusions: existing teacher eval windows
    tw = teacher_windows()
    excl_req = {(t["src"], t["off"]) for t in tw}
    eval_sh = set()
    for t in tw:
        eval_sh |= content_shingles(t["ids"])
    print(f"[excl] teacher windows {len(tw)}, shingles {len(eval_sh)}", flush=True)

    reqs = [r for r in load_requests(a.split_ts) if (r["src"], r["off"]) not in excl_req]
    rng.shuffle(reqs)
    by_half = defaultdict(list)
    for r in reqs:
        by_half[r["half"]].append(r)
    print(f"[req] calib-half {len(by_half['calib'])}, eval-half {len(by_half['eval'])}", flush=True)

    # =========================================================================================== JA-KL 32
    jak = defaultdict(list)
    jak_tasks, jak_sh = set(), set()
    need_req = {"ja-agent": JAKL["ja-agent"], "ja-assist": JAKL["ja-assist"]}
    short_segs = []
    seen_task = Counter()
    # pass 1: one window per task; pass 2 (only if short): up to two windows per task
    for ev_cap in (1, 2):
        for r in by_half["eval"]:
            if all(len(jak[k]) >= v for k, v in need_req.items()) and len(short_segs) >= 60:
                break
            g = REQ_SUB.get(r["cat"], "assist")
            if g == "short":
                if len(short_segs) >= 60 or ev_cap > 1:
                    continue
                s = req_segment(env, r, a.jakl_short_seg, 128)
                if s.sub == "ja-short" and overlap(s.sh, eval_sh) <= a.jakl_dup_teacher:
                    short_segs.append(s)
                continue
            k = "ja-" + g
            if len(jak.get(k, [])) >= need_req.get(k, 0) or seen_task[r["task_h"]] >= ev_cap:
                continue
            s = req_segment(env, r, a.seq_len, 1024)
            thr = a.jakl_min_jatok_agent if k == "ja-agent" else a.jakl_min_jatok
            if (s.sub != k or len(s.ids) < 1024 or s.jt < thr or overlap(s.sh, eval_sh) > a.jakl_dup_teacher
                    or overlap(s.sh, jak_sh) > 0.05):
                continue
            seen_task[r["task_h"]] += 1
            jak[k].append([s])
            jak_tasks.add(r["task_h"])
            jak_sh |= s.sh
    for s in short_segs:
        jak_tasks.add(s.key)
    short_rows = pack(list(short_segs), JAKL["ja-short"] * a.seq_len, a.seq_len, False)[: JAKL["ja-short"]]
    jak["ja-short"] = short_rows
    for rr in short_rows:
        for s in rr:
            jak_sh |= s.sh

    # documents: wiki / tech / aozora (split by item key hash so calib never sees the eval items)
    def split_items(items, n_eval, min_tokens):
        items = sorted(items, key=lambda x: h8(x[2] + str(a.seed)))
        ev, ca = [], []
        for it in items:
            if len(ev) < n_eval and len(it[1]) >= min_tokens:  # chars as a cheap pre-filter
                ids_len = len(env.encode(it[1]))
                if ids_len >= min_tokens:
                    ev.append(it)
                    continue
            ca.append(it)
        return ev, ca

    wiki = wiki_items(a.src_dir)
    tech = tech_items()
    aoz = aozora_items(a.src_dir)
    wiki_ev, wiki_ca = split_items(wiki, JAKL["ja-wiki"], a.seq_len)
    tech_ev, tech_ca = split_items(tech, JAKL["ja-tech"], a.seq_len)
    aoz_ev, aoz_ca = split_items(aoz, JAKL["ja-aozora"], a.seq_len)
    for k, ev in (("ja-wiki", wiki_ev), ("ja-tech", tech_ev), ("ja-aozora", aoz_ev)):
        for src, text, key in ev:
            s = doc_segment(env, text, a.seq_len, src, k, key, rng)
            s.spans = [[len(s.ids) - 2048, len(s.ids)]]  # score the last 2048 tokens (>= 2047 of context)
            jak[k].append([s])
            jak_sh |= s.sh
    print(f"[docs] wiki {len(wiki)} (eval {len(wiki_ev)}), tech {len(tech)} (eval {len(tech_ev)}), "
          f"aozora {len(aoz)} (eval {len(aoz_ev)})", flush=True)

    # chat: eval = sudoingX KLD-panel ja conversations (native V4.1) + oasst2-ja by first-prompt hash
    chats = chat_items(a.src_dir)
    chat_eval_keys = set()
    panel = [c for c in chats if c.get("native") and c.get("split") == "panel"]
    rest = sorted([c for c in chats if not (c.get("native") and c.get("split") == "panel")],
                  key=lambda c: h8(c["key"] + str(a.seed)))
    ev_segs = []
    for c in panel:
        s = conv_segment(env, c["messages"], a.seq_len, c["src"], "ja-chat", c["key"], c.get("tools"),
                         c.get("thinking", False), c.get("effort"))
        if s:
            ev_segs.append(s)
            chat_eval_keys.add(c["key"])
    oasst_ev = [c for c in rest if c["key"].startswith("oasst:")]
    pool_ev = []
    for c in oasst_ev:
        if sum(len(s.ids) for s in pool_ev) >= (JAKL["ja-chat"] - 1) * a.seq_len:
            break
        if c["key"] in chat_eval_keys:  # oasst2 branches of one thread share the first prompt
            continue
        s = conv_segment(env, c["messages"], 2048, c["src"], "ja-chat", c["key"])
        if s and len(s.ids) >= 300:
            pool_ev.append(s)
            chat_eval_keys.add(c["key"])
    chat_rows = pack(ev_segs + pool_ev, JAKL["ja-chat"] * a.seq_len, a.seq_len, False)[: JAKL["ja-chat"]]
    jak["ja-chat"] = chat_rows
    for rr in chat_rows:
        for s in rr:
            jak_sh |= s.sh
    # keys of chat segments that did not make it into a row go back to calib
    used_keys = {s.key for rr in chat_rows for s in rr}
    chat_eval_keys &= used_keys

    jakl_rows = []
    for k in JAKL:
        for segs in jak.get(k, []):
            row = make_row(segs, "ja", k)
            if k in ("ja-wiki", "ja-tech", "ja-aozora"):
                row["score_from"] = row["spans"][0][0]
            jakl_rows.append(row)
    print("[jakl] " + ", ".join(f"{k} {len(jak.get(k, []))}" for k in JAKL), flush=True)

    # =========================================================================================== calib pools
    pools = defaultdict(list)
    task_n = Counter()
    calib_sh = set()

    rej = Counter()

    def accept(s, cap=a.task_cap):
        if overlap(s.sh, eval_sh) > a.dup_eval:
            rej[(s.sub, "dup-teacher")] += 1
            return False
        if overlap(s.sh, jak_sh) > a.dup_eval:
            rej[(s.sub, "dup-jakl")] += 1
            return False
        if overlap(s.sh, calib_sh) > a.dup_self:
            rej[(s.sub, "dup-self")] += 1
            return False
        if task_n[s.key] >= cap:
            rej[(s.sub, "task-cap")] += 1
            return False
        task_n[s.key] += 1
        calib_sh.update(s.sh)
        return True

    want = {k: MIX[k][1] * a.large for k in MIX}
    req_subs = ("ja-agent", "ja-assist", "ja-short", "req-agent", "req-code")
    have = Counter()
    n_scan = Counter()
    for r in by_half["calib"]:
        if all(have[k] >= want[k] * 1.15 for k in req_subs):
            break
        if r["task_h"] in jak_tasks:
            continue
        g = REQ_SUB.get(r["cat"], "assist")
        short = g == "short"
        s = req_segment(env, r, 1024 if short else a.seg_len, 512 if short else a.out_cap)
        n_scan[s.sub] += 1
        if s.sub not in req_subs or have[s.sub] >= want[s.sub] * 1.15:
            continue
        if not short and len(s.ids) < a.seg_len // 2:
            continue
        if s.sub.startswith("ja-") and s.jt < a.calib_min_jatok:
            n_scan["low-jatok:" + s.sub] += 1
            continue
        if accept(s, cap=a.task_cap if not short else 1):
            pools[s.sub].append(s)
            have[s.sub] += len(s.ids)
    print(f"[req] scanned {dict(n_scan)}", flush=True)
    print(f"[req] pooled tokens {dict(have)}", flush=True)

    # chats (native first, then oasst2, dolly — interleaved so both sources appear in the small set)
    nat = [c for c in rest if c.get("native") and c["key"] not in chat_eval_keys]
    oas = [c for c in rest if c["key"].startswith("oasst:") and c["key"] not in chat_eval_keys]
    dol = [c for c in rest if c["key"].startswith("dolly:")]
    order = nat + [x for pair in zip(oas, dol + [None] * len(oas)) for x in pair if x is not None]
    for c in order:
        if have["ja-chat"] >= want["ja-chat"] * 1.15:
            break
        s = conv_segment(env, c["messages"], a.seg_len, c["src"], "ja-chat", c["key"], c.get("tools"),
                         c.get("thinking", False), c.get("effort"))
        if s and accept(s, cap=1):
            pools["ja-chat"].append(s)
            have["ja-chat"] += len(s.ids)

    for k, items in (("ja-wiki", wiki_ca), ("ja-tech", tech_ca), ("ja-aozora", aoz_ca)):
        rng.shuffle(items)
        for src, text, key in items:
            if have[k] >= want[k] * 1.15:
                break
            s = doc_segment(env, text, min(a.seg_len, 1024 if k == "ja-wiki" else a.seg_len), src, k, key, rng)
            if len(s.ids) < 200:
                continue
            if accept(s, cap=1):
                pools[k].append(s)
                have[k] += len(s.ids)

    sud_map = {"sud-agentic": {"agentic"}, "sud-code": {"code", "code_debug"}, "sud-en": SUD_EN,
               "sud-multi": {d for d in ("multilingual_" + x for x in
                                         "ar de el es fa fr he hi id it ko ms ne nl pl pt ru sv th tr uk vi zh".split())}}
    for k, doms in sud_map.items():
        items = sud_items(doms)
        rng.shuffle(items)
        if k == "sud-multi":  # zh first: the V4.1 zh-switch path matters most (SPEC §2.1)
            items.sort(key=lambda c: c["domain"] != "multilingual_zh")
        for c in items:
            if have[k] >= want[k] * 1.15:
                break
            s = conv_segment(env, c["messages"], a.seg_len, c["src"], k, c["key"], c.get("tools"),
                             c.get("thinking", False), c.get("effort"))
            if s and accept(s, cap=1):
                pools[k].append(s)
                have[k] += len(s.ids)
    print(f"[pool] tokens {dict(have)}", flush=True)
    print(f"[pool] rejected {dict(sorted(rej.items()))}", flush=True)

    # =========================================================================================== pack, nested
    trunc = {"ja-wiki", "ja-tech", "ja-aozora", "ja-chat", "sud-en", "sud-multi", "sud-code", "sud-agentic"}
    rows_small, rows_large = [], []
    deficit = Counter()
    for k, (lang, ratio) in MIX.items():
        pool = pools[k]
        rs = pack(pool, ratio * a.small, a.seq_len, k in trunc)
        got_s = sum(len(s.ids) for r in rs for s in r)
        rl = pack(pool, ratio * a.large - got_s, a.seq_len, k in trunc)
        for r in rs:
            rows_small.append(make_row(r, lang, k))
        rows_large_k = [make_row(r, lang, k) for r in rs + rl]
        rows_large += rows_large_k
        got_l = sum(len(r["ids"]) for r in rows_large_k)
        deficit[k] = int(ratio * a.large - got_l)
    # interleave categories so that every file prefix is mixed
    def interleave(rows):
        by = defaultdict(list)
        for r in rows:
            by[r["sub"]].append(r)
        keyed = []
        for k, rr in by.items():
            for i, r in enumerate(rr):
                keyed.append(((i + 0.5) / len(rr), list(MIX).index(k), r))
        keyed.sort(key=lambda x: (x[0], x[1]))
        return [r for _, _, r in keyed]

    rows_small = interleave(rows_small)
    rows_large = interleave(rows_large)

    # sanity: calib64k is a subset of calib256k; nothing overlaps jakl / teacher windows
    big_ids = {tuple(r["ids"]) for r in rows_large}
    assert all(tuple(r["ids"]) in big_ids for r in rows_small)
    for r in rows_large + jakl_rows:
        assert len(r["ids"]) <= a.seq_len, len(r["ids"])
        assert r["ids"][0] == BOS
        assert 0 < r["score_from"] < len(r["ids"])
        for x, y in r["spans"]:
            assert 0 < x < y <= len(r["ids"])

    def summary(rows):
        tok = Counter()
        n = Counter()
        sub = Counter()
        jach = [0, 0]
        jtok = Counter()
        for r in rows:
            jtok[r["lang"]] += r["ja_tok"] * len(r["ids"])
            tok[r["lang"]] += len(r["ids"])
            n[r["lang"]] += 1
            sub[r["sub"]] += len(r["ids"])
            st = charstats(strip_special(env.decode(r["ids"])))
            jach[0] += ja_frac(st) * st[0]
            jach[1] += st[0]
        T = sum(tok.values())
        return {"rows": len(rows), "tokens": T, "scored": sum(b - a_ for r in rows for a_, b in r["spans"]),
                "lang_tokens": dict(tok), "lang_rows": dict(n),
                "lang_ratio": {k: round(v / T, 4) for k, v in tok.items()},
                "sub_tokens": dict(sub), "ja_char_frac": round(jach[0] / max(jach[1], 1), 4),
                "ja_token_frac": round(sum(jtok.values()) / T, 4),
                "ja_token_frac_by_lang": {k: round(jtok[k] / tok[k], 4) for k in tok}}

    meta = {
        "schema": "dsv41.calib.v1",
        "script": os.path.abspath(__file__),
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "args": vars(a),
        "mix": {k: v[1] for k, v in MIX.items()}, "jakl_mix": JAKL,
        "deficit_tokens_large": {k: v for k, v in deficit.items() if v > 0},
        "excluded_teacher_windows": sorted(f"{s}@{o}" for s, o in excl_req),
        "jakl_task_hashes": sorted(jak_tasks),
        "jakl_chat_keys": sorted(chat_eval_keys),
        "sources": json.load(open(os.path.join(a.src_dir, "_sources.json")))
        if os.path.exists(os.path.join(a.src_dir, "_sources.json")) else None,
        "files": {"calib64k": summary(rows_small), "calib256k": summary(rows_large), "jakl32": summary(jakl_rows)},
    }
    # cross-set overlap (shingle) report
    def shs(rows):
        s = set()
        for r in rows:
            s |= content_shingles(r["ids"])
        return s
    S_l, S_j = shs(rows_large), shs(jakl_rows)
    meta["overlap"] = {"calib256k_vs_jakl32": round(overlap(S_l, S_j), 5),
                       "calib256k_vs_teacher": round(overlap(S_l, eval_sh), 5),
                       "jakl32_vs_teacher": round(overlap(S_j, eval_sh), 5)}
    print(json.dumps(meta["files"], ensure_ascii=False, indent=1))
    print("overlap", meta["overlap"], "deficit", meta["deficit_tokens_large"])
    if a.dry:
        return
    for name, rows in (("calib64k", rows_small), ("calib256k", rows_large), ("jakl32", jakl_rows)):
        with open(os.path.join(a.out_dir, name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out_dir, "calib_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
