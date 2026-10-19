"""教師の MoE 入出力 (入力 x と routing) を層ごとに記録する (QAD の L2 局所蒸留用)。

stream_ref.py を import して使い、stream_ref.py 自体は変更しない。
build() の後で MoE.forward と Gate.forward をモンキーパッチで包み、各層 (0..n_layers-1) の
  - MoE 入力 x (ffn_norm の後、[n, dim] bf16)
  - gate の出力 (indices [n, topk], weights [n, topk] fp32。route_scale 込み、expert に掛かる値そのもの)
  - トークンの言語タグ (系列ごとの domain と、トークン単位の仮名 / 漢字 / 日本語文脈の判定)
を保存する。教師の expert 出力は保存しない (学習時に同じ x から MXFP4 教師でその場で計算する。SPEC §7 R-8)。

サブコマンド
  plan  : 入力 jsonl をトークナイザだけで走査し、間引き率・保存量・ディスクを見積もる (GPU 不要)
  run   : 教師 (原重み、または --exl3-pack で学生 stream) で prefill して記録する。中断再開できる
  pack  : 層ごとの生ファイルを layer_LL.safetensors にまとめる (任意)
  info  : 記録の状態を表示する

保存形式 (dsv41.moe-capture.v1) は README_CAPTURE 相当の説明を本ファイル末尾の FORMAT に書いた。

画像付き系列 (vl_rows.py の行形式、"images" を持つ行) を入力に含めると、
  - run は Transformer.forward に images / token_types を渡す (画像区間は bias_vl で routing、Engram は DEAD)
  - 行に token_types を field "tt" (int8、TEXT = -1、0..3 = START/IMAGE/NEW_LINE/END) として保存する。
    flags の 8 bit はすべて使用済み (F_DUP = 128) なので画像ビットは flags に入れず、既存の F_* の値は変えない
  - 画像トークンの保持率を --img-boost 倍にし (日本語の --ja-boost と同じ扱い)、統計に bucket tok_img を足す
画像の無い入力では保存形式・指紋・間引きは従来と同じ (row_bytes 10290、field "tt" なし)。
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time

import numpy as np

SCHEMA = "dsv41.moe-capture.v1"

# --------------------------------------------------------------------------- 行の形式
# (名前, numpy dtype, 1 行の形 (dim, topk から), safetensors dtype)
FIELDS = (
    ("x", np.int16, lambda d, k: (d,), "BF16"),   # bf16 のビット列をそのまま int16 で持つ
    ("idx", np.int16, lambda d, k: (k,), "I16"),  # 専門家番号 (384 < 2^15)
    ("w", np.float32, lambda d, k: (k,), "F32"),  # routing 重み (Gate.forward の返り値そのもの)
    ("seq", np.int32, lambda d, k: (), "I32"),    # 系列番号 (meta の seqs[gid])
    ("pos", np.int32, lambda d, k: (), "I32"),    # 系列内の位置
    ("flags", np.uint8, lambda d, k: (), "U8"),   # 下の F_* のビット
    ("dom", np.uint8, lambda d, k: (), "U8"),     # 系列の domain 番号 (meta の domains)
    ("kp", np.float32, lambda d, k: (), "F32"),   # 間引きの保持確率 (逆数が重要度重み)
)
# 画像付きの記録だけが持つ field (meta["vl"] が真のとき FIELDS の後ろに付く)。token_types そのもの (TEXT = -1)
VL_FIELD = ("tt", np.int8, lambda d, k: (), "I8")
F_KANA, F_HAN, F_JA, F_ZH, F_SCORED, F_SPECIAL, F_FLOOR, F_DUP = 1, 2, 4, 8, 16, 32, 64, 128
FLAG_NAMES = {F_KANA: "kana", F_HAN: "han", F_JA: "ja", F_ZH: "zh", F_SCORED: "scored",
              F_SPECIAL: "special", F_FLOOR: "floor", F_DUP: "dup"}

PRESETS = {  # 1 層あたりの行数の目標と、専門家ごとの最低行数 (冷たい専門家の補充)
    "64k": (65536, 128),
    "128k": (131072, 256),
    "192k": (196608, 384),
    "256k": (262144, 512),
}


def fields_for(vl):
    """記録の field 一覧。vl は bool か meta dict。"""
    if isinstance(vl, dict):
        vl = bool(vl.get("vl"))
    return FIELDS + (VL_FIELD,) if vl else FIELDS


def _vl():
    """vl_rows.py (画像付きの行形式の補助) を遅延 import する。画像の無い入力では読まない。"""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vl_rows

    return vl_rows


def row_bytes(dim, topk, vl=False):
    return sum(np.dtype(dt).itemsize * int(np.prod(shp(dim, topk))) for _, dt, shp, _ in fields_for(vl))


# --------------------------------------------------------------------------- 言語タグ
def _is_kana(o):
    return 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F


def _is_han(o):
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF


def char_counts(text):
    """(仮名, 漢字, 空白以外の文字数)。tools/r3ja/r3ja_join.py の lang() と同じ数え方。"""
    k = c = n = 0
    for ch in text:
        if ch.isspace() or ch == "�":
            continue
        n += 1
        o = ord(ch)
        if _is_kana(o):
            k += 1
        elif _is_han(o):
            c += 1
    return k, c, n


def tag_tokens(ids, decode, special_ids, score_from=0, window=16, ja_ratio=0.25, ja_min_kana=3,
               max_run=8):
    """トークン単位のフラグ (uint8) を返す。

    - 各トークンの文字列は decode([id]) で取る。UTF-8 が途中で切れたトークン (U+FFFD) は、
      切れ目が閉じるまで (最大 max_run 個) まとめて decode し、その文字を run の全トークンに付ける
      (文字数の集計は run の最後のトークンに 1 回だけ入れる)。
    - F_KANA / F_HAN: そのトークン自身の文字列に仮名 / 漢字がある。
    - F_JA: 前後 window トークンの窓で 仮名 >= ja_min_kana かつ
      ((仮名+漢字)/文字数 >= ja_ratio、または自分が仮名 / 漢字を持つ)
      (SPEC §2.1 の「CJK と仮名が 25% 以上の窓を日本語が主とする」をトークン単位にしたもの)。
    - F_ZH: 窓に仮名が無く 漢字/文字数 >= ja_ratio。
    - F_SCORED: pos >= score_from (応答側)。F_SPECIAL: 特殊トークン (文字数に数えない)。
    """
    n = len(ids)
    flags = np.zeros(n, np.uint8)
    kc = np.zeros((n, 3), np.int64)
    run = []

    def flush():
        if not run:
            return
        k, c, m = char_counts(decode(run_ids))
        f = (F_KANA if k else 0) | (F_HAN if c else 0)
        for j in run:
            flags[j] |= f
        kc[run[-1]] = (k, c, m)
        run.clear()
        run_ids.clear()

    run_ids = []
    for j, t in enumerate(ids):
        if t in special_ids:
            flush()
            flags[j] |= F_SPECIAL
            continue
        run.append(j)
        run_ids.append(t)
        if len(run) >= max_run or "�" not in decode(run_ids):
            flush()
    flush()
    cs = np.vstack([np.zeros((1, 3), np.int64), np.cumsum(kc, 0)])
    lo = np.clip(np.arange(n) - window, 0, n)
    hi = np.clip(np.arange(n) + window + 1, 0, n)
    win = cs[hi] - cs[lo]
    K, C, N = win[:, 0], win[:, 1], np.maximum(win[:, 2], 1)
    own_cjk = (flags & (F_KANA | F_HAN)) > 0
    # 窓の比率に加えて、自分が仮名 / 漢字を持ち窓に仮名が ja_min_kana 以上あるトークンも日本語とする
    # (英語の直後の日本語の書き出しは、英語トークンの文字数が多く窓の比率が下がるため)
    ja = (K >= ja_min_kana) & (((K + C) / N >= ja_ratio) | own_cjk)
    zh = (K == 0) & (C / N >= ja_ratio)
    flags[ja] |= F_JA
    flags[zh] |= F_ZH
    flags[np.arange(n) >= score_from] |= F_SCORED
    return flags


def prefix_hashes(ids, salt=None):
    """位置 j までの接頭辞の 64bit ハッシュ (同じ接頭辞なら因果的に x も同じ)。
    salt (np.uint64 [n] か None): 画像区間の位置に画像の内容から決まる値を混ぜる (ids が全部 129264 なので、
    混ぜないと違う画像が同じ接頭辞に見える)。テキストの位置 (salt 0) と salt=None は従来と同じ値。"""
    out = np.empty(len(ids), np.uint64)
    h = 0x9E3779B97F4A7C15
    m = (1 << 64) - 1
    for j, t in enumerate(ids):
        h = ((h ^ (int(t) + 1)) * 0x100000001B3) & m
        if salt is not None and salt[j]:
            h = ((h ^ int(salt[j])) * 0x100000001B3) & m
        h ^= h >> 29
        out[j] = h
    return out


def uniform01(seed, gid, layer, n):
    """(seed, 系列, 層, 位置) から決まる [0,1) の一様乱数 (splitmix64)。再開しても同じ値になる。"""
    with np.errstate(over="ignore"):
        z = (np.uint64(seed & 0xFFFFFFFF) * np.uint64(0x9E3779B97F4A7C15)
             + np.uint64(gid) * np.uint64(0xD1B54A32D192ED03)
             + np.uint64(layer) * np.uint64(0x8CB92BA72F3D8DD7)
             + np.arange(n, dtype=np.uint64))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def sha256_file(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- 計画 (CPU のみ)
def parse_inputs(specs):
    out = []
    for s in specs:
        path, _, dom = s.partition("::")
        out.append((path, dom or None))
    return out


def infer_domain(rec, override, flags):
    if rec.get("domain"):
        return str(rec["domain"])
    if override:
        return override
    if rec.get("lang"):  # tools/build_calib.py の区分 (ja / agent / code / en / multi)
        return str(rec["lang"])
    kr = rec.get("kana_ratio")
    if kr is not None and kr >= 0.15:
        return "ja"
    if len(flags) and ((flags & F_JA) > 0).mean() >= 0.25:
        return "ja"
    if rec.get("tools"):
        return "agent"
    return "en"


def make_plan(a, tok=None, decode=None, special_ids=None):
    """入力を読み、タグ付け・重複接頭辞の除外・間引き率を決める。tok か (decode, special_ids) が要る。"""
    if decode is None:
        special_ids = set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items() if t.special}
        decode = lambda ids: tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)  # noqa: E731
    inputs = parse_inputs(a.input)
    img_id = int(getattr(a, "image_token_id", 129264))
    excl = set()
    for ex in a.exclude or []:
        for l in open(ex):
            r = json.loads(l)
            if "vl_ids" in r:  # tools/vl/build_sets.py の行
                r = _vl().normalize_row(r)
            ids = r.get("ids") or r.get("tokens")
            excl.add(hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest())
            if r.get("images"):
                excl.add(_vl().row_digest(r))
            if "src" in r and "off" in r:
                excl.add(f"{r['src']}#{r['off']}")
    seqs, n_excl, n_trunc, n_img_skip = [], 0, 0, 0
    for fi, (path, dom) in enumerate(inputs):
        for li, l in enumerate(open(path)):
            if not l.strip():
                continue
            r = json.loads(l)
            if "vl_ids" in r:  # tools/vl/build_sets.py の行 (vl_ids / token_types / image_spans)
                r = _vl().normalize_row(r)
            ids = r.get("ids") if "ids" in r else r.get("tokens")
            vl_row = bool(r.get("images"))
            hid = _vl().row_digest(r) if vl_row else hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()
            if hid in excl or ("src" in r and f"{r.get('src')}#{r.get('off')}" in excl):
                n_excl += 1
                continue
            if vl_row and _vl().image_end(r) > a.max_seq_len:
                n_img_skip += 1  # 画像区間を切ると forward できない
                continue
            if len(ids) > a.max_seq_len:
                ids = ids[: a.max_seq_len]
                n_trunc += 1
            s_ = {"file": fi, "line": li, "src": r.get("src"), "off": r.get("off"), "id": r.get("id"),
                  "ids": [int(t) for t in ids], "score_from": int(r.get("score_from", 0)),
                  "_rec": r, "_dom": dom}
            if vl_row:
                rr = dict(r, ids=s_["ids"])
                s_["tt"] = _vl().row_token_types(rr, img_id)
                s_["images"] = r["images"]
                s_["base_dir"] = os.path.dirname(os.path.abspath(path))
                s_["salt"] = _vl().image_salt(rr)
            elif any(int(t) == img_id for t in s_["ids"]):
                raise SystemExit(f"{path}:{li + 1}: image_token_id {img_id} があるのに images が無い")
            seqs.append(s_)
    if not a.no_shuffle:
        random.Random(a.seed).shuffle(seqs)
    if a.limit_seqs:
        seqs = seqs[: a.limit_seqs]
    seen = set()
    vl = any("tt" in s for s in seqs)
    tot = {"tokens": 0, "dup": 0, "ja": 0, "kana": 0, "zh": 0, "scored": 0}
    if vl:
        tot["img"] = 0
        special_ids = set(special_ids) | {img_id}
    for g, s in enumerate(seqs):
        s["gid"] = g
        fl = tag_tokens(s["ids"], decode, special_ids, s["score_from"], a.window, a.ja_ratio, a.ja_min_kana)
        if "tt" in s:  # 画像の位置は言語のビットを持たない (F_SPECIAL は残す)
            fl[s["tt"] >= 0] &= np.uint8(~(F_KANA | F_HAN | F_JA | F_ZH) & 0xFF)
        if not a.keep_dup_prefix:
            ph = prefix_hashes(s["ids"], s.get("salt"))
            dup = np.fromiter((int(h) in seen for h in ph), bool, len(ph))
            seen.update(int(h) for h in ph)
            fl[dup] |= F_DUP
        s["flags"] = fl
        s["domain"] = infer_domain(s.pop("_rec"), s.pop("_dom"), fl)
        tot["tokens"] += len(fl)
        tot["dup"] += int((fl & F_DUP).astype(bool).sum())
        live = (fl & F_DUP) == 0
        tot["ja"] += int(((fl & F_JA) > 0)[live].sum())
        tot["kana"] += int(((fl & F_KANA) > 0)[live].sum())
        tot["zh"] += int(((fl & F_ZH) > 0)[live].sum())
        tot["scored"] += int(((fl & F_SCORED) > 0)[live].sum())
        if vl:
            tot["img"] += int((s["tt"] >= 0)[live].sum()) if "tt" in s else 0
    domains = sorted({s["domain"] for s in seqs})
    for s in seqs:
        s["dom"] = domains.index(s["domain"])
    # 間引き率: target = p * N_other + min(1, p * boost) * N_ja を解く (1 層あたり、全 part 合計)
    target, floor = (PRESETS[a.preset] if a.preset != "all" else (None, 0))
    if a.rows_per_layer:
        target = a.rows_per_layer
    if a.expert_floor is not None:
        floor = a.expert_floor
    n_live = tot["tokens"] - tot["dup"]
    n_img = tot.get("img", 0)
    n_ja, n_oth = tot["ja"], n_live - tot["ja"] - n_img
    boost = a.ja_boost
    img_boost = float(getattr(a, "img_boost", 1.0))

    def expect(p):
        return p * n_oth + min(1.0, p * boost) * n_ja + min(1.0, p * img_boost) * n_img

    if target is None or expect(1.0) <= target:
        p = 1.0
    else:
        lo_, hi_ = 0.0, 1.0
        for _ in range(60):
            mid = (lo_ + hi_) / 2
            lo_, hi_ = (mid, hi_) if expect(mid) < target else (lo_, mid)
        p = hi_
    p_ja = min(1.0, p * boost)
    p_img = min(1.0, p * img_boost)
    for s in seqs:
        fl = s["flags"]
        kp = np.where((fl & F_JA) > 0, p_ja, p).astype(np.float32)
        if "tt" in s:
            kp[s["tt"] >= 0] = p_img
        kp[(fl & F_DUP) > 0] = 0.0
        s["kp"] = kp
    part_i, part_n = (int(v) for v in a.part.split("/"))
    exp_rows = expect(p)
    cap = a.max_rows_per_layer or (int(a.cap_factor * max(exp_rows, 1) / part_n) + 1 + a.n_experts * floor // a.topk)
    plan = {
        "seqs": [s for s in seqs if s["gid"] % part_n == part_i],
        "all_seqs": len(seqs), "excluded": n_excl, "truncated": n_trunc,
        "tokens": tot, "domains": domains,
        "sampling": {"preset": a.preset, "target_rows_per_layer": target, "p_base": p, "ja_boost": boost,
                     "p_ja": p_ja, "expected_rows_per_layer": exp_rows, "expert_floor": floor,
                     "max_rows_per_layer_this_part": cap, "seed": a.seed, "shuffle": not a.no_shuffle,
                     "dedup_prefix": not a.keep_dup_prefix},
        "tagger": {"window": a.window, "ja_ratio": a.ja_ratio, "ja_min_kana": a.ja_min_kana},
        "part": a.part,
    }
    if vl:
        plan["vl"] = True
        plan["image_token_id"] = img_id
        plan["skipped_image_overflow"] = n_img_skip
        plan["sampling"].update({"img_boost": img_boost, "p_img": p_img})
    elif n_img_skip:
        plan["skipped_image_overflow"] = n_img_skip
    return plan


def plan_summary(plan, dim, topk, n_layers, out_dir=None):
    vl = bool(plan.get("vl"))
    rb = row_bytes(dim, topk, vl)
    smp = plan["sampling"]
    part_n = int(plan["part"].split("/")[1])
    exp_gb = smp["expected_rows_per_layer"] / part_n * n_layers * rb / 1e9
    cap_gb = smp["max_rows_per_layer_this_part"] * n_layers * rb / 1e9
    by_dom = {}
    for s in plan["seqs"]:
        d = by_dom.setdefault(s["domain"], {"seqs": 0, "tokens": 0, "ja": 0, "dup": 0})
        d["seqs"] += 1
        d["tokens"] += len(s["ids"])
        d["ja"] += int(((s["flags"] & F_JA) > 0).sum())
        d["dup"] += int(((s["flags"] & F_DUP) > 0).sum())
        if vl:
            d["img"] = d.get("img", 0) + (int((s["tt"] >= 0).sum()) if "tt" in s else 0)
    out = {"row_bytes": rb, "all_seqs": plan["all_seqs"], "this_part_seqs": len(plan["seqs"]),
           "excluded": plan["excluded"], "truncated": plan["truncated"], "tokens": plan["tokens"],
           "by_domain_this_part": by_dom, "sampling": smp,
           "expected_GB_this_part": round(exp_gb, 2), "cap_GB_this_part": round(cap_gb, 2)}
    if out_dir:
        d = os.path.abspath(out_dir)
        while not os.path.isdir(d):
            d = os.path.dirname(d)
        out["disk_free_GB"] = round(shutil.disk_usage(d).free / 1e9, 1)
    return out


# --------------------------------------------------------------------------- 状態と書き出し
def _fsync_dir(d):
    try:
        fd = os.open(d, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    except OSError:
        pass


class CaptureStore:
    """out_dir の生ファイル群と state.npz (コミット済み行数・統計) を扱う。"""

    def __init__(self, out_dir, meta, fsync=True):
        self.dir = out_dir
        self.meta = meta
        self.fsync = fsync
        self.L, self.E, self.dim, self.k = meta["n_layers"], meta["n_experts"], meta["dim"], meta["topk"]
        self.fields = fields_for(meta)
        self.buckets = ["all", "tok_ja"] + (["tok_img"] if meta.get("vl") else []) + [f"dom_{d}" for d in meta["domains"]]
        os.makedirs(out_dir, exist_ok=True)
        mp = os.path.join(out_dir, "meta.json")
        if os.path.exists(mp):
            old = json.load(open(mp))
            if old.get("packed"):
                raise SystemExit(f"{out_dir} は pack 済みで追記できない")
            if old["fingerprint"] != meta["fingerprint"]:
                diff = {k: (old["fingerprint"].get(k), meta["fingerprint"].get(k))
                        for k in set(old["fingerprint"]) | set(meta["fingerprint"])
                        if old["fingerprint"].get(k) != meta["fingerprint"].get(k)}
                raise SystemExit(f"{out_dir} の既存記録と設定が違う (再開不可): {diff}")
        else:
            tmp = mp + ".tmp"
            json.dump(meta, open(tmp, "w"), ensure_ascii=False, indent=1)
            os.replace(tmp, mp)
        for l in range(self.L):
            os.makedirs(self.ldir(l), exist_ok=True)
        self.load_state()
        self.repair()

    def ldir(self, l):
        return os.path.join(self.dir, f"layer_{l:02d}")

    def fpath(self, l, name):
        return os.path.join(self.ldir(l), f"{name}.bin")

    def load_state(self):
        sp = os.path.join(self.dir, "state.npz")
        if os.path.exists(sp):
            z = np.load(sp, allow_pickle=False)
            self.info = json.loads(bytes(z["json"]).decode())
            self.rows = z["rows"].astype(np.int64)
            self.kept = z["kept"].astype(np.int64)
            self.stats = {k: z[k].astype(np.float64) for k in z.files if k.startswith(("cnt.", "wsum.", "xsq."))}
        else:
            self.info = {"done": [], "seqs": {}, "t_total": 0.0}
            self.rows = np.zeros(self.L, np.int64)
            self.kept = np.zeros((self.L, self.E), np.int64)
            self.stats = {}
        for b in self.buckets:
            for s in ("cnt", "wsum", "xsq"):
                self.stats.setdefault(f"{s}.{b}", np.zeros((self.L, self.E), np.float64))

    def repair(self):
        """コミットより後ろに書かれた行 (中断で残った部分) を切り捨てる。足りなければ壊れている。"""
        n_cut = 0
        for l in range(self.L):
            for name, dt, shp, _ in self.fields:
                p = self.fpath(l, name)
                want = int(self.rows[l]) * np.dtype(dt).itemsize * int(np.prod(shp(self.dim, self.k)))
                have = os.path.getsize(p) if os.path.exists(p) else 0
                if have < want:
                    raise SystemExit(f"{p}: {have} B < コミット済み {want} B (記録が壊れている)")
                if have > want or not os.path.exists(p):
                    with open(p, "ab") as f:
                        f.truncate(want)
                    n_cut += have > want
        if n_cut:
            print(f"[resume] 未コミットの書きかけを {n_cut} ファイルで切り捨てた", flush=True)

    def commit(self, bufs, rows, kept, stats, seq_info):
        """bufs[l] = {name: ndarray}。データを追記・同期してから state.npz を原子的に置き換える。"""
        for l in range(self.L):
            b = bufs.get(l)
            if not b:
                continue
            for name, _, _, _ in self.fields:
                with open(self.fpath(l, name), "ab") as f:
                    f.write(np.ascontiguousarray(b[name]).tobytes())
                    if self.fsync:
                        f.flush()
                        os.fsync(f.fileno())
        info = json.loads(json.dumps(self.info))
        info["done"].append(seq_info["gid"])
        info["seqs"][str(seq_info["gid"])] = seq_info
        info["t_total"] = info.get("t_total", 0.0) + seq_info.get("seconds", 0.0)
        sp = os.path.join(self.dir, "state.npz")
        tmp = os.path.join(self.dir, "state.tmp.npz")
        arrs = {"json": np.frombuffer(json.dumps(info, ensure_ascii=False).encode(), np.uint8),
                "rows": rows, "kept": kept}
        arrs.update(stats)
        with open(tmp, "wb") as f:
            np.savez(f, **arrs)
            if self.fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, sp)
        if self.fsync:
            _fsync_dir(self.dir)
        self.info, self.rows, self.kept, self.stats = info, rows, kept, stats


class Capturer:
    """MoE.forward / Gate.forward を包み、1 系列分の記録を層ごとにバッファする。"""

    def __init__(self, store, plan):
        self.st = store
        self.plan = plan
        smp = plan["sampling"]
        self.floor = int(smp["expert_floor"])
        self.cap = int(smp["max_rows_per_layer_this_part"])
        self.seed = int(smp["seed"])
        self.cur = None    # 系列の作業状態
        self.slot = None   # MoE → Gate の受け渡し
        self.n_layers = store.L

    # ---- パッチ
    def install(self, M):
        cap = self
        orig_moe, orig_gate = M.MoE.forward, M.Gate.forward

        def gate_forward(gself, x, image_mask=None):
            w, i = orig_gate(gself, x, image_mask)
            if cap.slot is not None and "w" not in cap.slot:
                cap.slot["w"], cap.slot["i"] = w, i
            return w, i

        def moe_forward(mself, x, image_mask=None):
            if cap.cur is None or mself.layer_id >= cap.n_layers:
                return orig_moe(mself, x, image_mask)
            cap.slot = {}
            try:
                y = orig_moe(mself, x, image_mask)
                slot = cap.slot
            finally:
                cap.slot = None
            if "w" not in slot:
                raise RuntimeError(f"layer {mself.layer_id}: gate が呼ばれなかった")
            cap.on_layer(mself.layer_id, x.reshape(-1, mself.dim), slot["w"], slot["i"])
            return y

        gate_forward._capture_orig = orig_gate
        moe_forward._capture_orig = orig_moe
        M.Gate.forward = gate_forward
        M.MoE.forward = moe_forward
        self.M = M

    def uninstall(self):
        self.M.Gate.forward = self.M.Gate.forward._capture_orig
        self.M.MoE.forward = self.M.MoE.forward._capture_orig

    # ---- 系列
    def begin(self, s):
        st = self.st
        self.cur = {"s": s, "bufs": {}, "seen": set(), "rows": st.rows.copy(), "kept": st.kept.copy(),
                    "stats": {k: v.copy() for k, v in st.stats.items()}, "t0": time.time()}

    def abort(self):
        self.cur = None

    def on_layer(self, l, x2d, w, i):
        import torch

        c = self.cur
        s = c["s"]
        n = len(s["ids"])
        if x2d.shape[0] != n:
            raise RuntimeError(f"layer {l}: x の行数 {x2d.shape[0]} != 系列長 {n}")
        if l in c["seen"]:
            raise RuntimeError(f"layer {l} が 1 系列で 2 回呼ばれた")
        c["seen"].add(l)
        idx = i.to(torch.int64).cpu().numpy()
        wv = w.float().cpu().numpy()
        xsq = x2d.float().square().sum(-1).cpu().numpy().astype(np.float64)
        fl = s["flags"]
        live = (fl & F_DUP) == 0
        E = self.st.E
        # 統計 (重複接頭辞を除く全トークン、間引きの前)
        buckets = [("all", live), ("tok_ja", live & ((fl & F_JA) > 0)), (f"dom_{s['domain']}", live)]
        if "tok_img" in self.st.buckets:
            buckets.append(("tok_img", live & (s["tt"] >= 0) if "tt" in s else np.zeros(n, bool)))
        for b, m in buckets:
            if not m.any():
                continue
            ii = idx[m].ravel()
            ww = wv[m].ravel().astype(np.float64)
            c["stats"][f"cnt.{b}"][l] += np.bincount(ii, minlength=E)
            c["stats"][f"wsum.{b}"][l] += np.bincount(ii, weights=ww, minlength=E)
            c["stats"][f"xsq.{b}"][l] += np.bincount(ii, weights=ww * np.repeat(xsq[m], idx.shape[1]), minlength=E)
        # 間引き
        u = uniform01(self.seed, s["gid"], l, n)
        keep = u < s["kp"]
        floor_mask = np.zeros(n, bool)
        if self.floor > 0:
            deficit = self.floor - c["kept"][l] - np.bincount(idx[keep].ravel(), minlength=E)
            if (deficit > 0).any():
                cand = np.flatnonzero(~keep & live)
                for t in cand[np.argsort(u[cand], kind="stable")]:
                    es = idx[t]
                    if (deficit[es] > 0).any():
                        floor_mask[t] = True
                        deficit[es] -= 1
                        if deficit.max() <= 0:
                            break
        keep |= floor_mask
        sel = np.flatnonzero(keep)
        room = self.cap - int(c["rows"][l])
        if len(sel) > room:
            sel = np.sort(sel[np.argsort(u[sel], kind="stable")[: max(room, 0)]])
        if len(sel):
            st = torch.from_numpy(sel).to(x2d.device)
            xs = x2d.index_select(0, st).contiguous()
            buf = {
                "x": xs.view(torch.int16).cpu().numpy(),
                "idx": idx[sel].astype(np.int16),
                "w": wv[sel].astype(np.float32),
                "seq": np.full(len(sel), s["gid"], np.int32),
                "pos": sel.astype(np.int32),
                "flags": (fl[sel] | np.where(floor_mask[sel], F_FLOOR, 0)).astype(np.uint8),
                "dom": np.full(len(sel), s["dom"], np.uint8),
                "kp": s["kp"][sel].astype(np.float32),
            }
            if "tok_img" in self.st.buckets:
                buf["tt"] = (s["tt"][sel] if "tt" in s else np.full(len(sel), -1)).astype(np.int8)
            c["bufs"][l] = buf
            c["rows"][l] += len(sel)
            c["kept"][l] += np.bincount(idx[sel].ravel(), minlength=E)

    def commit(self):
        c = self.cur
        s = c["s"]
        if len(c["seen"]) != self.n_layers:
            raise RuntimeError(f"seq {s['gid']}: {len(c['seen'])}/{self.n_layers} 層しか通らなかった")
        fl = s["flags"]
        info = {"gid": s["gid"], "file": s["file"], "line": s["line"], "src": s["src"], "off": s["off"],
                "id": s["id"], "n": len(s["ids"]), "score_from": s["score_from"], "domain": s["domain"],
                "n_ja": int(((fl & F_JA) > 0).sum()), "n_dup": int(((fl & F_DUP) > 0).sum()),
                "kept": [int(len(c["bufs"].get(l, {}).get("pos", ()))) for l in range(self.n_layers)],
                "seconds": round(time.time() - c["t0"], 2)}
        if "tt" in s:
            info["n_img"] = int((s["tt"] >= 0).sum())
        self.st.commit(c["bufs"], c["rows"], c["kept"], c["stats"], info)
        self.cur = None
        return info


def run_capture(store, plan, cap, forward_ids, log=print, stop_after=None):
    """plan の未処理系列を順に流す。forward_ids(ids) が 1 系列の prefill を行う。"""
    done = set(store.info["done"])
    todo = [s for s in plan["seqs"] if s["gid"] not in done]
    log(f"[capture] {len(done)} 系列は記録済み、残り {len(todo)} 系列", flush=True)
    t_start = time.time()
    n_tok = 0
    for k, s in enumerate(todo):
        if stop_after is not None and k >= stop_after:
            break
        cap.begin(s)
        try:
            forward_ids(s["ids"])
            info = cap.commit()
        except BaseException:
            cap.abort()
            raise
        n_tok += info["n"]
        el = time.time() - t_start
        rate = n_tok / max(el, 1e-9)
        left = sum(len(q["ids"]) for q in todo[k + 1:]) / max(rate, 1e-9)
        log(f"[capture] seq {s['gid']} ({s['domain']}, {info['n']} tok, ja {info['n_ja']}, dup {info['n_dup']}) "
            f"kept/layer mean {np.mean(info['kept']):.0f} | rows/layer mean {store.rows.mean():.0f} "
            f"| {info['seconds']:.1f}s, {rate:.0f} tok/s, 残り約 {left / 60:.0f} 分", flush=True)
    return store


# --------------------------------------------------------------------------- 読み込み (学習側)
def _st_header(path):
    with open(path, "rb") as f:
        hl = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(hl)), 8 + hl


def open_layer(d, layer):
    """1 つの記録ディレクトリの層を numpy memmap の dict で返す (生ファイルでも pack 後でも)。"""
    meta = json.load(open(os.path.join(d, "meta.json")))
    dim, k = meta["dim"], meta["topk"]
    fields = fields_for(meta)
    sp = os.path.join(d, f"layer_{layer:02d}.safetensors")
    out = {}
    if os.path.exists(sp):
        hdr, base = _st_header(sp)
        for name, dt, shp, _ in fields:
            v = hdr[name]
            a, b = v["data_offsets"]
            n = v["shape"][0]
            out[name] = (np.memmap(sp, dt, "r", base + a, (n,) + shp(dim, k)) if n
                         else np.zeros((0,) + shp(dim, k), dt))
        return out
    z = np.load(os.path.join(d, "state.npz"), allow_pickle=False)
    n = int(z["rows"][layer])
    for name, dt, shp, _ in fields:
        p = os.path.join(d, f"layer_{layer:02d}", f"{name}.bin")
        out[name] = np.memmap(p, dt, "r", 0, (n,) + shp(dim, k)) if n else np.zeros((0,) + shp(dim, k), dt)
    return out


def load_layer(dirs, layer, as_torch=True):
    """複数の記録 (part) を連結して返す。as_torch なら x は bf16 の torch.Tensor [n, dim]。
    seq は part ごとの gid のまま (gid は全 part 共通の通し番号なので衝突しない)。"""
    if isinstance(dirs, str):
        dirs = [dirs]
    parts = [open_layer(d, layer) for d in dirs]
    has_tt = any(VL_FIELD[0] in p for p in parts)
    fields = FIELDS + ((VL_FIELD,) if has_tt else ())
    for p in parts:  # 画像なしの part と混ぜるときは tt = TEXT (-1) で埋める
        if has_tt and VL_FIELD[0] not in p:
            p[VL_FIELD[0]] = np.full(len(p["seq"]), -1, np.int8)
    out = {name: np.concatenate([np.asarray(p[name]) for p in parts]) for name, _, _, _ in fields}
    if as_torch:
        import torch

        t = {k: torch.from_numpy(v) for k, v in out.items()}
        t["x"] = t["x"].view(torch.bfloat16)
        return t
    return out


def load_stats(d):
    z = np.load(os.path.join(d, "state.npz"), allow_pickle=False)
    return {k: z[k] for k in z.files if k != "json"}, json.loads(bytes(z["json"]).decode())


# --------------------------------------------------------------------------- pack
def pack(d, rm_raw=False):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "state.npz"), allow_pickle=False)
    rows = z["rows"]
    dim, k = meta["dim"], meta["topk"]
    fields = fields_for(meta)
    for l in range(meta["n_layers"]):
        n = int(rows[l])
        hdr, off = {}, 0
        for name, dt, shp, sdt in fields:
            nb = n * np.dtype(dt).itemsize * int(np.prod(shp(dim, k)))
            hdr[name] = {"dtype": sdt, "shape": [n, *shp(dim, k)], "data_offsets": [off, off + nb]}
            off += nb
        hdr["__metadata__"] = {"schema": SCHEMA, "layer": str(l), "rows": str(n),
                               "model": meta.get("model", ""), "stream": meta.get("stream", "")}
        hb = json.dumps(hdr).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        out = os.path.join(d, f"layer_{l:02d}.safetensors")
        tmp = out + ".tmp"
        with open(tmp, "wb") as f:
            f.write(len(hb).to_bytes(8, "little"))
            f.write(hb)
            for name, dt, shp, _ in fields:
                nb = hdr[name]["data_offsets"][1] - hdr[name]["data_offsets"][0]
                with open(os.path.join(d, f"layer_{l:02d}", f"{name}.bin"), "rb") as g:
                    left = nb
                    while left:
                        b = g.read(min(left, 1 << 26))
                        if not b:
                            raise IOError(f"layer {l} {name}: 生ファイルが短い")
                        f.write(b)
                        left -= len(b)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
        if rm_raw:
            shutil.rmtree(os.path.join(d, f"layer_{l:02d}"))
        print(f"[pack] layer {l:02d}: {n} rows -> {out}", flush=True)
    meta["packed"] = True
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------- verify (CPU)
def verify_rows(rows, gate, n_check=512, seed=0):
    """保存行の x から gate を再計算し、idx / w が記録と一致するかを返す (gate は model.py の Gate)。"""
    import torch

    n = len(rows["seq"])
    if n == 0:
        return {"rows": 0}
    sel = np.sort(np.random.default_rng(seed).choice(n, min(n, n_check), replace=False))
    x = rows["x"][torch.from_numpy(sel)]
    im = None
    if "tt" in rows:  # 画像の行は bias_vl で選ぶ (model.py:820-822)
        im = torch.as_tensor(np.asarray(rows["tt"])[sel] >= 0)
    with torch.inference_mode():
        w, i = gate(x) if im is None else gate(x, im)
    ri = rows["idx"][torch.from_numpy(sel)].long()
    rw = rows["w"][torch.from_numpy(sel)]
    same_set = (torch.sort(i, -1).values == torch.sort(ri, -1).values).all(-1).float().mean().item()
    exact = (i == ri).all(-1).float().mean().item()
    dw = (w.float() - rw).abs().max().item()
    out = {"rows": int(n), "checked": int(len(sel)), "idx_set_agree": same_set, "idx_exact": exact,
           "w_maxdiff": dw, "x_finite": bool(torch.isfinite(x.float()).all().item())}
    if im is not None:
        out["checked_img"] = int(im.sum())
    return out


def load_real_gate(layer, ckpt):
    """原重みの Gate (model.py のクラスそのもの) を CPU に作る。"""
    import torch

    os.environ.setdefault("DSV41_CKPT", ckpt)
    import stream_ref as SR

    cfg = json.load(open(os.path.join(ckpt, "inference", "config.json")))
    args = SR.M.ModelArgs(**cfg)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        g = SR.M.Gate(layer, args)
    finally:
        torch.set_default_dtype(prev)
    ti = SR.TensorIndex(ckpt)
    with torch.no_grad():
        for name in ("weight", "bias", "bias_vl"):
            if getattr(g, name, None) is None or f"layers.{layer}.ffn.gate.{name}" not in ti.meta:
                continue
            src = ti.load(f"layers.{layer}.ffn.gate.{name}")
            getattr(g, name).copy_(src.to(torch.float32).to(getattr(g, name).dtype))
    for fd in ti.fds.values():
        os.close(fd)
    return g


# --------------------------------------------------------------------------- CLI
def add_common(ap):
    ap.add_argument("--input", action="append", default=[], required=True,
                    help="校正系列 jsonl (ids か tokens、任意で score_from/domain)。'path::domain' で domain を上書き")
    ap.add_argument("--exclude", action="append", default=[], help="除外する系列の jsonl (評価集合など)")
    ap.add_argument("--out", required=True, help="記録ディレクトリ (part ごとに別にする)")
    ap.add_argument("--preset", default="64k", choices=list(PRESETS) + ["all"])
    ap.add_argument("--rows-per-layer", type=int, default=0, help="preset の行数を上書き (全 part 合計)")
    ap.add_argument("--expert-floor", type=int, default=None, help="専門家ごとの最低保持行数 (part ごと)")
    ap.add_argument("--max-rows-per-layer", type=int, default=0, help="1 層の行数の上限 (この part)。0=自動")
    ap.add_argument("--cap-factor", type=float, default=1.15)
    ap.add_argument("--ja-boost", type=float, default=2.0, help="日本語文脈トークンの保持率の倍率")
    ap.add_argument("--img-boost", type=float, default=2.0,
                    help="画像トークンの保持率の倍率 (画像付きの行があるときだけ効く。kp に入るので 1/kp で自然分布に戻せる)")
    ap.add_argument("--image-token-id", type=int, default=129264)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--ja-ratio", type=float, default=0.25)
    ap.add_argument("--ja-min-kana", type=int, default=3)
    ap.add_argument("--keep-dup-prefix", action="store_true", help="系列間で同じ接頭辞の位置も残す")
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--part", default="0/1", help="i/n: gid %% n == i の系列だけ処理 (GPU ごと)")
    ap.add_argument("--limit-seqs", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--exl3-pack", default="", help="学生 stream を記録する場合の EXL3 パック (既定は教師)")
    ap.add_argument("--dim", type=int, default=5120)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--n-experts", type=int, default=384)
    ap.add_argument("--n-layers", type=int, default=40)


def build_meta(a, plan, ckpt):
    here = os.path.dirname(os.path.abspath(__file__))
    fp = {
        "inputs": [(os.path.basename(p), d, sha256_file(p)) for p, d in parse_inputs(a.input)],
        "exclude": [(os.path.basename(p), sha256_file(p)) for p in a.exclude],
        "sampling": {k: plan["sampling"][k] for k in ("preset", "target_rows_per_layer", "p_base", "ja_boost",
                                                       "expert_floor", "max_rows_per_layer_this_part", "seed",
                                                       "shuffle", "dedup_prefix")},
        "tagger": plan["tagger"], "part": a.part, "limit_seqs": a.limit_seqs, "max_seq_len": a.max_seq_len,
        "stream": "student:" + os.path.basename(a.exl3_pack.rstrip("/")) if a.exl3_pack else "teacher",
        "dim": a.dim, "topk": a.topk, "n_experts": a.n_experts, "n_layers": a.n_layers,
    }
    if plan.get("vl"):  # 画像なしの記録の指紋は従来のまま (再開できる)
        fp["sampling"]["img_boost"] = plan["sampling"]["img_boost"]
        fp["vl"] = True
    fp = json.loads(json.dumps(fp))  # tuple → list (meta.json から読んだ値と比べられるように)
    code = {}
    for f in ("capture_moe.py", "stream_ref.py") + (("vl_rows.py",) if plan.get("vl") else ()):
        p = os.path.join(here, f)
        if os.path.exists(p):
            code[f] = sha256_file(p)
    seqs = [{"gid": s["gid"], "file": s["file"], "line": s["line"], "src": s["src"], "off": s["off"],
             "id": s["id"], "n": len(s["ids"]), "domain": s["domain"]} for s in plan["seqs"]]
    return {"schema": SCHEMA, "model": os.path.basename(os.path.normpath(ckpt)), "ckpt": ckpt,
            "stream": fp["stream"], "exl3_pack": a.exl3_pack, "dim": a.dim, "topk": a.topk,
            "n_experts": a.n_experts, "n_layers": a.n_layers, "domains": plan["domains"],
            "fields": [{"name": n, "dtype": sdt, "row_shape": list(shp(a.dim, a.topk))}
                       for n, _, shp, sdt in fields_for(bool(plan.get("vl")))],
            "flags": {v: k for k, v in FLAG_NAMES.items()}, "row_bytes": row_bytes(a.dim, a.topk, bool(plan.get("vl"))),
            "sampling": plan["sampling"], "tagger": plan["tagger"], "part": a.part,
            "inputs": fp["inputs"], "fingerprint": fp, "code_sha256": code, "seqs": seqs,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")} | (
        {"vl": True, "image_token_id": plan["image_token_id"]} if plan.get("vl") else {})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan", help="保存量の見積もり (GPU 不要)")
    add_common(p_plan)
    p_run = sub.add_parser("run", help="記録する")
    add_common(p_run)
    p_run.add_argument("--expert-budget-gb", type=float, default=20)
    p_run.add_argument("--no-fsync", action="store_true")
    p_run.add_argument("--force-disk", action="store_true", help="ディスク不足の見込みでも実行する")
    p_pack = sub.add_parser("pack", help="layer_LL.safetensors にまとめる")
    p_pack.add_argument("dir")
    p_pack.add_argument("--rm-raw", action="store_true")
    p_info = sub.add_parser("info")
    p_info.add_argument("dir")
    p_ver = sub.add_parser("verify", help="保存行の x から原重みの gate を CPU で再計算して照合する")
    p_ver.add_argument("dir")
    p_ver.add_argument("--layers", default="0,1,14,20,39")
    p_ver.add_argument("--n-check", type=int, default=512)
    a = ap.parse_args(argv)

    if a.cmd == "pack":
        return pack(a.dir, a.rm_raw)
    if a.cmd == "verify":
        ckpt = os.environ.get("DSV41_CKPT", "/model")
        res = {}
        for l in (int(v) for v in a.layers.split(",")):
            res[l] = verify_rows(load_layer(a.dir, l), load_real_gate(l, ckpt), a.n_check)
            print(f"[verify] layer {l}: {res[l]}", flush=True)
        bad = [l for l, r in res.items() if r.get("rows") and (r["idx_set_agree"] < 0.995 or r["w_maxdiff"] > 1e-3
                                                                 or not r["x_finite"])]
        if bad:
            raise SystemExit(f"[verify] 不一致の層: {bad}")
        print("[verify] OK", flush=True)
        return res
    if a.cmd == "info":
        meta = json.load(open(os.path.join(a.dir, "meta.json")))
        st, info = load_stats(a.dir)
        rows = st["rows"]
        print(json.dumps({"stream": meta["stream"], "part": meta["part"], "seqs_done": len(info["done"]),
                          "seqs_planned": len(meta["seqs"]), "rows_per_layer_min_mean_max":
                          [int(rows.min()), float(rows.mean()), int(rows.max())],
                          "GB": round(int(rows.sum()) * meta["row_bytes"] / 1e9, 2),
                          "kept_per_expert_min": int(st["kept"].min()),
                          "hours": round(info.get("t_total", 0) / 3600, 2)}, ensure_ascii=False, indent=1))
        return

    ckpt = os.environ.get("DSV41_CKPT", "/model")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ckpt)
    plan = make_plan(a, tok)
    summ = plan_summary(plan, a.dim, a.topk, a.n_layers, a.out)
    print(json.dumps(summ, ensure_ascii=False, indent=1), flush=True)
    for p, _ in parse_inputs(a.input):
        if os.path.basename(p) in ("ja16.jsonl", "agent24.jsonl", "agent4.jsonl") and not a.exclude:
            print(f"[warn] {p} は評価集合 (QUALITY.md)。学習に使うと KL 比較が無効になる", flush=True)
    if a.cmd == "plan":
        return summ
    need = summ["cap_GB_this_part"] + 20
    if "disk_free_GB" in summ and summ["disk_free_GB"] < need and not a.force_disk:
        raise SystemExit(f"ディスク不足の見込み: 空き {summ['disk_free_GB']} GB < 上限 {need:.0f} GB (--force-disk で続行)")
    meta = build_meta(a, plan, ckpt)
    store = CaptureStore(a.out, meta, fsync=not a.no_fsync)
    if all(s["gid"] in set(store.info["done"]) for s in plan["seqs"]):
        print("[capture] すべて記録済み", flush=True)
        return

    import torch
    import stream_ref as SR

    model, tok2, args = SR.build({"max_batch_size": 1, "max_seq_len": a.max_seq_len, "temperature": 0.0},
                                 a.expert_budget_gb, exl3_pack=a.exl3_pack)
    assert args.dim == a.dim and args.n_activated_experts == a.topk and args.n_routed_experts == a.n_experts \
        and args.n_layers == a.n_layers, "ModelArgs と --dim/--topk/--n-experts/--n-layers が合わない"
    cap = Capturer(store, plan)
    cap.install(SR.M)

    vlm = _vl() if plan.get("vl") else None
    if vlm is not None and model.vision is None:
        raise SystemExit("画像付きの行があるが、config に vision が無い (vision_n_layers == 0)")

    def fwd(ids):
        s = cap.cur["s"]
        if vlm is None or "tt" not in s:
            model.forward(torch.tensor([ids], dtype=torch.long), 0)
            return
        imgs = vlm.image_inputs({"ids": ids, "images": s["images"]}, args, s["base_dir"])
        tt = torch.tensor(s["tt"].astype(np.int64), dtype=torch.long)[None]  # 既定デバイス (build 後は cuda)
        model.forward(torch.tensor([ids], dtype=torch.long), 0, images=[imgs], token_types=tt)

    run_capture(store, plan, cap, fwd)
    print("[experts]", SR.STORE.stats, flush=True)


FORMAT = """
保存形式 dsv41.moe-capture.v1 (out_dir ごと。GPU / part ごとに 1 ディレクトリ)
  meta.json     schema, model, stream ("teacher" か "student:<pack>"), dim, topk, n_experts, n_layers,
                domains (dom 番号 → 名前), fields, flags, row_bytes, sampling, tagger, part,
                seqs [{gid, file, line, src, off, id, n, domain}] (この part の計画), fingerprint (再開の照合)
  state.npz     json (done: 記録済み gid、seqs: gid ごとの {n, n_ja, n_dup, kept[層], seconds})、
                rows [L] (層ごとのコミット済み行数)、kept [L,E] (保持行の専門家ごとの選択回数)、
                cnt./wsum./xsq.<bucket> [L,E] float64 (間引き前・重複接頭辞除外の routing 統計。
                bucket = all, tok_ja, dom_<domain>。xsq = Σ w‖x‖²)
  layer_LL/<field>.bin   生の行ファイル (追記)。行 r は全 field で同じトークン
      x     int16 [n, dim]  (bf16 のビット列。torch では .view(torch.bfloat16))
      idx   int16 [n, topk]   w  float32 [n, topk] (route_scale 込みの routing 重み)
      seq   int32 [n] (gid)   pos int32 [n]   flags uint8 [n]   dom uint8 [n]   kp float32 [n] (保持確率)
  layer_LL.safetensors   pack 後 (同じ名前の tensor、x は BF16)。pack すると追記はできない
  flags: 1 kana, 2 han, 4 ja (日本語文脈), 8 zh, 16 scored (pos>=score_from), 32 special, 64 floor (最低行数の補充で保持)
  画像付きの記録 (meta.vl = true) だけ: field tt int8 [n] (token_types。-1 テキスト、0 START、1 IMAGE、2 NEW_LINE、
      3 END)、統計の bucket tok_img、sampling.img_boost / p_img、seqs の info に n_img。row_bytes は 10291 (dim 5120)。
      画像の位置の flags は言語のビット (kana/han/ja/zh) を持たない。画像トークンの kp は p_img
"""

if __name__ == "__main__":
    main()
