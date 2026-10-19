"""CPU tests for build_calib.py (synthetic data + validation of the written files). No GPU.

usage: .venv/bin/python test_build_calib.py [--skip-files]
"""
import json
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_calib as B  # noqa: E402

OUT = B.ROOT + "/work/calib"


def test_charstats():
    ja = "今日は良い天気です。明日も晴れるでしょう。"
    zh = "今天天气很好，明天也会是晴天。我们去公园散步吧。"
    code = "def f(x):\n    return {k: v for k, v in x.items() if v[0] == '_'}"
    en = "The quick brown fox jumps over the lazy dog."
    assert B.classify(B.charstats(ja)) == "ja"
    assert B.classify(B.charstats(zh)) == "other", B.charstats(zh)  # no kana -> not Japanese
    assert B.ja_frac(B.charstats(zh)) == 0.0
    assert B.classify(B.charstats(code)) == "code"
    assert B.classify(B.charstats(en)) == "other"
    assert B.ja_frac(B.charstats(ja)) > 0.9


def test_shingles():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 120000, 3000).tolist()
    b = rng.integers(0, 120000, 3000).tolist()
    sa = B.shingles(a)
    assert 3000 / 8 * 0.5 < len(sa) < 3000 / 8 * 1.5
    shifted = B.shingles([7, 8, 9] + a[500:])  # same content at another offset
    assert B.overlap(shifted, sa) > 0.95
    assert B.overlap(B.shingles(b), sa) < 0.01
    # content_shingles skips the system/tool prefix before <｜User｜>
    p1 = list(range(1000, 1600)) + [B.USER_ID] + a[:800]
    p2 = list(range(1000, 1600)) + [B.USER_ID] + b[:800]
    assert B.overlap(B.content_shingles(p1), B.content_shingles(p2)) < 0.01
    assert B.overlap(B.shingles(p1), B.shingles(p2)) > 0.3


class _S(B.Seg):
    def __init__(self, n, tag):
        super().__init__([B.BOS] + [5] * (n - 1), [[1, n]], tag, "t", (n, 0, 0, 0), tag)


def test_pack_nested():
    B.CJK_TOK = np.zeros(200000, bool)
    rng = random.Random(0)
    pool = [_S(rng.randint(300, 2048), f"s{i}") for i in range(200)]
    total = sum(len(s.ids) for s in pool)
    rows_s = B.pack(pool, 20000, 4096, False)
    got_s = sum(len(s.ids) for r in rows_s for s in r)
    rows_l = B.pack(pool, 80000 - got_s, 4096, False)
    got_l = sum(len(s.ids) for r in rows_l for s in r)
    for r in rows_s + rows_l:
        assert sum(len(s.ids) for s in r) <= 4096
    assert 20000 - 2048 <= got_s <= 20000 + 2048, got_s
    assert 80000 - 2048 <= got_s + got_l <= 80000 + 2048, got_s + got_l
    names = [s.src for r in rows_s + rows_l for s in r]
    assert len(names) == len(set(names))  # every segment used once
    assert sum(len(s.ids) for s in pool) == total - got_s - got_l  # consumed from the pool
    # truncation mode hits the budget exactly
    pool2 = [_S(2000, f"d{i}") for i in range(20)]
    rows = B.pack(pool2, 5000, 4096, True)
    assert sum(len(s.ids) for r in rows for s in r) == 5000
    row = B.make_row(rows[0], "ja", "x")
    assert row["ids"][0] == B.BOS and row["score_from"] == 1
    assert all(0 < a < b <= len(row["ids"]) for a, b in row["spans"])
    assert row["spans"][1][0] == 2001  # second segment offset by the first


def test_req_window():
    env = B.Env()
    with tempfile.TemporaryDirectory() as d:
        # two requests in one stream: r0 (plen 100, olen 50, starts with BOS), r1 (plen 6000, olen 3000)
        r0 = [B.BOS] + list(range(10, 109)) + list(range(200, 250))
        r1 = [B.BOS] + list(range(1000, 6999)) + list(range(7000, 10000))
        np.asarray(r0 + r1, dtype=np.uint32).tofile(os.path.join(d, "x.u32"))
        old = B.REQ
        B.REQ = d
        env._u32.clear()
        try:
            ids, sf, st, en = B.req_window(env, {"src": "x.u32", "off": 0, "plen": 100, "olen": 50}, 4096, 1024)
            assert ids == r0 and sf == 100 and ids[sf] == 200 and st == 0
            ids, sf, st, en = B.req_window(env, {"src": "x.u32", "off": 150, "plen": 6000, "olen": 3000}, 4096, 1024)
            assert len(ids) == 4096 and ids[0] == B.BOS
            assert ids[sf] == 7000 and ids[sf - 1] == 6998  # first output token / last prompt token
            assert len(ids) - sf == 1024 and en == 7024
            ids, sf, _, _ = B.req_window(env, {"src": "x.u32", "off": 150, "plen": 6000, "olen": 3000}, 2048, 1024)
            assert len(ids) == 2048 and len(ids) - sf == 1024 and ids[sf] == 7000
        finally:
            B.REQ = old
            env._u32.clear()
    return env


def test_render(env):
    msgs = [{"role": "user", "content": "日本の首都はどこですか？"},
            {"role": "assistant", "content": "東京です。"},
            {"role": "user", "content": "人口は？"},
            {"role": "assistant", "content": "約1400万人です。"}]
    ids, spans = env.render(msgs)
    assert ids[0] == B.BOS and len(spans) == 2
    texts = [env.decode(ids[a:b]) for a, b in spans]
    assert "東京です。" in texts[0] and "首都" not in texts[0], texts
    assert "1400" in texts[1] and texts[1].endswith("<｜end▁of▁sentence｜>"), texts
    # thinking mode with tools: tools rendered on the system message, assistant tool call is in a span
    tools = [{"type": "function", "function": {"name": "get_time", "description": "now",
                                                "parameters": {"type": "object", "properties": {}}}}]
    msgs2 = [{"role": "user", "content": "今何時？"},
             {"role": "assistant", "content": "", "reasoning_content": "時刻を調べる。",
              "tool_calls": [{"type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
             {"role": "tool", "content": "12:00"},
             {"role": "assistant", "content": "12時です。", "reasoning_content": "答える。"}]
    ids, spans = env.render(msgs2, tools=tools, thinking=True, effort="high")
    full = env.decode(ids)
    assert "get_time" in full.split("<｜User｜>")[0]
    span_txt = "".join(env.decode(ids[a:b]) for a, b in spans)
    assert "12時です。" in span_txt and "今何時" not in span_txt and "12:00" not in span_txt


def test_files():
    rows = {n: [json.loads(line) for line in open(f"{OUT}/{n}.jsonl")] for n in ("calib64k", "calib256k", "jakl32")}
    meta = json.load(open(f"{OUT}/calib_meta.json"))
    for n, rr in rows.items():
        for r in rr:
            assert set(r) >= {"ids", "score_from", "lang", "src", "sub", "spans"}, r.keys()
            assert r["ids"][0] == B.BOS and len(r["ids"]) <= 4096
            assert 0 < r["score_from"] < len(r["ids"])
            assert r["score_from"] == r["spans"][0][0]
            assert all(0 < a < b <= len(r["ids"]) for a, b in r["spans"])
            assert all(isinstance(x, int) and 0 <= x < 129280 for x in r["ids"])
    big = {tuple(r["ids"]) for r in rows["calib256k"]}
    assert all(tuple(r["ids"]) in big for r in rows["calib64k"]), "calib64k must be a subset of calib256k"
    assert len(rows["jakl32"]) == 32 and all(r["lang"] == "ja" for r in rows["jakl32"])
    # token budgets and the ja share
    for n, target in (("calib64k", 65536), ("calib256k", 262144)):
        T = sum(len(r["ids"]) for r in rows[n])
        assert 0.9 * target <= T <= 1.02 * target, (n, T)
        ja = sum(len(r["ids"]) for r in rows[n] if r["lang"] == "ja") / T
        assert 0.37 <= ja <= 0.43, (n, ja)
    # hold-out: no teacher request, no JA-KL task / doc / chat key in calib
    teach = set()
    for f in ("agent24.jsonl", "ja16.jsonl"):
        for line in open(f"{B.TEACHER}/{f}"):
            t = json.loads(line)
            teach.add(f"{t['src']}@{t['off']}")
    def parts(r):
        return [p.split("+[")[0] for p in r["src"].split(";")]
    cal_parts = {p for r in rows["calib256k"] for p in parts(r)}
    jak_parts = {p for r in rows["jakl32"] for p in parts(r)}
    assert not (cal_parts & teach), cal_parts & teach
    assert not (jak_parts & teach), jak_parts & teach
    assert not (cal_parts & jak_parts), cal_parts & jak_parts
    # time split: calib request windows <= split, JA-KL request windows after it
    reqs = {f"{r['src']}@{r['off']}": r for r in B.load_requests(meta["args"]["split_ts"])}
    for p in cal_parts:
        if p in reqs:
            assert reqs[p]["half"] == "calib", p
    for p in jak_parts:
        if p in reqs:
            assert reqs[p]["half"] == "eval", p
    jt = set(meta["jakl_task_hashes"])
    cal_tasks = {reqs[p]["task_h"] for p in cal_parts if p in reqs}
    assert not (cal_tasks & jt), len(cal_tasks & jt)
    # chat keys (oasst first prompt hash) of JA-KL are not in calib
    jk = set(meta["jakl_chat_keys"])
    assert jk, "no chat keys recorded"
    key_of = {}
    for c in B.chat_items(B.ROOT + "/work/calib/src"):
        key_of[c["src"]] = c["key"]
    cal_chat = {key_of[p] for p in cal_parts if p in key_of}
    jak_chat = {key_of[p] for p in jak_parts if p in key_of}
    assert cal_chat and jak_chat and not (cal_chat & jak_chat), cal_chat & jak_chat
    assert jak_chat <= jk
    assert meta["overlap"]["calib256k_vs_jakl32"] < 0.01 and meta["overlap"]["calib256k_vs_teacher"] < 0.02
    print("files ok:", {n: (len(rr), sum(len(r["ids"]) for r in rr)) for n, rr in rows.items()})


if __name__ == "__main__":
    test_charstats()
    test_shingles()
    test_pack_nested()
    env = test_req_window()
    test_render(env)
    if "--skip-files" not in sys.argv:
        test_files()
    print("all tests passed")
