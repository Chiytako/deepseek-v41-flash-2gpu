"""中国語混入の生成評価に使う日本語プロンプト 200 本を作る (CPU)。

構成 (r5_zh_mixing.md §5.1):
  traffic (要求ログの後半 = 2026-10-15 09:32 以降。校正に使う前半とは分ける)
    post_tool 50 : 最後のメッセージがツール結果で、GLM が日本語の本文で締めた要求 (ツール後の要約)
    coding    30 : ツール付きで、最後が利用者の日本語の指示。GLM の出力に日本語の本文がある
    explain   40 : 最後が利用者の日本語の質問で、GLM の日本語回答が 300〜1000 字 (説明・計算・計画・比較)
    long      20 : GLM の日本語回答が 1000 字以上 (コードブロックと空白を除く。後半で 2000 字以上は 5 件しかない)
    short     10 : GLM の日本語回答が 100 字以下 (状況の要約などの短い指示)
  public (外に出せる、再現できる部分)
    elyza     20 : ELYZA-tasks-100 (先頭から 5 件おき)
    jmtbench  10 : Japanese MT-Bench の第 1 ターン (カテゴリごとに 1〜2 件)
  stress (中国語に引かれやすい条件。手書きの雛形 + 中国語コーパスの断片)
    stress    20 : 中国語を含むツール結果の日本語要約、中国の地名・人名・故事、漢字の多い専門文、中国語コメントのコード
同じタスク (最初の利用者発話のハッシュ) からは 2 本まで、同じ雛形 (最後の利用者発話の書き出し 12 字) からは
TEMPLATE_CAP 本まで。後半で足りない区分は少なくなる (その分は数を表示する)。乱数の種は固定。
harness (原重み、1.4〜1.75 tok/s) 用に、prompt が 8,000 トークン以下のものから区分ごとに層化して 20 本を選び
"harness": true を付ける。

出力: work/zh/promptset/prompts.jsonl (1 行 1 本: id, cat, source, messages, tools, thinking, reasoning_effort,
      plen, harness, glm_answer_chars, glm_level)。要求ログ由来の行は外に出さないこと。
usage: zh_promptset.py [--reqlog $GLM53/dump/reqlog] [--workers 8]
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
from collections import Counter, defaultdict

import orjson

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zh_detect as Z  # noqa: E402

ROOT = DSV41
CUTOFF = "2026-10-15T09:32"
SEED = 20261018
QUOTA = {"post_tool": 50, "coding": 30, "explain": 40, "long": 20, "short": 10}
HARNESS_QUOTA = {"post_tool": 4, "coding": 2, "explain": 4, "long": 2, "short": 1, "elyza": 2, "jmtbench": 2, "stress": 3}

TC_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join((b.get("text") or "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def first_text(msgs):
    if not msgs:
        return ""
    last = msgs[-1].get("content")
    if isinstance(last, list):
        last = " ".join(b.get("text", "") for b in last if isinstance(b, dict))
    return (last or "")[:40]


def scan(args):
    """1 ファイルの要求を (rid, ts, messages, tools, thinking, effort, glm_text) にする (x2_extract と同じ対応付け)。"""
    path, want = args
    from collections import deque
    pend_oa = deque()
    pending = {}
    out = []
    with open(path, "rb") as f:
        for line in f:
            head = line[:400]
            i = line.find(b"{")
            if i < 0:
                continue
            if b'"event": "request.received.openai"' in head:
                try:
                    d = orjson.loads(line[i:])
                except orjson.JSONDecodeError:
                    continue
                o = d.get("obj") or {}
                ctk = o.get("chat_template_kwargs") or {}
                th = ctk.get("enable_thinking", ctk.get("thinking", True))
                pend_oa.append({"ts": d.get("timestamp"), "messages": o.get("messages") or [], "tools": o.get("tools"),
                                "thinking": th is not False, "effort": o.get("reasoning_effort")})
            elif b'"event": "request.received"' in head:
                if not pend_oa:
                    continue
                d = orjson.loads(line[i:])
                rec = pend_oa.popleft()
                if first_text(rec["messages"]) in ((d.get("obj") or {}).get("text") or ""):
                    pending[d.get("rid")] = rec
            elif b'"event": "request.finished"' in head:
                m = re.search(rb'"rid": "([0-9a-f]+)"', head)
                if not m:
                    continue
                rid = m.group(1).decode()
                rec = pending.pop(rid, None)
                if rec is None or rid not in want:
                    continue
                d = orjson.loads(line[i:])
                rec["rid"] = rid
                rec["glm_text"] = ((d.get("out") or {}).get("text") or "")
                out.append(rec)
    return out


def glm_answer(text):
    rest = text.split("</think>", 1)[1] if "</think>" in text else text
    has_tc = bool(TC_RE.search(rest))
    return TC_RE.sub("", rest).strip(), has_tc


def task_hash(msgs):
    for m in msgs:
        if m.get("role") == "user":
            t = re.sub(r"<system-reminder>.*?</system-reminder>", "", text_of(m.get("content")), flags=re.S).strip()
            return hashlib.md5(t[:400].encode()).hexdigest()[:10]
    return "none"


def template_key(msgs):
    """最後の利用者発話の書き出し 12 字。同じ雛形 (定型の指示の繰り返し) の偏りを抑えるのに使う。"""
    for m in reversed(msgs):
        if m.get("role") == "user":
            t = re.sub(r"<system-reminder>.*?</system-reminder>", "", text_of(m.get("content")), flags=re.S).strip()
            return t[:12]
    return ""


TEMPLATE_CAP = {"post_tool": 8, "coding": 4, "explain": 5, "long": 4, "short": 10}


def categorize(rec):
    msgs = rec["messages"]
    if not msgs:
        return None
    last = msgs[-1].get("role")
    ans, has_tc = glm_answer(rec["glm_text"])
    a = Z.clean(ans)
    if not Z.is_japanese(a):
        return None
    n = len(a)
    lu = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            lu = re.sub(r"<system-reminder>.*?</system-reminder>", "", text_of(m.get("content")), flags=re.S)
            break
    if not Z.is_japanese(lu, min_kana=3):
        return None
    if n >= 1000:
        return "long"
    if last == "tool" and not has_tc and n >= 80:
        return "post_tool"
    if last == "user" and n <= 100 and not has_tc:
        return "short"
    if last == "user" and rec.get("tools") and n >= 80:
        return "coding"
    if last == "user" and 300 <= n < 1000 and not has_tc:
        return "explain"
    return None


STRESS = [
    # (説明, 利用者の発話, ツール結果 (None ならツール無し))
    ("中国語の README を日本語で要約", "このリポジトリの README を読んで、何をするツールか日本語で 3 行にまとめてください。", "ZH_README"),
    ("中国語コメントのコードの説明", "次の Python 関数が何をしているか、日本語で説明してください。\n```python\ndef 计算平均值(数据):\n    # 过滤掉空值，然后计算平均数\n    有效 = [x for x in 数据 if x is not None]\n    return sum(有效) / len(有效) if 有效 else 0\n```", None),
    ("中国語のエラーログの分析", "ビルドが失敗しました。ログを見て原因と直し方を日本語で教えてください。", "ZH_LOG"),
    ("中国の地名", "上海・深圳・杭州のそれぞれの産業の特徴を、日本語で簡潔に比べてください。", None),
    ("故事成語", "「臥薪嘗胆」「呉越同舟」の由来と、現代の日本語での使い方を説明してください。", None),
    ("漢詩", "杜甫の「春望」の内容を、日本語の現代語訳と一緒に解説してください。", None),
    ("中国語の単語", "中国語の「加油」「麻烦」「没关系」は日本語でどういうニュアンスですか。日本語で説明してください。", None),
    ("中国の歴史", "清朝末期のアヘン戦争から辛亥革命までの流れを、日本語で年表風にまとめてください。", None),
    ("漢字の多い法律文", "民法の「善意の第三者」と「悪意の第三者」の違いを、具体例を挙げて日本語で説明してください。", None),
    ("漢字の多い医学文", "急性心筋梗塞の初期症状と、救急搬送までに周囲の人がすべきことを日本語で説明してください。", None),
    ("経済の専門語", "金融緩和政策が為替相場と輸出企業の業績に与える影響を、日本語で順を追って説明してください。", None),
    ("中国企業の比較", "華為技術 (Huawei) と小米 (Xiaomi) の事業構成の違いを日本語で整理してください。", None),
    ("中国語の論文要旨", "次の論文の要旨を読んで、日本語で要点を 5 つに整理してください。", "ZH_ABSTRACT"),
    ("中国語の issue", "この GitHub issue の内容を日本語で要約し、再現手順を整理してください。", "ZH_ISSUE"),
    ("日中翻訳の依頼", "次の中国語を自然な日本語に訳してください:「这个功能目前还在测试阶段，请谨慎使用。」", None),
    ("繁体字の資料", "台湾の健康保険制度の特徴を、日本の制度と比べて日本語で説明してください。", None),
    ("中国料理", "麻婆豆腐と回鍋肉の作り方の違いを、家庭で作る前提で日本語で説明してください。", None),
    ("長い漢字列の固有名詞", "中華人民共和国国家発展改革委員会の役割を、日本の省庁にたとえて日本語で説明してください。", None),
    ("コード + 中国語の仕様", "次の仕様書 (中国語) に沿って、Python で関数を実装し、日本語で説明してください。", "ZH_SPEC"),
    ("三国志", "三国志の赤壁の戦いについて、曹操・孫権・劉備の立場から日本語で簡単に説明してください。", None),
]


def stress_tool_texts():
    """中国語コーパス (work/zh/data) からツール結果に使う断片を作る。"""
    alp = []
    for p in sorted(glob.glob(f"{ROOT}/work/zh/data/alpaca_zh_*.json")):
        alp += [r["row"] for r in json.load(open(p, encoding="utf-8"))["rows"]]
    wiki = []
    for p in sorted(glob.glob(f"{ROOT}/work/zh/data/wiki_zh_*.json")):
        wiki += [r["row"]["text"] for r in json.load(open(p, encoding="utf-8"))["rows"]]
    rnd = random.Random(SEED)
    a = rnd.sample(alp, 4)
    return {
        "ZH_README": "# 项目简介\n\n" + a[0]["output"][:600] + "\n\n## 安装\n\n```bash\npip install -r requirements.txt\n```\n\n## 使用方法\n\n运行 `python main.py --config config.yaml` 即可开始。",
        "ZH_LOG": "error: 编译失败\n  --> src/main.rs:42:17\n   |\n42 |     let x: i32 = \"hello\";\n   |                  ^^^^^^^ 期望类型 `i32`，实际类型 `&str`\n\n错误: 由于之前的 1 个错误，无法编译 `demo`",
        "ZH_ABSTRACT": wiki[rnd.randrange(len(wiki))][:900],
        "ZH_ISSUE": "标题: 在 Windows 上运行时崩溃\n\n" + a[1]["output"][:400] + "\n\n复现步骤:\n1. 安装最新版本\n2. 打开设置页面\n3. 点击“保存”按钮后程序崩溃",
        "ZH_SPEC": "函数名: merge_intervals\n输入: 区间列表，每个区间为 [start, end]\n输出: 合并所有重叠区间后的列表，按起点升序排列\n要求: 时间复杂度 O(n log n)，不修改输入列表",
    }


def build_stress():
    tt = stress_tool_texts()
    out = []
    for k, (desc, user, tool) in enumerate(STRESS):
        if tool is None:
            msgs = [{"role": "user", "content": user}]
            tools = None
        else:
            tools = [{"type": "function", "function": {"name": "read", "description": "Read a file or URL",
                                                       "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
            msgs = [{"role": "user", "content": user},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{k}", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "README.md"})}}]},
                    {"role": "tool", "tool_call_id": f"call_{k}", "content": tt[tool]}]
        out.append({"id": f"stress_{k:02d}", "cat": "stress", "source": f"handwritten:{desc}", "messages": msgs, "tools": tools,
                    "thinking": True, "reasoning_effort": "high"})
    return out


def build_public():
    import pandas as pd
    out = []
    d = pd.read_csv(f"{ROOT}/work/r3ja/data/elyza100_test.csv")
    for k, q in enumerate(d["input"].tolist()[::5][:20]):
        out.append({"id": f"elyza_{k:02d}", "cat": "elyza", "source": "ELYZA-tasks-100", "messages": [{"role": "user", "content": q}],
                    "tools": None, "thinking": True, "reasoning_effort": "high"})
    rows = [json.loads(l) for l in open(f"{ROOT}/work/r3ja/data/jmtbench_q.jsonl", encoding="utf-8")]
    bycat = defaultdict(list)
    for r in rows:
        bycat[r["category"]].append(r)
    picked = []
    for c in sorted(bycat):
        picked += bycat[c][:2]
    for k, r in enumerate(picked[:10]):
        out.append({"id": f"jmtb_{k:02d}", "cat": "jmtbench", "source": f"Japanese MT-Bench q{r['question_id']} ({r['category']})",
                    "messages": [{"role": "user", "content": r["turns"][0]}], "tools": None, "thinking": True, "reasoning_effort": "high"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reqlog", default=GLM_REQLOG)
    ap.add_argument("--glm-eval", default=f"{ROOT}/work/zh/glm_eval.json")
    ap.add_argument("--out", default=f"{ROOT}/work/zh/promptset/prompts.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    rows = json.load(open(a.glm_eval, encoding="utf-8"))["rows"]
    want = {r["rid"]: r for r in rows if r.get("ts") and r["ts"] >= CUTOFF}
    # plen は x2 の meta から
    plen = {}
    for mf in glob.glob(f"{ROOT}/work/x2/req/*.meta.json"):
        for m in json.load(open(mf)):
            if m["rid"] in want:
                plen[m["rid"]] = m["plen"]
    files = sorted(glob.glob(a.reqlog + "/*"))
    recs = []
    with mp.Pool(a.workers) as pool:
        for r in pool.imap_unordered(scan, [(f, set(want)) for f in files]):
            recs.extend(r)
    print(f"[scan] 後半の日本語回答の候補 {len(want)} 件、要求ログで見つかった {len(recs)} 件")
    bycat = defaultdict(list)
    for r in recs:
        c = categorize(r)
        if c:
            bycat[c].append(r)
    print("[cat] 候補数", {k: len(v) for k, v in bycat.items()})
    rnd = random.Random(SEED)
    out = []
    for c, q in QUOTA.items():
        pool_ = sorted(bycat[c], key=lambda r: r["rid"])
        rnd.shuffle(pool_)
        per_task = Counter()
        per_tpl = Counter()
        sel = []
        for r in pool_:
            th = task_hash(r["messages"])
            tk = template_key(r["messages"])
            if per_task[th] >= 2 or per_tpl[tk] >= TEMPLATE_CAP[c]:
                continue
            per_task[th] += 1
            per_tpl[tk] += 1
            sel.append(r)
            if len(sel) >= q:
                break
        for k, r in enumerate(sel):
            ans, _ = glm_answer(r["glm_text"])
            out.append({"id": f"{c}_{k:02d}", "cat": c, "source": f"reqlog:{r['rid']}", "messages": r["messages"], "tools": r["tools"],
                        "thinking": r["thinking"], "reasoning_effort": r.get("effort") or "high", "plen": plen.get(r["rid"]),
                        "glm_answer_chars": len(Z.clean(ans)), "glm_level": want[r["rid"]]["level"]})
    # 後半で足りない explain は ELYZA-tasks-100 の別の問 (2, 3 番目から 5 件おき) で埋める (校正との重なりを避けるため前半は使わない)
    short_n = QUOTA["explain"] - sum(o["cat"] == "explain" for o in out)
    if short_n > 0:
        import pandas as pd
        _q = pd.read_csv(f"{ROOT}/work/r3ja/data/elyza100_test.csv")["input"].tolist()
        qs = _q[2::5] + _q[3::5]
        for k, q in enumerate(qs[:short_n]):
            out.append({"id": f"explain_e{k:02d}", "cat": "explain", "source": "ELYZA-tasks-100 (explain の補充)",
                        "messages": [{"role": "user", "content": q}], "tools": None, "thinking": True, "reasoning_effort": "high"})
        print(f"[fill] explain を ELYZA で {min(short_n, len(qs))} 本補充")
    out += build_public()
    out += build_stress()
    # harness 用の 20 本 (prompt が短いものから層化)
    for c, q in HARNESS_QUOTA.items():
        lim = 32000 if c == "coding" else 8000  # coding は長い会話しか無い (中央 2.9 万トークン)
        cand = [o for o in out if o["cat"] == c and (o.get("plen") or 0) <= lim]
        cand.sort(key=lambda o: o["id"])
        rnd.shuffle(cand)
        for o in cand[:q]:
            o["harness"] = True
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for o in out:
            o.setdefault("harness", False)
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    cnt = Counter(o["cat"] for o in out)
    print(f"[write] {a.out} {len(out)} 本", dict(cnt), "harness", sum(o["harness"] for o in out))


if __name__ == "__main__":
    main()
