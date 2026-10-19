"""tools/vl_eval.py の採点関数の CPU テスト。 tools/.venv/bin/python tools/vl/test_vl_eval.py"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import importlib.util
import json
spec = importlib.util.spec_from_file_location("vl_eval", DSV41 + "/tools/vl_eval.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

def it(atype, answer, **kw):
    return dict(atype=atype, answer=answer, **kw)

CASES = [
    (it("str", "神戸"), "神戸", 1), (it("str", "神戸"), "神戸支店です。", 1), (it("str", "神戸"), "大阪", 0),
    (it("str", "feature/login-form"), "`feature/login-form`", 1), (it("str", "オン", aliases=["有効"]), "オフ", 0),
    (it("str", "オン", aliases=["有効"]), "有効です", 1), (it("str", "test_round_half"), "答え: test_round_half", 1),
    (it("str", "メールアドレスまたはパスワードが正しくありません。"), "「メールアドレスまたはパスワードが正しくありません。」", 1),
    (it("num", "1412"), "¥1,412", 1), (it("num", "1412"), "1,421円", 0), (it("num", "3070.44"), "$3,070.44", 1),
    (it("num", "２０２３"), "2023年", 1), (it("num", "15"), "１５", 1),
    (it("mcq", "B", options=["x", "y", "z", "w"]), "B", 1), (it("mcq", "B", options=["x", "y", "z", "w"]), "(B) y", 1),
    (it("mcq", "B", options=["x", "y", "z", "w"]), "答えは C です", 0), (it("mcq", "C", options=["りんご", "みかん", "ぶどう", "もも"]), "ぶどう", 1),
    (it("mcq", "A", options=["a1", "a2"]), "Ａ", 1),
    (it("ja_short", "白"), "白い", 1), (it("ja_short", "2つ"), "2", 1), (it("ja_short", "3人"), "三人", 1), (it("ja_short", "赤"), "青", 0),
    (it("ja_short", "白です。"), "白色", 1), (it("ja_short", "黒色"), "黒", 1), (it("ja_short", "一人"), "1人", 1), (it("ja_short", "茶色です。"), "黒色です", 0),
    (it("ned", "本日限定 鶏ちゃん"), "本日限定鶏ちゃん", 1), (it("ned", "abcd"), "abce", 0.75),
    (it("relaxed", "14"), "14.5", 1), (it("relaxed", "14"), "16", 0), (it("relaxed", "Yes"), "Yes", 1), (it("relaxed", "45%"), "44", 1),
    (it("anls", "0.28", answers=["0.28"]), "0.28.", 1), (it("anls", "abcdef", answers=["abcdef"]), "abcdeg", 1 - 1 / 6),
    (it("vqa", "dakota", answers=["dakota"] * 5 + ["dakota digital"] * 5), "Dakota", 1),
    (it("vqa", "x", answers=["dakota digital"] * 2 + ["y"] * 8), "dakota digital", 2 / 3),
    # レビューで足した境界例: 英数字の短い正解は語の境界で一致 / 先頭の冠詞 A を選択肢 A と取らない
    (it("str", "no"), "not sure", 0), (it("str", "no"), "No.", 1), (it("str", "Yes"), "yes, it is", 1),
    (it("relaxed", "No"), "Not available", 0), (it("str", "postgres-41682", aliases=["postgres"]), "postgres-41682 です", 1),
    (it("str", "2"), "12", 0), (it("str", "TaskQueue"), "クラス名は TaskQueue です", 1),
    (it("mcq", "B", options=["the bananas", "the stems"]), "A banana plant has stems.", 0),
    (it("mcq", "B", options=["the bananas", "the stems"]), "A banana plant has stems, so B.", 1),
    (it("mcq", "A", options=["the bananas", "the stems"]), "A the bananas", 1),
    (it("mcq", "A", options=["the bananas", "the stems"]), "A.", 1), (it("mcq", "C", options=["x", "y", "z", "w"]), "C wins", 1),
    (it("mcq", "B", options=["x", "y", "z", "w"]), "The answer is b", 0),
]
bad = 0
for item, pred, want in CASES:
    got = E.score(item, pred)
    if abs(got - want) > 1e-6:
        bad += 1
        print("FAIL", item, repr(pred), got, want)
assert E.zh_count("这是一个问题") >= 3 and E.zh_count("これは日本語の文章です。見る、関係、応答、様子、時間") == 0
# zh_only は tools/zh_regex.py と同じ定義 (旧表に無い简体字も数える。日本語の常用漢字は数えない)
assert E.zh_count("软件设计们") == 4 and E.zh_count("机场和铁路") == 2 and E.zh_count("国際空港と鉄道、機械学習") == 0
assert E.zh_count("这个", gold="这个") == 0
# セットの全問で正解そのものを入れたら満点になること (採点の自己整合)
items = [json.loads(l) for l in open(DSV41 + "/work/vl/vl_gen.jsonl")]
low = [(x["id"], E.score(x, x["answer"])) for x in items if E.score(x, x["answer"]) < 1.0]
print(f"{len(CASES)} cases, {bad} failed; gold-as-pred < 1.0: {low[:10]} ({len(low)}/{len(items)})")
assert bad == 0 and not low
