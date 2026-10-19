"""日本語の文章に混入した中国語を検出する (CPU、標準ライブラリのみ)。

判定に使う信号 (強い順):
  1. 字種
     J   : 日本語の字。JIS X 0208 (shift_jis) ∪ cp932 拡張 ∪ 常用漢字 ∪ 人名用漢字 (tools/zh_kanji_ja.json)。
     ZS  : 簡体字。GB2312 にあり J にない漢字 (JIS X 0213 にある簡体字、例: 运 达 种 关 两 么 你 她 も ZS に入れる)。
     ZT  : J にも JIS X 0213 にも GB2312 にもない漢字 (繁体字専用、拡張領域の稀字など)。日本語でも中国語でもない
           可能性があるので弱い信号として別に数える。
     JX  : J になく JIS X 0213 にあり GB2312 にない漢字 (旧字体など)。日本語として扱い、数えない。
  2. 中国語の機能語 (tools/zh_words.json。zh_calib.py が日本語コーパスと中国語コーパスでの出現率から選ぶ)。
     strong: 日本語コーパスでほぼ 0 回 (可以 没有 但是 如果 什么 …)。どこにあっても数える。
     weak  : 日本語でも稀に出る (其他 是否 …)。仮名を含まない句の中でだけ数える。
  3. 中国語の句: 句点・改行で区切った句のうち、仮名が 0 で漢字が 4 字以上あり、機能語・簡体字・中国語の
     句読点の点数が閾値以上のもの。文単位の中国語 (コードスイッチ) を拾う。
  4. 句読点: 仮名を含まない漢字の句の中の全角カンマ「，」と全角セミコロン「；」。
  5. 拼音: 声調記号付きのラテン文字の音節が 2 語以上続く、または第 3 声 (ǎ ě ǐ ǒ ǔ ǚ) を含む語。

偽陽性を減らす工夫:
  - コードブロック (```)、インラインコード、URL、DSML のツール呼び出しを除いてから判定する。
  - 機能語は「日本語コーパスで出ない」ことを実測して選ぶ (zh_calib.py)。日本語の漢字語 (可能、需要、
    非常、目前、了解 …) と重なる語は使わない。
  - 「的」「了」「是」「在」など日本語の漢字語にも出る 1 字は、仮名の無い句の中で、日本語で使わない
    並び (了 の直後が句読点で直前が 完終修 等でない、など) のときだけ点数にする。
  - context (プロンプトやツール結果) を渡すと、そこに既にある簡体字・機能語・句を「copied」として
    分け、モデルが新たに出したもの (novel) と区別する。

usage:
  zh_detect.py text [FILE|-]          1 行 1 文書 (JSON 文字列でも可) を判定して JSON Lines を出す
  zh_detect.py demo                    組み込みの例で動作を見る
library:
  from zh_detect import detect, char_class
"""
import json
import os
import re
import sys
import unicodedata
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
KANJI_JSON = os.path.join(HERE, "zh_kanji_ja.json")
WORDS_JSON = os.path.join(HERE, "zh_words.json")


# --------------------------------------------------------------------------------------------
# 字種
# --------------------------------------------------------------------------------------------
def _enc(ch, codec):
    try:
        ch.encode(codec)
        return True
    except UnicodeEncodeError:
        return False


def is_han(o):
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x3134F or 0x2E80 <= o <= 0x2FDF)


def is_kana(o):
    # ひらがな、カタカナ、カタカナ拡張、半角カナ。長音符 ー (30FC) を含む。
    return 0x3041 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F


_JA_LIST = None


def _ja_list():
    global _JA_LIST
    if _JA_LIST is None:
        try:
            d = json.load(open(KANJI_JSON, encoding="utf-8"))
            _JA_LIST = set(d.get("joyo", "")) | set(d.get("jinmeiyo", ""))
        except OSError:
            _JA_LIST = set()
    return _JA_LIST


# 実測で日本語に普通に出ることが分かった ZS 字を J に戻すための例外 (zh_calib.py の結果を見て手で足す)。
JA_EXCEPT = set("")


@lru_cache(maxsize=None)
def char_class(ch):
    """'J' | 'JX' | 'ZS' | 'ZT' | '' (漢字でない)"""
    o = ord(ch)
    if not is_han(o):
        return ""
    if ch in JA_EXCEPT or ch in _ja_list() or _enc(ch, "cp932") or _enc(ch, "shift_jis"):
        return "J"
    gb = _enc(ch, "gb2312")
    if gb:
        return "ZS"
    if _enc(ch, "euc_jis_2004"):
        return "JX"
    return "ZT"


# --------------------------------------------------------------------------------------------
# 機能語
# --------------------------------------------------------------------------------------------
# zh_calib.py が評価する候補。日本語の漢字語と重なりそうなもの (可能 需要 非常 目前 一般 重要 主要 了解 …) は
# 最初から入れない。ZS 字を含む語も入れておく (句の点数に使う。字の判定でも拾われる)。
CANDIDATES = """
可以 没有 但是 如果 什么 怎么 那么 这么 一个 一些 一下 不是 就是 也是 都是 还是 只是 而且 或者 然后
之后 以后 因此 所以 虽然 不过 比如 例如 其中 其他 以及 并且 而是 即使 只有 只要 无论 不仅 甚至 通过
对于 关于 由于 根据 按照 为了 因为 已经 正在 曾经 一直 一样 特别 十分 比较 更加 最好 应该 必须 能够
我们 你们 他们 她们 它们 我的 你的 他的 她的 它的 这个 那个 这些 那些 这样 那样 这里 那里 哪里 什么样
的是 是一 是在 在于 有一 有些 有的 很多 很好 很大 很有 很难 不能 不会 不要 不用 不同的 的时候 的话
的方法 的问题 的东西 得到 起来 出来 下来 进行 使得 让我 给我 告诉 知道 觉得 认为 希望 喜欢 现在 时间
东西 地方 问题 工作 一起 而言 来说 来看 总之 首先 其次 最后 另外 此外 同时 当然 然而 是否 如何 怎样
为什么 为何 多少 几个 每个 各种 所有 任何 自己 别人 大家 其实 确实 实际上 基本上 的一 了一 是不是
也就是说 就是说 也不 都不 还有 还没 没什么 有没有 能不能 可不可以 一种 这种 那种 该 吗 呢 吧 嘛 啊 呀
的人 的事 的话题 中的 上的 下的 我是 你是 他是 请 请问 谢谢 您 您好 你好
""".split()

DEFAULT_WORDS = {
    # zh_words.json が無いときの控えめな既定 (日本語では使わないことが明らかなもの)。
    "strong": "可以 没有 但是 如果 什么 怎么 那么 一个 一些 不是 就是 也是 都是 而且 然后 虽然 我们 他们 这个 那个 这些 那些 这样 那样 很多 的时候 为什么 是不是 也就是说".split(),
    "weak": "其他 是否 以及 其中 例如 只是 只有 只要 不过 此外 同时 首先".split(),
}


@lru_cache(maxsize=1)
def _words():
    try:
        d = json.load(open(WORDS_JSON, encoding="utf-8"))
        return {"strong": list(d["strong"]), "weak": list(d["weak"])}
    except (OSError, KeyError, ValueError):
        return DEFAULT_WORDS


@lru_cache(maxsize=1)
def _word_res():
    w = _words()

    def mk(ws):
        ws = sorted(set(ws), key=len, reverse=True)
        return re.compile("|".join(re.escape(x) for x in ws)) if ws else None

    return mk(w["strong"]), mk(w["weak"])


# 仮名の無い句の中でだけ数える 1 字の機能語の並び (日本語の漢字語と衝突しにくい形に限る)。
_RE_LE = re.compile(r"(?<![完終修満魅投校読未明了])了(?=[。，,！!？?、\s]|$)")   # 文末の 了
_RE_DE = re.compile(r"(?<=[我你他她它们這这那])的|的(?=[一这那我你他她它])")             # 我的 / 的一
_RE_SHI = re.compile(r"(?<=[这那我你他她它])是|是(?=[一这那我你他她它])")               # 这是 / 是一
_RE_FUNC1 = re.compile(r"(?<=[\u3400-\u9fff])的(?=[\u3400-\u9fff])|[了是在也都很就把被着过还这们]")
_RE_FINAL = re.compile(r"[吗呢吧嘛啊呀](?=[。，,！!？?\s]|$)")

_RE_PINYIN_TONE = re.compile(r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]")
_RE_PINYIN_3RD = re.compile(r"[ǎěǐǒǔǚ]")
_PINYIN_SYL = re.compile(
    r"^(?:(?:zh|ch|sh|[bpmfdtnlgkhjqxrzcsyw])?"
    r"(?:iang|iong|uang|ueng|ang|eng|ing|ong|uan|uai|ian|iao|uen|üan|ai|ei|ao|ou|an|en|er|in|un|ün|ia|ie|iu|ua|uo|ui|üe|ue|a|o|e|i|u|ü|v)"
    r"(?:ng|n|r)?)+$")


def _strip_tone(w):
    return "".join(c for c in unicodedata.normalize("NFD", w) if not unicodedata.combining(c))


def _pinyin_words(text):
    out = []
    words = re.findall(r"[A-Za-zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüÜ']+", text)
    run = []
    for w in words + [""]:
        lw = w.lower()
        ok = bool(lw) and bool(_RE_PINYIN_TONE.search(lw)) and bool(_PINYIN_SYL.match(_strip_tone(lw).replace("'", "")))
        if ok:
            run.append(w)
            continue
        if run and (len(run) >= 2 or any(_RE_PINYIN_3RD.search(x.lower()) for x in run)):
            out.extend(run)
        run = []
    return out


# --------------------------------------------------------------------------------------------
# 前処理
# --------------------------------------------------------------------------------------------
_RE_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
_RE_INLINE = re.compile(r"`[^`\n]{1,400}`")
_RE_URL = re.compile(r"https?://\S+")
_RE_DSML = re.compile(r"<｜DSML｜.*", re.S)
_RE_TAG = re.compile(r"</?(?:think|system-reminder)[^>]*>")


def clean(text, strip_code=True):
    t = _RE_DSML.sub("", text)
    if strip_code:
        t = _RE_FENCE.sub("\n", t)
        t = _RE_INLINE.sub(" ", t)
    t = _RE_URL.sub(" ", t)
    t = _RE_TAG.sub(" ", t)
    return t


_RE_PHRASE_SPLIT = re.compile(r"(?<=[。！？!?\n])|(?<=\.)\s")


def text_stats(t):
    kana = han = nonsp = 0
    for ch in t:
        if ch.isspace():
            continue
        nonsp += 1
        o = ord(ch)
        if is_kana(o):
            kana += 1
        elif is_han(o):
            han += 1
    return kana, han, nonsp


def is_japanese(t, min_kana=10):
    k, h, n = text_stats(t)
    return k >= min_kana and (k + h) >= 0.1 * max(1, n)


# --------------------------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------------------------
def _phrase_score(ph, strong_re, weak_re):
    """仮名の無い漢字の句の点数と内訳。"""
    s = 0.0
    why = []
    for c in ph:
        if char_class(c) == "ZS":
            s += 1.0
            why.append(c)
    if strong_re:
        for m in strong_re.finditer(ph):
            s += 1.5
            why.append(m.group())
    if weak_re:
        for m in weak_re.finditer(ph):
            s += 1.0
            why.append(m.group())
    for rx, w in ((_RE_LE, 1.0), (_RE_DE, 1.0), (_RE_SHI, 1.0), (_RE_FINAL, 1.0)):
        for m in rx.finditer(ph):
            s += w
            why.append(m.group())
    n_punct = len(re.findall(r"(?<=[㐀-鿿])[，；](?=[㐀-鿿])", ph))
    if n_punct:
        s += 0.5 * n_punct
        why.append("，" * n_punct)
    return s, why, n_punct


def detect(text, context=None, strip_code=True, clause_min_han=4, clause_thresh=2.0):
    """text の中国語混入を数える。context (プロンプトやツール結果の文字列) にあるものは copied に分ける。

    返り値の level: 0 = 混入なし、1 = 弱い信号だけ (ZT 字、弱い機能語、句読点、拼音)、
                     2 = 簡体字 (ZS) か強い機能語がある、3 = 中国語の句 (文単位の中国語) がある。
    level は novel (context に無いもの) だけで決める。"""
    t = clean(text, strip_code)
    ctx = clean(context, strip_code) if context else ""
    strong_re, weak_re = _word_res()
    kana, han, nonsp = text_stats(t)

    zs, zt = [], []
    for c in t:
        k = char_class(c)
        if k == "ZS":
            zs.append(c)
        elif k == "ZT":
            zt.append(c)
    ctx_chars = set(ctx) if ctx else set()

    def split_novel(items):
        nov = [x for x in items if not (ctx and (x in ctx_chars if len(x) == 1 else x in ctx))]
        return nov, len(items) - len(nov)

    sw = [m.group() for m in strong_re.finditer(t)] if strong_re else []
    # 弱い機能語は仮名の無い句の中だけ
    ww = []
    clauses = []
    n_punct = 0
    for ph in _RE_PHRASE_SPLIT.split(t):
        ph = ph.strip()
        if not ph:
            continue
        k, h, n = text_stats(ph)
        if k > 0 or h < clause_min_han:
            continue
        if weak_re:
            ww.extend(m.group() for m in weak_re.finditer(ph))
        s, why, npun = _phrase_score(ph, strong_re, weak_re)
        n_punct += npun
        s_np = s - min(1.0, 0.5 * npun)
        if s_np >= 1.0 and (s >= clause_thresh or (h >= 12 and h >= 0.6 * n)):
            copied = bool(ctx) and (ph[:12] in ctx)
            clauses.append({"text": ph[:120], "score": round(s, 1), "why": why[:8], "copied": copied})

    pin = _pinyin_words(t)
    zs_n, zs_c = split_novel(zs)
    zt_n, zt_c = split_novel(zt)
    sw_n, sw_c = split_novel(sw)
    ww_n, ww_c = split_novel(ww)
    cl_n = [c for c in clauses if not c["copied"]]

    if cl_n:
        level = 3
    elif zs_n or sw_n:
        level = 2
    elif zt_n or ww_n or n_punct or pin:
        level = 1
    else:
        level = 0
    return {
        "level": level,
        "is_ja": kana >= min(10, max(3, 0.2 * nonsp)) and (kana + han) >= 0.1 * max(1, nonsp),  # 短い回答 (C1 型) も拾う
        "chars": nonsp, "kana": kana, "han": han,
        "zs": "".join(zs_n), "zs_copied": zs_c,
        "zt": "".join(zt_n), "zt_copied": zt_c,
        "strong_words": sw_n, "strong_copied": sw_c,
        "weak_words": ww_n, "weak_copied": ww_c,
        "clauses": cl_n, "clauses_copied": len(clauses) - len(cl_n),
        "zh_punct": n_punct, "pinyin": pin,
    }


DEMO = [
    "この関数は入力が空のときに None を返します。修正後はテストがすべて通りました。",
    "この関数は可以正常に動きます。我们は次にテストを書きます。",
    "修正しました。这个函数现在可以正常工作了，测试也全部通过。",
    "具体的には、基本的な方針として現在の実装を維持し、必要に応じて完了したタスクを整理します。",
    "結論：方針A（推奨）、方針B，方針C。",
    "北京（Běijīng）は中国の首都です。",
    "```python\n# 这是注释\nprint('你好')\n```\nコードのコメントは中国語のままにしています。",
    "東京都千代田区永田町一丁目七番一号所在国会議事堂",
]


def main(argv):
    if len(argv) < 2 or argv[1] not in ("text", "demo"):
        print(__doc__)
        return 2
    if argv[1] == "demo":
        for s in DEMO:
            r = detect(s)
            print(json.dumps({"text": s[:60], **{k: r[k] for k in ("level", "zs", "zt", "strong_words", "weak_words", "clauses", "zh_punct", "pinyin")}}, ensure_ascii=False))
        return 0
    f = open(argv[2], encoding="utf-8") if len(argv) > 2 and argv[2] != "-" else sys.stdin
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            s = json.loads(line)
            if isinstance(s, dict):
                s = s.get("text", "")
        except ValueError:
            s = line
        print(json.dumps(detect(s), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
