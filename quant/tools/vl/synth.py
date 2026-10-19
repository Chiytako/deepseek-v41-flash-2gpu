"""自作画像 (図表・日本語/英語の文書 OCR・コード/ターミナル/Web 画面のスクリーンショット) を作り、正解付きの問いを返す。

画像は自分で描画するので正解は生成時に確定し、ライセンスは自作 (フォントは Noto CJK / DejaVu、SIL OFL / 自由)。
描画: 図表は matplotlib (Agg)、それ以外は HTML を Firefox headless でスクリーンショットにする。

各テンプレートは Inst を返す。qas は短答・数値の問い (生成評価用)、kl_q / kl_ref は説明を求める問いと参照回答
(教師 KL・校正用の応答本文)。乱数の種を split ごとに分けるので、評価と校正で同じ画像は出ない。
"""
import datetime
import html as H
import os
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

JA_FONT = "Noto Sans CJK JP"
MONO = "DejaVu Sans Mono"


@dataclass
class Inst:
    kind: str            # chart-bar, receipt, code, terminal-pytest, web-login, ...
    cat: str             # chart | ocr | screenshot
    lang: str            # ja | en
    qas: list            # [{q, a, aliases, atype, tol}]
    kl_q: str
    kl_ref: str
    html: str = ""
    size: tuple = (1280, 800)
    fig: object = None
    meta: dict = field(default_factory=dict)


def yen(v):
    return f"¥{v:,}"


WEEK_JA = "月火水木金土日"


def rand_date(rng, y0=2024, y1=2026):
    d = datetime.date(rng.randint(y0, y1), rng.randint(1, 12), rng.randint(1, 28))
    return d


def ja_date(d):
    return f"{d.year}年{d.month}月{d.day}日({WEEK_JA[d.weekday()]})"


def date_aliases(d):
    return [f"{d.year}年{d.month}月{d.day}日", f"{d.year}/{d.month:02d}/{d.day:02d}", f"{d.year}/{d.month}/{d.day}",
            f"{d.year}-{d.month:02d}-{d.day:02d}"]


# ------------------------------------------------------------------------------------------------ pools
JA_BRANCH = ["札幌", "仙台", "東京", "名古屋", "大阪", "広島", "福岡", "那覇", "横浜", "神戸", "京都", "金沢", "新潟", "静岡"]
EN_REGION = ["North", "South", "East", "West", "Central", "Pacific", "Mountain", "Atlantic", "Midwest", "Gulf"]
JA_PROD = ["ノートPC", "タブレット", "スマートフォン", "モニター", "キーボード", "プリンター", "ルーター", "イヤホン"]
EN_PROD = ["Laptops", "Tablets", "Phones", "Monitors", "Keyboards", "Printers", "Routers", "Headphones"]
JA_SHARE = ["電車", "バス", "自転車", "徒歩", "自家用車", "タクシー", "その他"]
JA_SHARE2 = ["Python", "TypeScript", "Go", "Rust", "Java", "C++", "その他"]
EN_SHARE = ["Search", "Social", "Email", "Direct", "Referral", "Ads", "Other"]
JA_PEOPLE = ["佐藤 健太", "鈴木 美咲", "高橋 翔", "田中 優子", "伊藤 大輔", "渡辺 彩", "山本 拓也", "中村 真由美", "小林 誠", "加藤 恵",
             "吉田 陽菜", "山田 蓮", "松本 結衣", "井上 悠真", "木村 さくら"]
JA_PEOPLE_ROMA = {"佐藤": "sato", "鈴木": "suzuki", "高橋": "takahashi", "田中": "tanaka", "伊藤": "ito", "渡辺": "watanabe",
                  "山本": "yamamoto", "中村": "nakamura", "小林": "kobayashi", "加藤": "kato", "吉田": "yoshida", "山田": "yamada",
                  "松本": "matsumoto", "井上": "inoue", "木村": "kimura"}
EN_PEOPLE = ["Emily Carter", "James Wilson", "Olivia Brown", "Liam Johnson", "Sophia Miller", "Noah Davis", "Ava Martinez",
             "Ethan Clark", "Mia Lewis", "Lucas Walker"]
JA_COMPANY = ["株式会社ミナト技研", "青葉システム株式会社", "さくらデータ合同会社", "株式会社北斗ソリューションズ", "東雲ラボ株式会社",
              "株式会社つばめ物流", "みなみ設計事務所"]
JA_COMPANY_DOM = ["minato-giken.co.jp", "aoba-sys.co.jp", "sakura-data.jp", "hokuto-sol.co.jp", "shinonome-lab.jp",
                  "tsubame-logi.co.jp", "minami-sekkei.jp"]
EN_COMPANY = ["Northwind Labs", "Bluefin Analytics", "Cedar & Pine Ltd.", "Orbit Systems Inc.", "Harbor Supply Co."]
JA_TITLE = ["営業部 部長", "開発部 課長", "情報システム部 主任", "経理部 係長", "技術本部 シニアエンジニア", "人事部 マネージャー"]
JA_STORE = ["まるやま商店", "カフェ こもれび", "ベーカリー 小麦堂", "スーパー あおば 中央店", "書店 ふみの森", "ドラッグ ひかり 駅前店"]
JA_ITEMS = [("おにぎり 鮭", 160), ("緑茶 500ml", 140), ("食パン 6枚切", 280), ("牛乳 1L", 230), ("バナナ", 198),
            ("卵 10個入", 268), ("ボールペン 黒", 120), ("大学ノート", 180), ("ブレンドコーヒー", 350), ("ミックスサンド", 420),
            ("カレーパン", 210), ("ヨーグルト", 158), ("単三電池 4本", 498), ("ハンドソープ", 328), ("のど飴", 238)]
EN_ITEMS = [("Consulting (hours)", 120), ("Cloud hosting - monthly", 85), ("SSL certificate", 49), ("Domain renewal", 18),
            ("Design revision", 60), ("Support plan", 199), ("Data migration", 450), ("Training session", 300)]
JA_EVENT = ["地域防災訓練", "夏祭り 盆踊り大会", "プログラミング勉強会", "古本市", "フリーマーケット", "健康ウォーキング大会", "子ども科学教室"]
JA_PLACE = ["市民会館 大ホール", "中央公園 野外ステージ", "第二小学校 体育館", "駅前コミュニティセンター 3階", "図書館 多目的室"]


def pick(rng, xs, k):
    return rng.sample(xs, k)


# ------------------------------------------------------------------------------------------------ charts
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager  # noqa: F401
    plt.rcParams["font.family"] = [JA_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def chart_bar(rng, lang):
    plt = _mpl()
    n = rng.randint(5, 7)
    if lang == "ja":
        labels = pick(rng, JA_BRANCH, n)
        year = rng.randint(2021, 2025)
        title = f"{year}年度 支店別売上高（百万円）"
        unit = "百万円"
    else:
        labels = pick(rng, EN_REGION, n)
        year = rng.randint(2021, 2025)
        title = f"FY{year} revenue by region (USD million)"
        unit = "million USD"
    while True:
        vals = [rng.randint(20, 240) for _ in labels]
        s = sorted(vals)
        if s[-1] - s[-2] >= 8 and s[1] - s[0] >= 8:
            break
    annot = rng.random() < 0.7
    horiz = rng.random() < 0.3
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=110)
    color = rng.choice(["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"])
    if horiz:
        bars = ax.barh(labels, vals, color=color)
        ax.invert_yaxis()
    else:
        bars = ax.bar(labels, vals, color=color)
    if annot:
        ax.bar_label(bars, labels=[str(v) for v in vals], padding=3, fontsize=12)
    ax.set_title(title, fontsize=15)
    ax.grid(axis="x" if horiz else "y", alpha=0.3)
    fig.tight_layout()
    imax, imin = vals.index(max(vals)), vals.index(min(vals))
    qas = []
    if lang == "ja":
        qas.append(dict(q="このグラフで売上高が最も大きい支店はどこですか？", a=labels[imax], atype="str"))
        qas.append(dict(q="このグラフで売上高が最も小さい支店はどこですか？", a=labels[imin], atype="str"))
    else:
        qas.append(dict(q="Which region has the highest revenue in this chart?", a=labels[imax], atype="str"))
        qas.append(dict(q="Which region has the lowest revenue in this chart?", a=labels[imin], atype="str"))
    if annot:
        j = rng.randrange(n)
        if lang == "ja":
            qas.append(dict(q=f"{labels[j]}支店の売上高は何百万円ですか？数値で答えてください。", a=str(vals[j]), atype="num"))
            qas.append(dict(q="最も大きい支店と最も小さい支店の売上高の差は何百万円ですか？", a=str(max(vals) - min(vals)), atype="num"))
            qas.append(dict(q="売上高が 100 百万円を超えている支店はいくつありますか？", a=str(sum(v > 100 for v in vals)), atype="num"))
        else:
            qas.append(dict(q=f"What is the revenue of the {labels[j]} region? Answer with a number.", a=str(vals[j]), atype="num"))
            qas.append(dict(q="What is the difference between the highest and the lowest revenue?", a=str(max(vals) - min(vals)), atype="num"))
    else:
        a, b = rng.sample(range(n), 2)
        while abs(vals[a] - vals[b]) < 0.2 * max(vals[a], vals[b]):
            a, b = rng.sample(range(n), 2)
        w = labels[a] if vals[a] > vals[b] else labels[b]
        if lang == "ja":
            qas.append(dict(q=f"{labels[a]}と{labels[b]}では、どちらの売上高が大きいですか？", a=w, atype="str"))
        else:
            qas.append(dict(q=f"Which has higher revenue, {labels[a]} or {labels[b]}?", a=w, atype="str"))
    order = sorted(range(n), key=lambda i: -vals[i])
    if lang == "ja":
        kl_q = "このグラフから読み取れることを説明してください。"
        body = "、".join(f"{labels[i]}が{vals[i]}" for i in order) if annot else ""
        kl_ref = (f"これは{year}年度の支店別売上高（単位は百万円）を示す{'横' if horiz else '縦'}棒グラフです。"
                  f"最も売上高が大きいのは{labels[imax]}で、最も小さいのは{labels[imin]}です。")
        if annot:
            kl_ref += (f"値の大きい順に{body}（百万円）となっており、最大と最小の差は{max(vals) - min(vals)}百万円です。"
                       f"{n}支店の合計は{sum(vals)}百万円で、1支店あたりの平均は約{round(sum(vals) / n)}百万円です。")
        else:
            kl_ref += f"大きい順に並べると、{'、'.join(labels[i] for i in order)}の順になります。各棒の値は目盛りから読み取る必要があります。"
    else:
        kl_q = "Describe what this chart shows."
        kl_ref = (f"This is a {'horizontal' if horiz else 'vertical'} bar chart of FY{year} revenue by region, in millions of USD. "
                  f"{labels[imax]} has the highest revenue and {labels[imin]} the lowest.")
        if annot:
            kl_ref += (" In descending order: " + ", ".join(f"{labels[i]} {vals[i]}" for i in order) +
                       f". The gap between the highest and lowest is {max(vals) - min(vals)} million, and the total is {sum(vals)} million.")
    return Inst("chart-bar", "chart", lang, qas, kl_q, kl_ref, fig=fig)


def chart_line(rng, lang):
    plt = _mpl()
    years = list(range(2016, 2026))
    if lang == "ja":
        names = pick(rng, JA_PROD, 2)
        title = "製品別 年間出荷台数（千台）"
    else:
        names = pick(rng, EN_PROD, 2)
        title = "Annual shipments by product (thousand units)"
    series = []
    for _ in names:
        v = rng.randint(40, 120)
        s = []
        for _y in years:
            v = max(10, v + rng.randint(-18, 22))
            s.append(v)
        series.append(s)
    # 最大の年が一意になるように
    for s in series:
        m = max(s)
        if s.count(m) > 1:
            s[s.index(m)] = m + 3
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=110)
    for nm, s, mk in zip(names, series, ["o", "s"]):
        ax.plot(years, s, marker=mk, label=nm, linewidth=2)
    for nm, s in zip(names, series):
        ax.annotate(str(s[-1]), (years[-1], s[-1]), textcoords="offset points", xytext=(6, 0), fontsize=11)
    ax.set_xticks(years)
    ax.set_title(title, fontsize=15)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    k = rng.randrange(2)
    peak = years[series[k].index(max(series[k]))]
    yy = rng.choice([y for y in years if abs(series[0][years.index(y)] - series[1][years.index(y)]) >= 12] or [years[-1]])
    hi = names[0] if series[0][years.index(yy)] > series[1][years.index(yy)] else names[1]
    if lang == "ja":
        qas = [dict(q=f"{names[k]}の出荷台数が最も多かったのは何年ですか？", a=str(peak), atype="num"),
               dict(q=f"{yy}年の出荷台数が多いのは{names[0]}と{names[1]}のどちらですか？", a=hi, atype="str"),
               dict(q=f"2025年の{names[k]}の出荷台数は何千台ですか？", a=str(series[k][-1]), atype="num")]
        kl_q = "この折れ線グラフの傾向を説明してください。"
        kl_ref = (f"このグラフは2016年から2025年までの{names[0]}と{names[1]}の年間出荷台数（千台）を示す折れ線グラフです。"
                  f"{names[0]}は{series[0][0]}千台から始まり、{years[series[0].index(max(series[0]))]}年に最大の{max(series[0])}千台となり、"
                  f"2025年は{series[0][-1]}千台です。{names[1]}は{series[1][0]}千台から始まり、"
                  f"{years[series[1].index(max(series[1]))]}年に最大の{max(series[1])}千台となり、2025年は{series[1][-1]}千台です。"
                  f"2025年時点では{names[0] if series[0][-1] > series[1][-1] else names[1]}の方が多くなっています。")
    else:
        qas = [dict(q=f"In which year were {names[k]} shipments the highest?", a=str(peak), atype="num"),
               dict(q=f"In {yy}, which product had more shipments, {names[0]} or {names[1]}?", a=hi, atype="str"),
               dict(q=f"How many thousand units of {names[k]} were shipped in 2025?", a=str(series[k][-1]), atype="num")]
        kl_q = "Describe the trends in this line chart."
        kl_ref = (f"The chart shows annual shipments of {names[0]} and {names[1]} from 2016 to 2025 in thousands of units. "
                  f"{names[0]} starts at {series[0][0]}, peaks at {max(series[0])} in {years[series[0].index(max(series[0]))]}, and ends at {series[0][-1]} in 2025. "
                  f"{names[1]} starts at {series[1][0]}, peaks at {max(series[1])} in {years[series[1].index(max(series[1]))]}, and ends at {series[1][-1]}.")
    return Inst("chart-line", "chart", lang, qas, kl_q, kl_ref, fig=fig)


def chart_pie(rng, lang):
    plt = _mpl()
    n = rng.randint(4, 6)
    if lang == "ja":
        pool, title = rng.choice([(JA_SHARE, "通勤手段の割合（社員アンケート）"), (JA_SHARE2, "社内リポジトリの主要言語の割合")])
        labels = pick(rng, pool[:-1], n - 1) + [pool[-1]]
    else:
        labels = pick(rng, EN_SHARE[:-1], n - 1) + ["Other"]
        title = "Website traffic by source"
    while True:
        cuts = sorted(rng.sample(range(3, 98), n - 1))
        p = [b - a for a, b in zip([0] + cuts, cuts + [100])]
        sp = sorted(p)
        if min(p) >= 4 and len(set(p)) == n and sp[-1] - sp[-2] >= 3 and sp[-2] - sp[-3] >= 3:
            break
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    ax.pie(p, labels=labels, autopct="%d%%", startangle=90, counterclock=False, textprops={"fontsize": 12})
    ax.set_title(title, fontsize=15)
    fig.tight_layout()
    order = sorted(range(n), key=lambda i: -p[i])
    j = rng.randrange(n)
    if lang == "ja":
        qas = [dict(q="この円グラフで2番目に割合が大きい項目は何ですか？", a=labels[order[1]], atype="str"),
               dict(q=f"「{labels[j]}」の割合は何％ですか？", a=str(p[j]), atype="num"),
               dict(q="この円グラフで最も割合が大きい項目は何ですか？", a=labels[order[0]], atype="str")]
        kl_q = "この円グラフの内容を説明してください。"
        kl_ref = (f"「{title}」の円グラフです。割合の大きい順に、" + "、".join(f"{labels[i]}が{p[i]}%" for i in order) +
                  f"です。最も多いのは{labels[order[0]]}で全体の{p[order[0]]}%を占め、上位2項目で{p[order[0]] + p[order[1]]}%になります。")
    else:
        qas = [dict(q="Which category has the second-largest share in this pie chart?", a=labels[order[1]], atype="str"),
               dict(q=f"What percentage is '{labels[j]}'?", a=str(p[j]), atype="num"),
               dict(q="Which category has the largest share?", a=labels[order[0]], atype="str")]
        kl_q = "Explain what this pie chart shows."
        kl_ref = (f"The pie chart shows website traffic by source. In descending order: " +
                  ", ".join(f"{labels[i]} {p[i]}%" for i in order) +
                  f". {labels[order[0]]} is the largest source, and the top two together account for {p[order[0]] + p[order[1]]}%.")
    return Inst("chart-pie", "chart", lang, qas, kl_q, kl_ref, fig=fig)


# ------------------------------------------------------------------------------------------------ OCR documents
BASE_CSS = f"""
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: "{JA_FONT}", "DejaVu Sans", sans-serif; color: #1f2328; }}
"""


def receipt(rng, lang):
    d = rand_date(rng)
    k = rng.randint(3, 6)
    if lang == "ja":
        store = rng.choice(JA_STORE)
        items = [(n, p, rng.choice([1, 1, 1, 2, 3])) for n, p in pick(rng, JA_ITEMS, k)]
        sub = sum(p * q for _, p, q in items)
        tax = sub * 10 // 100
        total = sub + tax
        tel = f"0{rng.randint(3, 99)}-{rng.randint(100, 9999):04d}-{rng.randint(0, 9999):04d}"
        pay = rng.choice(["現金", "クレジット", "交通系IC", "QRコード決済"])
        rows = "".join(f"<tr><td>{H.escape(n)}{'' if q == 1 else f' ×{q}'}</td><td class=r>{yen(p * q)}</td></tr>" for n, p, q in items)
        tm = f"{rng.randint(8, 21)}:{rng.randint(0, 59):02d}"
        no = rng.randint(1000, 9999)
        body = f"""<div class=rc><div class=c><b style="font-size:26px">{H.escape(store)}</b><br>TEL {tel}</div>
<div>{ja_date(d)} {tm}　レジ{rng.randint(1, 4)}　No.{no}</div><hr>
<div class=c style="font-size:20px;margin:6px">領 収 書</div>
<table>{rows}</table><hr><table>
<tr><td>小計</td><td class=r>{yen(sub)}</td></tr><tr><td>消費税(10%)</td><td class=r>{yen(tax)}</td></tr>
<tr class=t><td>合計</td><td class=r>{yen(total)}</td></tr><tr><td>お支払い（{pay}）</td><td class=r>{yen(total)}</td></tr></table><hr>
<div class=c>お買い上げありがとうございました</div></div>"""
        j = rng.randrange(k)
        qas = [dict(q="このレシートの合計金額（税込）はいくらですか？", a=str(total), atype="num"),
               dict(q="このレシートの店名は何ですか？", a=store, atype="str"),
               dict(q="このレシートの電話番号を書き写してください。", a=tel, atype="str"),
               dict(q="消費税額はいくらですか？", a=str(tax), atype="num"),
               dict(q="支払い方法は何ですか？", a=pay, atype="str"),
               dict(q=f"「{items[j][0]}」の金額はいくらですか？", a=str(items[j][1] * items[j][2]), atype="num")]
        kl_q = "このレシートの内容を読み取って、まとめてください。"
        kl_ref = (f"{store}（TEL {tel}）のレシートで、日時は{ja_date(d)} {tm}です。購入したのは" +
                  "、".join(f"{n}{'' if q == 1 else f'を{q}点'}（{yen(p * q)}）" for n, p, q in items) +
                  f"の{k}品目です。小計は{yen(sub)}、消費税（10%）が{yen(tax)}で、合計は{yen(total)}です。支払い方法は{pay}です。")
    else:
        co = rng.choice(EN_COMPANY)
        client = rng.choice(EN_PEOPLE)
        items = [(n, p, rng.randint(1, 5)) for n, p in pick(rng, EN_ITEMS, k)]
        sub = sum(p * q for _, p, q in items)
        tax = round(sub * 0.08, 2)
        total = round(sub + tax, 2)
        inv = f"INV-{rng.randint(2024, 2026)}-{rng.randint(100, 999)}"
        due = d + datetime.timedelta(days=30)
        rows = "".join(f"<tr><td>{H.escape(n)}</td><td class=r>{q}</td><td class=r>${p:,.2f}</td><td class=r>${p * q:,.2f}</td></tr>" for n, p, q in items)
        body = f"""<div class=rc style="width:760px"><div style="display:flex;justify-content:space-between"><b style="font-size:26px">{H.escape(co)}</b>
<div style="text-align:right">INVOICE<br><b>{inv}</b></div></div>
<div>Bill to: {client}<br>Issue date: {d.strftime('%B %d, %Y')}<br>Due date: {due.strftime('%B %d, %Y')}</div><hr>
<table><tr><th style="text-align:left">Description</th><th class=r>Qty</th><th class=r>Unit price</th><th class=r>Amount</th></tr>{rows}</table><hr>
<table><tr><td>Subtotal</td><td class=r>${sub:,.2f}</td></tr><tr><td>Sales tax (8%)</td><td class=r>${tax:,.2f}</td></tr>
<tr class=t><td>Total due</td><td class=r>${total:,.2f}</td></tr></table></div>"""
        qas = [dict(q="What is the total amount due on this invoice?", a=f"{total:.2f}", atype="num"),
               dict(q="What is the invoice number?", a=inv, atype="str"),
               dict(q="Who is the invoice billed to?", a=client, atype="str"),
               dict(q="How many line items are on this invoice?", a=str(k), atype="num")]
        kl_q = "Summarize this invoice."
        kl_ref = (f"This is invoice {inv} from {co}, billed to {client}, issued {d.strftime('%B %d, %Y')} and due {due.strftime('%B %d, %Y')}. "
                  f"It has {k} line items: " + "; ".join(f"{n} x{q} = ${p * q:,.2f}" for n, p, q in items) +
                  f". The subtotal is ${sub:,.2f}, sales tax (8%) is ${tax:,.2f}, and the total due is ${total:,.2f}.")
    css = BASE_CSS + """body{background:#e9e6df;padding:30px;font-size:17px}
.rc{width:520px;margin:auto;background:#fffdf8;padding:26px 30px;box-shadow:0 2px 10px #0003}
.c{text-align:center} table{width:100%;border-collapse:collapse} td,th{padding:3px 0} .r{text-align:right}
.t td{font-size:22px;font-weight:bold;padding-top:6px} hr{border:none;border-top:1px dashed #555}"""
    page = f"<!doctype html><html><head><meta charset=utf-8><style>{css}</style></head><body>{body}</body></html>"
    return Inst("receipt" if lang == "ja" else "invoice", "ocr", lang, qas, kl_q, kl_ref, html=page, size=(900, 900))


def card(rng, lang):
    i = rng.randrange(len(JA_COMPANY))
    co, dom = JA_COMPANY[i], JA_COMPANY_DOM[i]
    nm = rng.choice(JA_PEOPLE)
    sei = nm.split()[0]
    mail = f"{JA_PEOPLE_ROMA[sei]}.{rng.choice(['k', 'm', 'y', 't', 'h', 's'])}@{dom}"
    title = rng.choice(JA_TITLE)
    tel = f"03-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    zipc = f"〒{rng.randint(100, 199)}-{rng.randint(0, 9999):04d}"
    addr = rng.choice(["東京都千代田区丸の内二丁目4番1号", "東京都港区芝浦三丁目1番21号", "東京都品川区大崎一丁目6番3号",
                       "東京都渋谷区神南一丁目12番16号"])
    bld = f"{rng.randint(5, 30)}階"
    page = f"""<!doctype html><html><head><meta charset=utf-8><style>{BASE_CSS}
body{{background:#d8dde3;padding:40px}} .cd{{width:860px;height:500px;background:#fff;margin:auto;padding:40px 50px;box-shadow:0 3px 14px #0004;position:relative}}
.co{{font-size:24px;color:#1b4f8a;font-weight:bold}} .ti{{margin-top:50px;font-size:18px;color:#444}} .nm{{font-size:46px;letter-spacing:8px;margin-top:6px}}
.ft{{position:absolute;bottom:40px;left:50px;font-size:17px;line-height:1.7}}</style></head><body><div class=cd>
<div class=co>{co}</div><div class=ti>{title}</div><div class=nm>{nm}</div>
<div class=ft>{zipc} {addr} {bld}<br>TEL {tel}　E-mail {mail}</div></div></body></html>"""
    qas = [dict(q="この名刺のメールアドレスを書き写してください。", a=mail, atype="str"),
           dict(q="この名刺の人物の氏名は何ですか？", a=nm, aliases=[nm.replace(" ", "")], atype="str"),
           dict(q="この名刺の会社名は何ですか？", a=co, atype="str"),
           dict(q="この名刺の電話番号は何ですか？", a=tel, atype="str"),
           dict(q="この人物の役職（部署を含む）は何ですか？", a=title, aliases=[title.split()[-1]], atype="str")]
    kl_q = "この名刺に書かれている情報をすべて書き出してください。"
    kl_ref = (f"会社名: {co}\n部署・役職: {title}\n氏名: {nm}\n住所: {zipc} {addr} {bld}\n電話番号: {tel}\nメールアドレス: {mail}")
    return Inst("card", "ocr", "ja", qas, kl_q, kl_ref, html=page, size=(960, 600))


def notice_vertical(rng, lang):
    ev = rng.choice(JA_EVENT)
    place = rng.choice(JA_PLACE)
    d = rand_date(rng, 2025, 2026)
    h0 = rng.choice([9, 10, 11, 13, 14])
    nth = rng.randint(3, 25)
    fee = rng.choice(["無料", "五百円", "千円"])
    org = rng.choice(["あおば町内会", "みなと区 地域振興課", "さくら台自治会", "市立図書館"])
    kan = "〇一二三四五六七八九"

    def kj(n):
        if n < 10:
            return kan[n]
        t, o = divmod(n, 10)
        return ("" if t == 1 else kan[t]) + "十" + ("" if o == 0 else kan[o])
    dt = f"{kj(d.month)}月{kj(d.day)}日（{WEEK_JA[d.weekday()]}）"
    tm = f"午前{kj(h0)}時" if h0 < 12 else f"午後{kj(h0 - 12)}時"
    page = f"""<!doctype html><html><head><meta charset=utf-8><style>{BASE_CSS}
body{{background:#f1ead8;padding:30px}} .v{{writing-mode:vertical-rl;font-family:"Noto Serif CJK JP";height:620px;margin:auto;font-size:26px;line-height:2.1;
background:#fffaf0;padding:30px;border:3px double #7a5b2e}} h1{{font-size:44px;margin:0 0 0 20px}}</style></head><body>
<div class=v><h1>第{kj(nth)}回 {ev}</h1><p>日時　{dt}　{tm}より</p><p>会場　{place}</p><p>参加費　{fee}</p><p>主催　{org}</p>
<p>皆様お誘い合わせの上、ご参加ください。</p></div></body></html>"""
    hh = h0
    qas = [dict(q="このお知らせの会場はどこですか？", a=place, aliases=[place.split()[0]], atype="str"),
           dict(q="このお知らせの主催者は誰ですか？", a=org, atype="str"),
           dict(q="開始時刻は何時ですか？（24時間表記の数字で）", a=str(hh), atype="num"),
           dict(q="参加費はいくらですか？", a=fee, aliases={"無料": ["0円", "無料"], "五百円": ["500円", "500"], "千円": ["1000円", "1,000円", "1000"]}[fee], atype="str"),
           dict(q="何回目の開催ですか？数字で答えてください。", a=str(nth), atype="num")]
    kl_q = "この縦書きのお知らせを読み取り、内容を説明してください。"
    kl_ref = (f"縦書きのお知らせで、第{nth}回「{ev}」の案内です。日時は{d.month}月{d.day}日（{WEEK_JA[d.weekday()]}）の{h0}時から、"
              f"会場は{place}です。参加費は{fee}で、主催は{org}です。最後に「皆様お誘い合わせの上、ご参加ください。」と参加を呼びかけています。")
    return Inst("notice-vertical", "ocr", "ja", qas, kl_q, kl_ref, html=page, size=(1000, 760))


def manual_page(rng, lang):
    """技術マニュアル風の日本語文書 (段落 + 注意書き + 手順)。"""
    port = rng.choice([8080, 8443, 5432, 6379, 9000, 3000])
    ver = f"{rng.randint(1, 4)}.{rng.randint(0, 12)}.{rng.randint(0, 9)}"
    mem = rng.choice([4, 8, 16, 32])
    svc, unit = rng.choice([("データ連携サーバー", "datalink"), ("社内チャットボット", "chatbot"), ("ログ収集エージェント", "logagent"),
                            ("画像変換サービス", "imgconv")])
    cmd = rng.choice(["sudo systemctl restart ", "docker compose restart ", "kubectl rollout restart deployment/"])
    page = f"""<!doctype html><html><head><meta charset=utf-8><style>{BASE_CSS}
body{{background:#fff;padding:40px 60px;font-size:18px;line-height:1.8;width:1100px}} h2{{border-left:6px solid #2f6feb;padding-left:10px}}
.note{{background:#fff8c5;border:1px solid #d4a72c;padding:10px 16px;margin:14px 0}} code{{font-family:"{MONO}";background:#f0f2f5;padding:1px 5px}}</style></head><body>
<h2>3.2 {svc}の設定（バージョン {ver}）</h2>
<p>{svc}は、既定でポート <code>{port}</code> で待ち受けます。設定ファイルは <code>/etc/{unit}/config.yaml</code> にあり、変更後はサービスの再起動が必要です。</p>
<div class=note>注意：本バージョンでは、メモリが {mem}GB 未満の環境での動作は保証されません。</div>
<ol><li>設定ファイルのバックアップを取ります。</li><li><code>listen_port</code> の値を変更します。</li>
<li><code>{cmd}{unit}</code> を実行して再起動します。</li><li>ログに「起動完了」と表示されることを確認します。</li></ol></body></html>"""
    qas = [dict(q="このサービスが既定で待ち受けるポート番号は何ですか？", a=str(port), atype="num"),
           dict(q="動作が保証される最小メモリは何GBですか？", a=str(mem), atype="num"),
           dict(q="設定ファイルのパスを書き写してください。", a=f"/etc/{unit}/config.yaml", atype="str"),
           dict(q="この文書が対象とするバージョンは何ですか？", a=ver, atype="str")]
    kl_q = "この文書を読んで、手順と注意点を要約してください。"
    kl_ref = (f"{svc}（バージョン {ver}）の設定方法を説明した文書です。既定の待ち受けポートは{port}で、設定ファイルは/etc/{unit}/config.yamlにあります。"
              f"手順は、(1) 設定ファイルのバックアップを取る、(2) listen_port の値を変更する、(3) {cmd}{unit} を実行して再起動する、"
              f"(4) ログに「起動完了」と表示されることを確認する、の4つです。注意点として、メモリが{mem}GB未満の環境では動作が保証されません。")
    return Inst("manual", "ocr", "ja", qas, kl_q, kl_ref, html=page, size=(1220, 720))


# ------------------------------------------------------------------------------------------------ code screenshots
def _code_py(rng, lang):
    cls = rng.choice(["OrderService", "UserRepository", "CacheClient", "ReportBuilder", "TaskQueue"])
    fn = rng.choice(["load_config", "parse_args", "read_settings", "init_logger"])
    m = rng.choice(["fetch_items", "retry_request", "build_summary", "flush_pending", "resolve_path"])
    rtype = rng.choice(["list[dict]", "int", "str", "bool", "dict[str, int]"])
    retry = rng.randint(2, 9)
    timeout = rng.choice([5, 10, 15, 30, 60, 120])
    mods = pick(rng, ["os", "json", "time", "logging", "re", "sys", "hashlib", "itertools"], rng.randint(2, 4))
    c = {"ja": ["# 設定ファイルを読み込んで辞書で返す", "# 失敗したら最大 MAX_RETRY 回まで再試行する", "# 結果をまとめて返す"],
         "en": ["# Load the config file and return it as a dict", "# Retry up to MAX_RETRY times on failure", "# Return the collected result"]}[lang]
    lines = [f"import {x}" for x in mods] + ["from pathlib import Path", "", f"MAX_RETRY = {retry}", f"TIMEOUT_SEC = {timeout}", "", "",
             c[0], f"def {fn}(path: str) -> dict:", "    with open(Path(path), encoding=\"utf-8\") as f:",
             "        return json.load(f)" if "json" in mods else "        return dict(line.split(\"=\", 1) for line in f)", "", "",
             f"class {cls}:", "    def __init__(self, base_url: str, timeout: int = TIMEOUT_SEC):", "        self.base_url = base_url",
             "        self.timeout = timeout", "        self._pending = []", "",
             f"    def {m}(self, key: str) -> {rtype}:", "        " + c[1], "        for attempt in range(MAX_RETRY):", "            try:",
             "                return self._call(key)", "            except TimeoutError:", "                time.sleep(2 ** attempt)" if "time" in mods else "                continue",
             "        raise RuntimeError(f\"failed: {key}\")"]
    fn_line = lines.index(f"def {fn}(path: str) -> dict:") + 1
    m_line = next(i for i, l in enumerate(lines) if l.startswith(f"    def {m}(")) + 1
    fname = rng.choice(["service.py", "client.py", "app/core.py", "src/worker.py"])
    if lang == "ja":
        qas = [dict(q="MAX_RETRY の値はいくつですか？", a=str(retry), atype="num"),
               dict(q=f"関数 {fn} は何行目で定義されていますか？行番号で答えてください。", a=str(fn_line), atype="num"),
               dict(q=f"メソッド {m} の戻り値の型注釈は何ですか？", a=rtype, atype="str"),
               dict(q="このファイルで定義されているクラスの名前は何ですか？", a=cls, atype="str"),
               dict(q="TIMEOUT_SEC の値はいくつですか？", a=str(timeout), atype="num")]
        kl_q = "このコードのスクリーンショットを読んで、何をするコードか説明してください。"
        kl_ref = (f"{fname} の Python コードです。{', '.join(mods)} と pathlib.Path を import し、定数 MAX_RETRY = {retry} と TIMEOUT_SEC = {timeout} を定義しています。"
                  f"{fn_line} 行目の関数 {fn} は、設定ファイルを開いて辞書として返します。クラス {cls} はコンストラクタで base_url と timeout（既定値は TIMEOUT_SEC）を受け取り、"
                  f"{m_line} 行目のメソッド {m}（戻り値の型は {rtype}）は self._call(key) を呼び、TimeoutError が出たら最大 {retry} 回まで再試行し、"
                  f"すべて失敗すると RuntimeError を送出します。")
    else:
        qas = [dict(q="What is the value of MAX_RETRY?", a=str(retry), atype="num"),
               dict(q=f"On which line is the function {fn} defined? Answer with the line number.", a=str(fn_line), atype="num"),
               dict(q=f"What is the return type annotation of the method {m}?", a=rtype, atype="str"),
               dict(q="What is the name of the class defined in this file?", a=cls, atype="str")]
        kl_q = "Read this code screenshot and explain what the code does."
        kl_ref = (f"This is Python code from {fname}. It imports {', '.join(mods)} and pathlib.Path, and defines MAX_RETRY = {retry} and TIMEOUT_SEC = {timeout}. "
                  f"The function {fn} on line {fn_line} opens a config file and returns it as a dict. The class {cls} stores base_url and timeout, "
                  f"and its method {m} (line {m_line}, returning {rtype}) calls self._call(key), retrying up to {retry} times on TimeoutError before raising RuntimeError.")
    return "python", fname, "\n".join(lines), qas, kl_q, kl_ref


def _code_ts(rng, lang):
    iface = rng.choice(["User", "Order", "Invoice", "Ticket"])
    fields = [("id", "number"), ("name", "string"), ("createdAt", "Date"), (rng.choice(["status", "role", "priority"]), "string")]
    port = rng.choice([3000, 4000, 5173, 8080, 8787])
    route = rng.choice(["/api/users", "/api/orders", "/api/tickets", "/api/invoices"])
    fn = rng.choice(["fetchAll", "loadPage", "listRecent", "getItems"])
    limit = rng.choice([10, 20, 25, 50, 100])
    cm = {"ja": ["// 一覧を取得する（最大 PAGE_LIMIT 件）", "// サーバーを起動する"], "en": ["// Fetch the list (up to PAGE_LIMIT items)", "// Start the server"]}[lang]
    lines = ["import express from \"express\";", "import { z } from \"zod\";", "", f"const PORT = {port};", f"const PAGE_LIMIT = {limit};", "",
             f"interface {iface} {{"] + [f"  {a}: {b};" for a, b in fields] + ["}", "", cm[0],
             f"async function {fn}(page: number): Promise<{iface}[]> {{", f"  const res = await fetch(`{route}?page=${{page}}&limit=${{PAGE_LIMIT}}`);",
             "  if (!res.ok) throw new Error(`HTTP ${res.status}`);", f"  return (await res.json()) as {iface}[];", "}", "", "const app = express();",
             f"app.get(\"{route}\", async (req, res) => {{", f"  res.json(await {fn}(Number(req.query.page ?? 1)));", "});", "", cm[1],
             "app.listen(PORT, () => console.log(`listening on ${PORT}`));"]
    fn_line = next(i for i, l in enumerate(lines) if l.startswith(f"async function {fn}")) + 1
    fname = rng.choice(["server.ts", "src/index.ts", "api/routes.ts"])
    if lang == "ja":
        qas = [dict(q="サーバーが待ち受けるポート番号は何ですか？", a=str(port), atype="num"),
               dict(q=f"インターフェース {iface} のフィールドはいくつありますか？", a=str(len(fields)), atype="num"),
               dict(q="API のパス（ルート）は何ですか？", a=route, atype="str"),
               dict(q="PAGE_LIMIT の値はいくつですか？", a=str(limit), atype="num"),
               dict(q=f"関数 {fn} は何行目で定義されていますか？", a=str(fn_line), atype="num")]
        kl_q = "このコードの内容を説明してください。"
        kl_ref = (f"{fname} の TypeScript コードで、express と zod を import しています。PORT = {port}、PAGE_LIMIT = {limit} を定義し、"
                  f"インターフェース {iface} は {', '.join(f'{a}: {b}' for a, b in fields)} の {len(fields)} つのフィールドを持ちます。"
                  f"{fn_line} 行目の非同期関数 {fn} は {route} にページ番号と PAGE_LIMIT を付けて fetch し、応答が ok でなければエラーを投げ、JSON を {iface}[] として返します。"
                  f"express アプリは GET {route} でこの関数の結果を返し、ポート {port} で起動します。")
    else:
        qas = [dict(q="Which port does the server listen on?", a=str(port), atype="num"),
               dict(q=f"How many fields does the {iface} interface have?", a=str(len(fields)), atype="num"),
               dict(q="What is the API route path?", a=route, atype="str"),
               dict(q="What is the value of PAGE_LIMIT?", a=str(limit), atype="num")]
        kl_q = "Explain what this code does."
        kl_ref = (f"This TypeScript file ({fname}) sets PORT = {port} and PAGE_LIMIT = {limit} and declares an interface {iface} with {len(fields)} fields. "
                  f"The async function {fn} (line {fn_line}) fetches {route} with the page number and limit, throws on a non-OK response, and returns the JSON as {iface}[]. "
                  f"An express app serves GET {route} using {fn} and listens on port {port}.")
    return "typescript", fname, "\n".join(lines), qas, kl_q, kl_ref


def code_shot(rng, lang):
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name
    lx, fname, src, qas, kl_q, kl_ref = (rng.choice([_code_py, _code_py, _code_ts]))(rng, lang)
    style = rng.choice(["monokai", "default", "one-dark", "friendly", "github-dark", "solarized-light"])
    fmt = HtmlFormatter(style=style, linenos="table", noclasses=True, nobackground=False)
    code_html = highlight(src, get_lexer_by_name(lx), fmt)
    dark = style in ("monokai", "one-dark", "github-dark")
    bg = "#1e1f22" if dark else "#f3f3f3"
    page = f"""<!doctype html><html><head><meta charset=utf-8><style>{BASE_CSS}
body{{background:{bg}}} .tab{{background:{'#2b2d30' if dark else '#e4e4e4'};color:{'#ddd' if dark else '#222'};padding:8px 16px;font-size:15px;font-family:"{JA_FONT}"}}
.tab b{{background:{'#1e1f22' if dark else '#fff'};padding:6px 14px;border-radius:6px 6px 0 0}}
pre, td.linenos pre{{font-family:"{MONO}","{JA_FONT}" !important;font-size:16px !important;line-height:1.45 !important;margin:0}}
td.linenos pre{{color:#888 !important;padding-right:14px}} table{{margin:10px 6px}}
</style></head><body><div class=tab><b>{fname}</b>　</div>{code_html}</body></html>"""
    n_lines = src.count("\n") + 1
    h = min(1000, 80 + int(n_lines * 23.5))
    return Inst("code", "screenshot", lang, qas, kl_q, kl_ref, html=page, size=(1100, h), meta={"style": style, "lang_code": lx})


# ------------------------------------------------------------------------------------------------ terminal screenshots
def _term_page(lines_html, title, w=1180):
    return f"""<!doctype html><html><head><meta charset=utf-8><style>{BASE_CSS}
body{{background:#2b2b2b;padding:0}} .bar{{background:#3c3c3c;color:#ccc;font-size:14px;padding:6px 12px}}
pre{{margin:0;padding:12px 16px;color:#d6d6d6;background:#1d1f21;font-family:"{MONO}","{JA_FONT}";font-size:16px;line-height:1.45;white-space:pre-wrap;width:{w}px}}
.g{{color:#8fd16a}} .r{{color:#ff6b6b}} .y{{color:#e5c07b}} .b{{color:#61afef}} .d{{color:#888}} .w{{color:#fff;font-weight:bold}}</style></head>
<body><div class=bar>{H.escape(title)}</div><pre>{lines_html}</pre></body></html>"""


def term_pytest(rng, lang):
    user = rng.choice(["dev", "taro", "ci", "hana"])
    proj = rng.choice(["invoice-api", "scheduler", "text-utils", "billing"])
    mod = rng.choice(["test_parser", "test_dates", "test_pricing", "test_tokenize"])
    fails = rng.sample(["test_parse_empty", "test_parse_unicode", "test_round_half", "test_timezone_jst", "test_discount_rate",
                        "test_split_sentences", "test_negative_amount"], rng.randint(1, 2))
    n_pass = rng.randint(12, 60)
    line = rng.randint(20, 140)
    exp, got = rng.choice([("'2025-04-01'", "'2025-03-31'"), ("1100", "1099"), ("['今日は', '晴れ']", "['今日は晴れ']"), ("0.15", "0.1"), ("True", "False")])
    t = rng.uniform(0.4, 9.0)
    P = f'<span class=g>{user}@devbox</span>:<span class=b>~/{proj}</span>$ '
    out = [P + "pytest -q tests/", "." * min(n_pass, 40) + "F" * len(fails) + "." * max(0, n_pass - 40),
           "<span class=r>" + "=" * 30 + " FAILURES " + "=" * 30 + "</span>",
           f"<span class=r>______________ {fails[0]} ______________</span>", "",
           f"    def {fails[0]}():", f"&gt;       assert result == {H.escape(exp)}",
           f"<span class=r>E       AssertionError: assert {H.escape(got)} == {H.escape(exp)}</span>", "",
           f"tests/{mod}.py:{line}: AssertionError", "<span class=y>" + "=" * 24 + " short test summary info " + "=" * 24 + "</span>"]
    out += [f"<span class=r>FAILED</span> tests/{mod}.py::{f} - AssertionError" for f in fails]
    out += [f"<span class=r>{len(fails)} failed</span>, <span class=g>{n_pass} passed</span> in {t:.2f}s", P + "<span class=w>▌</span>"]
    page = _term_page("\n".join(out), f"{user}@devbox: ~/{proj}")
    if lang == "ja":
        qas = [dict(q="最初に詳細が表示されている失敗したテスト関数の名前は何ですか？", a=fails[0], atype="str"),
               dict(q="失敗したテストは何件ですか？", a=str(len(fails)), atype="num"),
               dict(q="成功（passed）したテストは何件ですか？", a=str(n_pass), atype="num"),
               dict(q="失敗したアサーションは何行目ですか？", a=str(line), atype="num"),
               dict(q="失敗したテストのファイル名は何ですか？", a=f"tests/{mod}.py", aliases=[f"{mod}.py"], atype="str")]
        kl_q = "このターミナルの出力を見て、何が起きているかと次にやるべきことを説明してください。"
        kl_ref = (f"~/{proj} で pytest を実行した結果、{n_pass} 件が成功し、{len(fails)} 件が失敗しています。失敗したのは " + "、".join(fails) +
                  f" で、tests/{mod}.py の {line} 行目の assert result == {exp} が成り立たず、実際の値は {got} でした。"
                  f"期待値 {exp} と実際の値 {got} の違いから、テスト対象の関数の処理（境界条件や丸め、区切りの扱いなど）を確認し、"
                  f"修正後に pytest tests/{mod}.py::{fails[0]} だけを再実行して通ることを確かめるのがよいでしょう。")
    else:
        qas = [dict(q="What is the name of the first failing test function shown in detail?", a=fails[0], atype="str"),
               dict(q="How many tests failed?", a=str(len(fails)), atype="num"),
               dict(q="How many tests passed?", a=str(n_pass), atype="num"),
               dict(q="On which line did the assertion fail?", a=str(line), atype="num")]
        kl_q = "Look at this terminal output and explain what happened and what to do next."
        kl_ref = (f"Running pytest in ~/{proj} gave {n_pass} passed and {len(fails)} failed tests ({', '.join(fails)}). "
                  f"The assertion at tests/{mod}.py line {line} expected {exp} but got {got}. Check the function under test for that case, "
                  f"fix it, and rerun pytest tests/{mod}.py::{fails[0]} to confirm.")
    return Inst("terminal-pytest", "screenshot", lang, qas, kl_q, kl_ref, html=page, size=(1220, 560))


def term_traceback(rng, lang):
    exc, msg = rng.choice([("KeyError", "'user_id'"), ("ZeroDivisionError", "division by zero"), ("FileNotFoundError", "[Errno 2] No such file or directory: 'data/input.csv'"),
                           ("TypeError", "unsupported operand type(s) for +: 'int' and 'str'"), ("AttributeError", "'NoneType' object has no attribute 'strip'"),
                           ("ValueError", "invalid literal for int() with base 10: 'abc'")])
    f1, f2 = rng.choice([("app/main.py", "app/handlers.py"), ("src/cli.py", "src/loader.py"), ("server.py", "utils/convert.py")])
    l1, l2 = rng.randint(10, 90), rng.randint(5, 200)
    fn1, fn2 = rng.choice([("main", "handle_request"), ("run", "load_rows"), ("cli", "convert")])
    P = '<span class=g>dev@devbox</span>:<span class=b>~/proj</span>$ '
    out = [P + f"python {f1}", "Traceback (most recent call last):", f'  File "{f1}", line {l1}, in &lt;module&gt;', f"    {fn1}()",
           f'  File "{f1}", line {l1 - 3}, in {fn1}', f"    result = {fn2}(payload)", f'  File "{f2}", line {l2}, in {fn2}',
           "    value = data[key] + offset", f"<span class=r>{exc}: {H.escape(msg)}</span>", P + "<span class=w>▌</span>"]
    page = _term_page("\n".join(out), "dev@devbox: ~/proj")
    if lang == "ja":
        qas = [dict(q="発生した例外の種類（クラス名）は何ですか？", a=exc, atype="str"),
               dict(q=f"例外が発生したのは {f2} の何行目ですか？", a=str(l2), atype="num"),
               dict(q="例外が発生した関数の名前は何ですか？", a=fn2, atype="str")]
        kl_q = "このエラーの原因と直し方を説明してください。"
        kl_ref = (f"python {f1} の実行中に {exc} が発生しています。呼び出しの流れは {f1} の {l1} 行目から {fn1}() が呼ばれ、"
                  f"{fn1} の中で {fn2}(payload) が呼ばれ、{f2} の {l2} 行目 `value = data[key] + offset` で例外になりました。"
                  f"メッセージは「{exc}: {msg}」です。この行で使っている値（data、key、offset）が想定どおりかを確認し、"
                  f"必要なら事前の検査や例外処理を追加してください。")
    else:
        qas = [dict(q="What exception type was raised?", a=exc, atype="str"),
               dict(q=f"On which line of {f2} was the exception raised?", a=str(l2), atype="num"),
               dict(q="In which function was the exception raised?", a=fn2, atype="str")]
        kl_q = "Explain the cause of this error and how to fix it."
        kl_ref = (f"Running python {f1} raised {exc}: {msg}. The call chain goes from {fn1}() in {f1} to {fn2}(payload), and the error occurs at {f2} line {l2}, "
                  f"`value = data[key] + offset`. Check that data, key and offset hold the expected values there, and add validation or error handling.")
    return Inst("terminal-traceback", "screenshot", lang, qas, kl_q, kl_ref, html=page, size=(1220, 420))


def term_git(rng, lang):
    br = rng.choice(["feature/login-form", "fix/date-parse", "feat/csv-export", "refactor/api-client", "hotfix/null-check"])
    mod = pick(rng, ["src/app.ts", "src/api/client.ts", "README.md", "package.json", "src/utils/date.ts", "tests/date.test.ts"], rng.randint(1, 3))
    staged = pick(rng, ["src/components/LoginForm.tsx", "src/hooks/useAuth.ts", "src/styles/form.css"], rng.randint(1, 2))
    untr = pick(rng, ["notes.txt", ".env.local", "scripts/seed.ts", "tmp/"], rng.randint(1, 3))
    ahead = rng.randint(1, 5)
    P = '<span class=g>dev@devbox</span>:<span class=b>~/webapp</span>$ '
    if lang == "ja":
        out = [P + "git status", f"ブランチ {br}", f"このブランチは 'origin/{br}' よりも{ahead}コミット進んでいます。",
               "  (use \"git push\" to publish your local commits)", "", "コミット予定の変更点:", "  (use \"git restore --staged &lt;file&gt;...\" to unstage)"]
        out += [f"<span class=g>\t変更されたファイル:   {f}</span>" for f in staged]
        out += ["", "コミット予定に追加されていない変更:", "  (use \"git add &lt;file&gt;...\" to update what will be committed)"]
        out += [f"<span class=r>\t変更されたファイル:   {f}</span>" for f in mod]
        out += ["", "追跡されていないファイル:", "  (use \"git add &lt;file&gt;...\" to include in what will be committed)"]
        out += [f"<span class=r>\t{f}</span>" for f in untr]
    else:
        out = [P + "git status", f"On branch {br}", f"Your branch is ahead of 'origin/{br}' by {ahead} commit{'s' if ahead > 1 else ''}.",
               "  (use \"git push\" to publish your local commits)", "", "Changes to be committed:", "  (use \"git restore --staged &lt;file&gt;...\" to unstage)"]
        out += [f"<span class=g>\tmodified:   {f}</span>" for f in staged]
        out += ["", "Changes not staged for commit:", "  (use \"git add &lt;file&gt;...\" to update what will be committed)"]
        out += [f"<span class=r>\tmodified:   {f}</span>" for f in mod]
        out += ["", "Untracked files:", "  (use \"git add &lt;file&gt;...\" to include in what will be committed)"]
        out += [f"<span class=r>\t{f}</span>" for f in untr]
    out.append(P + "<span class=w>▌</span>")
    page = _term_page("\n".join(out), "dev@devbox: ~/webapp")
    if lang == "ja":
        qas = [dict(q="現在のブランチ名は何ですか？", a=br, atype="str"),
               dict(q="追跡されていないファイルはいくつありますか？", a=str(len(untr)), atype="num"),
               dict(q="リモートより何コミット進んでいますか？", a=str(ahead), atype="num"),
               dict(q="コミット予定に追加されていない（ステージされていない）変更のあるファイルはいくつですか？", a=str(len(mod)), atype="num")]
        kl_q = "この git status の出力を説明し、次にすべき操作を提案してください。"
        kl_ref = (f"現在のブランチは {br} で、origin/{br} より {ahead} コミット進んでいます。ステージ済みの変更は " + "、".join(staged) +
                  f" で、まだステージされていない変更が " + "、".join(mod) + f" の {len(mod)} ファイルあります。追跡されていないファイルは " + "、".join(untr) +
                  f" です。必要な変更を git add でステージしてからコミットし、git push でリモートに反映するとよいでしょう。.env.local のような秘密情報を含むファイルはコミットしないよう注意してください。"
                  if ".env.local" in untr else
                  f"現在のブランチは {br} で、origin/{br} より {ahead} コミット進んでいます。ステージ済みの変更は " + "、".join(staged) +
                  f" で、まだステージされていない変更が " + "、".join(mod) + f" の {len(mod)} ファイルあります。追跡されていないファイルは " + "、".join(untr) +
                  " です。必要な変更を git add でステージしてからコミットし、git push でリモートに反映するとよいでしょう。")
    else:
        qas = [dict(q="What is the current branch name?", a=br, atype="str"),
               dict(q="How many untracked files are listed?", a=str(len(untr)), atype="num"),
               dict(q="By how many commits is the branch ahead of origin?", a=str(ahead), atype="num")]
        kl_q = "Explain this git status output and suggest the next steps."
        kl_ref = (f"You are on branch {br}, {ahead} commit(s) ahead of origin/{br}. Staged: {', '.join(staged)}. Not staged: {', '.join(mod)}. "
                  f"Untracked: {', '.join(untr)}. Stage what you need with git add, commit, then git push.")
    return Inst("terminal-git", "screenshot", lang, qas, kl_q, kl_ref, html=page, size=(1220, 40 + 24 * len(out) + 30))


def term_cli_ja(rng, lang):
    """日本語メッセージを出す CLI (ビルド・デプロイ) のエラー。en は rust の cargo エラー。"""
    P = '<span class=g>dev@devbox</span>:<span class=b>~/svc</span>$ '
    if lang == "ja":
        f = rng.choice(["config.yaml", "settings.toml", ".env.production", "deploy/values.yaml"])
        code = rng.choice([2, 3, 64, 78])
        env = rng.choice(["staging", "production", "dev"])
        step = rng.randint(2, 5)
        out = [P + f"./deploy.sh --env {env}", f"[1/{step + 1}] 依存パッケージを確認しています... <span class=g>完了</span>",
               f"[2/{step + 1}] Docker イメージをビルドしています... <span class=g>完了</span> (42.3 秒)",
               f"[3/{step + 1}] 設定ファイルを検証しています...", f"<span class=r>エラー: 設定ファイル '{f}' が見つかりません。</span>",
               "<span class=y>ヒント: --config オプションでパスを指定するか、テンプレートからコピーしてください。</span>",
               f"<span class=r>デプロイを中止しました（終了コード {code}）</span>", P + f"echo $?", str(code), P + "<span class=w>▌</span>"]
        qas = [dict(q="見つからなかったファイルの名前は何ですか？", a=f, atype="str"),
               dict(q="終了コードはいくつですか？", a=str(code), atype="num"),
               dict(q="デプロイ先の環境名（--env の値）は何ですか？", a=env, atype="str")]
        kl_q = "このターミナルで何が起きたか説明し、対処法を教えてください。"
        kl_ref = (f"./deploy.sh --env {env} でデプロイを実行したところ、依存パッケージの確認と Docker イメージのビルドは完了しましたが、"
                  f"3 番目の設定ファイルの検証で「設定ファイル '{f}' が見つかりません」というエラーになり、デプロイは終了コード {code} で中止されました。"
                  f"ヒントのとおり、--config オプションで正しいパスを指定するか、テンプレートから {f} をコピーして作成し、再実行してください。")
        name = "terminal-cli-ja"
    else:
        ecode, msg = rng.choice([("E0308", "mismatched types"), ("E0382", "borrow of moved value: `cfg`"), ("E0425", "cannot find value `count` in this scope"),
                                 ("E0599", "no method named `unwrap_or_default` found for enum `Result`")])
        fl = rng.choice(["src/main.rs", "src/config.rs", "src/server/mod.rs"])
        ln, col = rng.randint(10, 300), rng.randint(5, 40)
        out = [P + "cargo build", "   <span class=g>Compiling</span> svc v0.3.1 (/home/dev/svc)", f"<span class=r>error[{ecode}]</span><span class=w>: {H.escape(msg)}</span>",
               f"  <span class=b>--&gt;</span> {fl}:{ln}:{col}", "   <span class=b>|</span>", f"<span class=b>{ln:>3}</span> <span class=b>|</span>     let total = compute(items);",
               "   <span class=b>|</span>                 <span class=r>^^^^^^^^^^^^^^</span>", "",
               f"For more information about this error, try `rustc --explain {ecode}`.", "<span class=r>error</span>: could not compile `svc` (bin \"svc\") due to 1 previous error",
               P + "<span class=w>▌</span>"]
        qas = [dict(q="What is the Rust error code?", a=ecode, atype="str"),
               dict(q="Which file contains the error?", a=fl, atype="str"),
               dict(q="On which line is the error?", a=str(ln), atype="num")]
        kl_q = "Explain this build error and how to investigate it."
        kl_ref = (f"cargo build failed with error[{ecode}]: {msg} at {fl}:{ln}:{col}, on the line `let total = compute(items);`. "
                  f"Run `rustc --explain {ecode}` for details, then check the types and ownership of the values used on that line.")
        name = "terminal-cargo"
    page = _term_page("\n".join(out), "dev@devbox: ~/svc")
    return Inst(name, "screenshot", lang, qas, kl_q, kl_ref, html=page, size=(1220, 40 + 24 * len(out) + 30))


def term_pods(rng, lang):
    names = pick(rng, ["api-gateway", "auth-service", "billing-worker", "redis", "postgres", "frontend", "search-indexer", "mailer"], rng.randint(5, 7))
    bad = rng.randrange(len(names))
    bad_status = rng.choice(["CrashLoopBackOff", "ImagePullBackOff", "Error", "Pending"])
    rows = []
    for i, n in enumerate(names):
        h = "".join(rng.choice("abcdef0123456789") for _ in range(5))
        st = bad_status if i == bad else "Running"
        rs = rng.randint(4, 30) if i == bad and bad_status in ("CrashLoopBackOff", "Error") else rng.choice([0, 0, 0, 1])
        ready = "0/1" if i == bad else "1/1"
        age = f"{rng.randint(1, 30)}d" if i != bad else f"{rng.randint(2, 50)}m"
        rows.append((f"{n}-{h}", ready, st, rs, age))
    P = '<span class=g>ops@bastion</span>:<span class=b>~</span>$ '
    out = [P + "kubectl get pods -n prod", f"{'NAME':<28}{'READY':<8}{'STATUS':<20}{'RESTARTS':<10}AGE"]
    for r in rows:
        col = "r" if r[2] != "Running" else "d"
        out.append(f"{r[0]:<28}{r[1]:<8}<span class={col}>{r[2]:<20}</span>{r[3]:<10}{r[4]}")
    out.append(P + "<span class=w>▌</span>")
    page = _term_page("\n".join(out), "ops@bastion: ~")
    b = rows[bad]
    if lang == "ja":
        qas = [dict(q="Running 以外の状態になっている Pod の名前は何ですか？", a=b[0], aliases=[names[bad]], atype="str"),
               dict(q=f"Pod {b[0]} の STATUS は何ですか？", a=b[2], atype="str"),
               dict(q="表示されている Pod はいくつですか？", a=str(len(rows)), atype="num")]
        kl_q = "この kubectl の出力から、問題のある Pod と調べ方を説明してください。"
        kl_ref = (f"prod 名前空間の Pod が {len(rows)} 個表示されており、{b[0]} だけが {b[2]}（READY {b[1]}、再起動 {b[3]} 回）で、他はすべて Running です。"
                  f"kubectl describe pod {b[0]} -n prod でイベントを確認し、kubectl logs {b[0]} -n prod --previous で直前のログを見て原因を調べるとよいでしょう。")
    else:
        qas = [dict(q="Which pod is not in the Running state?", a=b[0], aliases=[names[bad]], atype="str"),
               dict(q=f"What is the STATUS of pod {b[0]}?", a=b[2], atype="str"),
               dict(q="How many pods are listed?", a=str(len(rows)), atype="num")]
        kl_q = "Which pod has a problem and how would you investigate it?"
        kl_ref = (f"{len(rows)} pods are listed in the prod namespace. {b[0]} is {b[2]} (READY {b[1]}, {b[3]} restarts) while the rest are Running. "
                  f"Use kubectl describe pod {b[0]} -n prod to see events and kubectl logs {b[0]} -n prod --previous for the last logs.")
    return Inst("terminal-kubectl", "screenshot", lang, qas, kl_q, kl_ref, html=page, size=(1000, 60 + 24 * len(out) + 30))


# ------------------------------------------------------------------------------------------------ web UI screenshots
WEB_CSS = BASE_CSS + """body{background:#f6f8fa;font-size:16px}
.nav{background:#24292f;color:#fff;padding:12px 24px;display:flex;gap:24px;align-items:center}
.nav b{font-size:18px} .badge{background:#e5534b;color:#fff;border-radius:10px;padding:1px 8px;font-size:13px;margin-left:4px}
.main{padding:24px 32px} .card{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:18px 22px;margin-bottom:16px}
.btn{display:inline-block;padding:8px 18px;border-radius:6px;border:1px solid #1f883d;background:#1f883d;color:#fff;margin-right:8px}
.btn.s{background:#f6f8fa;color:#24292f;border-color:#d0d7de} .btn.x{background:#eaeef2;color:#8c959f;border-color:#d0d7de}
.alert{background:#ffebe9;border:1px solid #ff8182;color:#82071e;padding:10px 14px;border-radius:6px;margin:10px 0}
.ok{background:#dafbe1;border:1px solid #4ac26b;color:#116329;padding:10px 14px;border-radius:6px;margin:10px 0}
input{border:1px solid #d0d7de;border-radius:6px;padding:7px 10px;width:340px;font-size:15px} label{display:block;margin:12px 0 4px;font-weight:bold}
table{border-collapse:collapse;width:100%} th,td{border-bottom:1px solid #d8dee4;padding:8px 10px;text-align:left} th{background:#f6f8fa}
.tg{display:inline-block;width:42px;height:22px;border-radius:11px;vertical-align:middle;position:relative}
.tg.on{background:#1f883d} .tg.off{background:#c9d1d9} .tg::after{content:"";position:absolute;top:3px;width:16px;height:16px;border-radius:8px;background:#fff}
.tg.on::after{right:3px} .tg.off::after{left:3px}
.pill{border-radius:12px;padding:2px 10px;font-size:13px} .p1{background:#dafbe1;color:#116329} .p2{background:#fff8c5;color:#7d4e00} .p3{background:#ffebe9;color:#82071e} .p4{background:#ddf4ff;color:#0969da}
"""


def _web(body, nav):
    return f"<!doctype html><html><head><meta charset=utf-8><style>{WEB_CSS}</style></head><body><div class=nav>{nav}</div><div class=main>{body}</div></body></html>"


def web_login(rng, lang):
    app = rng.choice(["Acme Console", "Kumo Admin", "Hikari Portal", "Nimbus Cloud"])
    code = f"E{rng.randint(1000, 9999)}"
    att = rng.randint(1, 4)
    if lang == "ja":
        msg = rng.choice(["メールアドレスまたはパスワードが正しくありません。", "アカウントが一時的にロックされています。", "二要素認証コードの有効期限が切れています。"])
        email = f"{rng.choice(list(JA_PEOPLE_ROMA.values()))}@example.jp"
        body = f"""<div class=card style="width:480px;margin:30px auto"><h2 style="margin-top:0">{app} にログイン</h2>
<div class=alert>{msg}（エラーコード: {code}）<br>残り {att} 回失敗するとロックされます。</div>
<label>メールアドレス</label><input value="{email}"><label>パスワード</label><input type=password value="password1">
<p><input type=checkbox style="width:auto"> ログイン状態を保持する</p><span class=btn>ログイン</span><span class=btn s>パスワードを忘れた場合</span></div>"""
        nav = f"<b>{app}</b><span>ヘルプ</span><span>お問い合わせ</span>"
        qas = [dict(q="表示されているエラーメッセージを書き写してください（エラーコードは除く）。", a=msg, atype="str"),
               dict(q="エラーコードは何ですか？", a=code, atype="str"),
               dict(q="あと何回失敗するとロックされますか？", a=str(att), atype="num"),
               dict(q="入力されているメールアドレスは何ですか？", a=email, atype="str")]
        kl_q = "この画面の状況を説明し、ユーザーが次にすべきことを教えてください。"
        kl_ref = (f"{app} のログイン画面で、メールアドレス欄に {email} が入力された状態でログインに失敗しています。赤い警告欄に「{msg}」（エラーコード {code}）と表示され、"
                  f"残り {att} 回失敗するとアカウントがロックされると書かれています。入力内容を確認し、パスワードがわからない場合は「パスワードを忘れた場合」から再設定してください。")
    else:
        msg = rng.choice(["Incorrect email or password.", "Your account has been temporarily locked.", "The two-factor code has expired."])
        email = f"{rng.choice(EN_PEOPLE).split()[0].lower()}@example.com"
        body = f"""<div class=card style="width:480px;margin:30px auto"><h2 style="margin-top:0">Sign in to {app}</h2>
<div class=alert>{msg} (error code: {code})<br>{att} attempt{'s' if att > 1 else ''} remaining before lockout.</div>
<label>Email</label><input value="{email}"><label>Password</label><input type=password value="password1">
<p><input type=checkbox style="width:auto"> Keep me signed in</p><span class=btn>Sign in</span><span class=btn s>Forgot password?</span></div>"""
        nav = f"<b>{app}</b><span>Docs</span><span>Support</span>"
        qas = [dict(q="What is the error code shown?", a=code, atype="str"),
               dict(q="How many attempts remain before lockout?", a=str(att), atype="num"),
               dict(q="What email address has been entered?", a=email, atype="str")]
        kl_q = "Describe what is happening on this screen and what the user should do."
        kl_ref = (f"This is the {app} sign-in page. Sign-in failed for {email} with the message \"{msg}\" (error code {code}), and {att} attempt(s) remain before lockout. "
                  f"The user should check the credentials or use \"Forgot password?\" to reset the password.")
    return Inst("web-login", "screenshot", lang, qas, kl_q, kl_ref, html=_web(body, nav), size=(1100, 620))


def web_orders(rng, lang):
    n = rng.randint(6, 9)
    if lang == "ja":
        stat = [("発送済み", "p1"), ("処理中", "p2"), ("キャンセル", "p3"), ("支払い待ち", "p4")]
        cust = [p for p in JA_PEOPLE]
    else:
        stat = [("Shipped", "p1"), ("Processing", "p2"), ("Cancelled", "p3"), ("Awaiting payment", "p4")]
        cust = EN_PEOPLE
    base = rng.randint(10200, 10800)
    rows = []
    for i in range(n):
        s = rng.choice(stat)
        rows.append((f"#{base + i}", rng.choice(cust), rng.randint(12, 980) * 100, s))
    kpi_sales = sum(r[2] for r in rows if r[3][0] not in ("キャンセル", "Cancelled"))
    j = rng.randrange(n)
    ncan = sum(r[3][0] in ("キャンセル", "Cancelled") for r in rows)
    if lang == "ja":
        tr = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{yen(c)}</td><td><span class='pill {s[1]}'>{s[0]}</span></td></tr>" for a, b, c, s in rows)
        body = f"""<div style="display:flex;gap:16px"><div class=card style="flex:1">本日の注文数<br><b style="font-size:28px">{n} 件</b></div>
<div class=card style="flex:1">売上（キャンセル除く）<br><b style="font-size:28px">{yen(kpi_sales)}</b></div></div>
<div class=card><h3 style="margin-top:0">最近の注文</h3><table><tr><th>注文番号</th><th>顧客</th><th>金額</th><th>ステータス</th></tr>{tr}</table></div>"""
        nav = "<b>ショップ管理</b><span>注文</span><span>商品</span><span>顧客</span><span>設定</span>"
        qas = [dict(q=f"注文番号 {rows[j][0]} のステータスは何ですか？", a=rows[j][3][0], atype="str"),
               dict(q="ステータスが「キャンセル」の注文は何件ですか？", a=str(ncan), atype="num"),
               dict(q=f"注文番号 {rows[j][0]} の顧客名は何ですか？", a=rows[j][1], aliases=[rows[j][1].replace(" ", "")], atype="str"),
               dict(q="売上（キャンセル除く）として表示されている金額はいくらですか？", a=str(kpi_sales), atype="num")]
        kl_q = "この管理画面に表示されている情報を要約してください。"
        kl_ref = (f"ショップ管理画面の注文一覧です。本日の注文数は {n} 件、キャンセルを除いた売上は {yen(kpi_sales)} です。最近の注文は次のとおりです。" +
                  "".join(f"\n- {a}: {b} 様、{yen(c)}、{s[0]}" for a, b, c, s in rows) + f"\nキャンセルは {ncan} 件です。")
    else:
        tr = "".join(f"<tr><td>{a}</td><td>{b}</td><td>${c / 100:,.2f}</td><td><span class='pill {s[1]}'>{s[0]}</span></td></tr>" for a, b, c, s in rows)
        body = f"""<div style="display:flex;gap:16px"><div class=card style="flex:1">Orders today<br><b style="font-size:28px">{n}</b></div>
<div class=card style="flex:1">Revenue (excl. cancelled)<br><b style="font-size:28px">${kpi_sales / 100:,.2f}</b></div></div>
<div class=card><h3 style="margin-top:0">Recent orders</h3><table><tr><th>Order</th><th>Customer</th><th>Amount</th><th>Status</th></tr>{tr}</table></div>"""
        nav = "<b>Shop Admin</b><span>Orders</span><span>Products</span><span>Customers</span><span>Settings</span>"
        qas = [dict(q=f"What is the status of order {rows[j][0]}?", a=rows[j][3][0], atype="str"),
               dict(q="How many orders are Cancelled?", a=str(ncan), atype="num"),
               dict(q=f"Who is the customer for order {rows[j][0]}?", a=rows[j][1], atype="str")]
        kl_q = "Summarize the information on this admin screen."
        kl_ref = (f"The shop admin dashboard shows {n} orders today and revenue of ${kpi_sales / 100:,.2f} excluding cancelled orders. Recent orders: " +
                  "; ".join(f"{a} {b} ${c / 100:,.2f} {s[0]}" for a, b, c, s in rows) + f". {ncan} order(s) are cancelled.")
    return Inst("web-orders", "screenshot", lang, qas, kl_q, kl_ref, html=_web(body, nav), size=(1200, 260 + 41 * n + 60))


def web_settings(rng, lang):
    if lang == "ja":
        names = ["メール通知", "プッシュ通知", "自動バックアップ", "二要素認証", "ダークモード", "利用状況の送信", "自動アップデート"]
    else:
        names = ["Email notifications", "Push notifications", "Automatic backups", "Two-factor authentication", "Dark mode", "Send usage data", "Auto updates"]
    sel = pick(rng, names, 5)
    st = [rng.random() < 0.5 for _ in sel]
    if all(st) or not any(st):
        st[0] = not st[0]
    unread = rng.randint(1, 15)
    j = rng.randrange(5)
    on_w, off_w = ("オン", "オフ") if lang == "ja" else ("On", "Off")
    rows = "".join(f"<div style='display:flex;justify-content:space-between;padding:12px 0;border-bottom:1px solid #eee'><span>{n}</span>"
                   f"<span><span class='tg {'on' if s else 'off'}'></span> {on_w if s else off_w}</span></div>" for n, s in zip(sel, st))
    if lang == "ja":
        nav = f"<b>Kumo Admin</b><span>ダッシュボード</span><span>通知<span class=badge>{unread}</span></span><span>設定</span>"
        body = f"<div class=card style='width:640px'><h3 style='margin-top:0'>アカウント設定</h3>{rows}<p><span class=btn>保存</span><span class='btn x'>変更を破棄</span></p></div>"
        qas = [dict(q=f"「{sel[j]}」はオンですか、オフですか？", a=on_w if st[j] else off_w, aliases=["有効", "ON"] if st[j] else ["無効", "OFF"], atype="str"),
               dict(q="オンになっている設定はいくつありますか？", a=str(sum(st)), atype="num"),
               dict(q="ナビゲーションバーの「通知」に表示されている未読数はいくつですか？", a=str(unread), atype="num")]
        kl_q = "この設定画面の状態を説明してください。"
        kl_ref = ("アカウント設定画面です。" + "、".join(f"{n}は{on_w if s else off_w}" for n, s in zip(sel, st)) +
                  f"になっています。オンの設定は {sum(st)} 個です。下に「保存」ボタンと、灰色で無効になっている「変更を破棄」ボタンがあります。"
                  f"上部のナビゲーションの「通知」には未読 {unread} 件のバッジが付いています。")
    else:
        nav = f"<b>Kumo Admin</b><span>Dashboard</span><span>Notifications<span class=badge>{unread}</span></span><span>Settings</span>"
        body = f"<div class=card style='width:640px'><h3 style='margin-top:0'>Account settings</h3>{rows}<p><span class=btn>Save</span><span class='btn x'>Discard changes</span></p></div>"
        qas = [dict(q=f"Is '{sel[j]}' on or off?", a=on_w if st[j] else off_w, aliases=["enabled"] if st[j] else ["disabled"], atype="str"),
               dict(q="How many settings are turned on?", a=str(sum(st)), atype="num"),
               dict(q="What unread count is shown on Notifications in the navigation bar?", a=str(unread), atype="num")]
        kl_q = "Describe the state of this settings screen."
        kl_ref = ("Account settings: " + ", ".join(f"{n} is {on_w if s else off_w}" for n, s in zip(sel, st)) +
                  f". {sum(st)} settings are on. There is a Save button and a greyed-out Discard changes button. Notifications shows {unread} unread.")
    return Inst("web-settings", "screenshot", lang, qas, kl_q, kl_ref, html=_web(body, nav), size=(1000, 520))


def web_pr(rng, lang):
    num = rng.randint(120, 4999)
    checks = ["lint", "unit-tests", "build", "e2e", "type-check", "security-scan"]
    sel = pick(rng, checks, 4)
    bad = rng.randrange(4)
    files = rng.randint(2, 25)
    add, dele = rng.randint(10, 900), rng.randint(0, 400)
    if lang == "ja":
        title = rng.choice(["ログインフォームの入力検証を追加", "日付パーサーのタイムゾーン処理を修正", "CSV エクスポート機能を実装", "API クライアントのリトライ処理を整理"])
        ck = "".join(f"<div style='padding:6px 0'>{'<span style=color:#cf222e>✕</span>' if i == bad else '<span style=color:#1f883d>✓</span>'} {c} — "
                     f"{'失敗しました' if i == bad else '成功しました'}</div>" for i, c in enumerate(sel))
        body = f"""<h2 style="margin:0">{title} <span style="color:#57606a">#{num}</span></h2>
<p><span class='pill p1' style='background:#1f883d;color:#fff'>Open</span>　yamada-dev が main に {files} ファイルの変更をマージしようとしています
<span style=color:#1a7f37>+{add}</span> <span style=color:#cf222e>−{dele}</span></p>
<div class=card><b>一部のチェックが失敗しました</b><br>1 件失敗、3 件成功{ck}</div><span class='btn x'>マージ（ブロック中）</span>"""
        nav = "<b>Git Hub-like</b><span>コード</span><span>Issues</span><span>プルリクエスト</span><span>Actions</span>"
        qas = [dict(q="失敗しているチェックの名前は何ですか？", a=sel[bad], atype="str"),
               dict(q="このプルリクエストの番号は何番ですか？", a=str(num), atype="num"),
               dict(q="このプルリクエストのタイトルは何ですか？", a=title, atype="str"),
               dict(q="変更されたファイル数はいくつですか？", a=str(files), atype="num")]
        kl_q = "このプルリクエスト画面の状況を説明し、マージするために何をすべきか教えてください。"
        kl_ref = (f"プルリクエスト #{num}「{title}」の画面です。yamada-dev が main に {files} ファイルの変更（+{add} / −{dele}）をマージしようとしています。"
                  f"チェックは 4 件中 {sel[bad]} の 1 件が失敗し、" + "、".join(c for i, c in enumerate(sel) if i != bad) +
                  f" は成功しています。失敗があるためマージボタンはブロックされています。{sel[bad]} のログを開いて原因を確認し、修正をプッシュしてチェックを通す必要があります。")
    else:
        title = rng.choice(["Add input validation to login form", "Fix timezone handling in date parser", "Implement CSV export", "Refactor API client retries"])
        ck = "".join(f"<div style='padding:6px 0'>{'<span style=color:#cf222e>✕</span>' if i == bad else '<span style=color:#1f883d>✓</span>'} {c} — "
                     f"{'Failing' if i == bad else 'Successful'}</div>" for i, c in enumerate(sel))
        body = f"""<h2 style="margin:0">{title} <span style="color:#57606a">#{num}</span></h2>
<p><span class='pill p1' style='background:#1f883d;color:#fff'>Open</span>　alex-dev wants to merge {files} changed files into main
<span style=color:#1a7f37>+{add}</span> <span style=color:#cf222e>−{dele}</span></p>
<div class=card><b>Some checks were not successful</b><br>1 failing, 3 successful{ck}</div><span class='btn x'>Merge (blocked)</span>"""
        nav = "<b>Git Hub-like</b><span>Code</span><span>Issues</span><span>Pull requests</span><span>Actions</span>"
        qas = [dict(q="Which check is failing?", a=sel[bad], atype="str"),
               dict(q="What is the pull request number?", a=str(num), atype="num"),
               dict(q="How many files are changed?", a=str(files), atype="num")]
        kl_q = "Explain the state of this pull request and what is needed to merge it."
        kl_ref = (f"Pull request #{num} \"{title}\" wants to merge {files} changed files (+{add}/−{dele}) into main. Of 4 checks, {sel[bad]} is failing and the rest pass, "
                  f"so merging is blocked. Open the {sel[bad]} logs, fix the cause, and push so the check passes.")
    return Inst("web-pr", "screenshot", lang, qas, kl_q, kl_ref, html=_web(body, nav), size=(1100, 520))


TEMPLATES = {
    "chart": [chart_bar, chart_bar, chart_line, chart_pie],
    "ocr-ja": [receipt, card, notice_vertical, manual_page],
    "ocr-en": [receipt],
    "code": [code_shot],
    "terminal": [term_pytest, term_traceback, term_git, term_cli_ja, term_pods],
    "web": [web_login, web_orders, web_settings, web_pr],
}


# ------------------------------------------------------------------------------------------------ rendering
class Renderer:
    def __init__(self, work):
        self.work = work
        os.makedirs(work, exist_ok=True)
        self.prof = tempfile.mkdtemp(prefix="ffprof-", dir=work)

    def close(self):
        shutil.rmtree(self.prof, ignore_errors=True)

    def render(self, inst: Inst, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if inst.fig is not None:
            import matplotlib.pyplot as plt
            inst.fig.savefig(path)
            plt.close(inst.fig)
            inst.fig = None
            return
        hp = os.path.join(self.work, "page.html")
        with open(hp, "w", encoding="utf-8") as f:
            f.write(inst.html)
        w, h = inst.size
        for _ in range(3):
            if os.path.exists(path):
                os.remove(path)
            subprocess.run(["firefox", "--headless", "--no-remote", "--profile", self.prof, f"--window-size={w},{h}",
                            "--screenshot", path, "file://" + hp], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            if os.path.exists(path) and os.path.getsize(path) > 1000:
                return
        raise RuntimeError(f"firefox screenshot failed: {path}")
