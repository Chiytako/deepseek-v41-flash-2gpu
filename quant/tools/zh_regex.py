"""ds41rt の既存機能 (response_format: {"type":"regex"}) だけで回答部分の簡体字専用字を禁じるための
正規表現を作る。コード変更なしで「マスクすると日本語の品質がどう変わるか」を評価するための道具。

  - 文字単位 (XGrammar の regex) なので、バイト片トークンの連結で簡体字を作る経路も塞がる
  - 思考モードでは ds41rt が sequence[any_tokens(</think> 以外), </think>, regex] に包む
    (rust/crates/ds41rt-api/src/native_v41/constraints.rs の response_constraint)
  - tools を付けた要求では or[regex, ツール呼び出し] になり「文章の後にツール呼び出し」が書けなくなるので、
    ツール無しの評価 (ELYZA-tasks-100、Japanese MT-Bench、MGSM ja など) 専用

usage: tools/.venv/bin/python -I tools/zh_regex.py
出力: work/zh/serve/answer_regex.txt、work/zh/serve/example_request.json
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import json

ROOT = DSV41
OUT = f'{ROOT}/work/zh/serve'

def zh_only(ch):
    o = ord(ch)
    if not (0x4e00 <= o <= 0x9fff or 0x3400 <= o <= 0x4dbf):
        return False
    for enc in ('euc_jp', 'euc_jis_2004', 'shift_jis_2004'):
        try:
            ch.encode(enc); return False
        except UnicodeEncodeError:
            pass
    try:
        ch.encode('gb2312'); return True
    except UnicodeEncodeError:
        return False

cps = [o for o in list(range(0x3400, 0x4dc0)) + list(range(0x4e00, 0xa000)) if zh_only(chr(o))]
ranges = []
for o in cps:
    if ranges and ranges[-1][1] == o - 1:
        ranges[-1][1] = o
    else:
        ranges.append([o, o])
cls = ''.join(chr(a) if a == b else chr(a) + chr(b) if b == a + 1 else f'{chr(a)}-{chr(b)}' for a, b in ranges)
pattern = f'[^{cls}]*'
open(f'{OUT}/answer_regex.txt', 'w').write(pattern)
body = {
    "model": "deepseek-ai/DeepSeek-V4.1-Flash",
    "messages": [{"role": "user", "content": "日本の四季について、それぞれの特徴を3文で説明してください。"}],
    "temperature": 1.0, "top_p": 1.0, "max_tokens": 1024,
    "response_format": {"type": "regex", "regex": pattern},
}
json.dump(body, open(f'{OUT}/example_request.json', 'w'), ensure_ascii=False)
print(f'chars {len(cps)}, ranges {len(ranges)}, pattern {len(pattern)} chars / {len(pattern.encode())} bytes')
