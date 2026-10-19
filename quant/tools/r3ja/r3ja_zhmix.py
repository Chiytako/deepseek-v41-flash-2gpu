"""日本語応答への簡体字混入の検出 (CPU)。簡体字専用字 = EUC-JP で表せず GB2312 で表せる CJK 字。
usage: r3ja_zhmix.py <x2_req_dir>   (GLM 出力の content 部分で基準値を出す)"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import sys, glob, json, re, collections, numpy as np
from tokenizers import Tokenizer
tok = Tokenizer.from_file(MODELS + '/DeepSeek-V4.1-Flash-1017/tokenizer.json')
TE = tok.token_to_id('</think>')
def zh_only(ch):
    o = ord(ch)
    if not (0x4e00 <= o <= 0x9fff): return False
    try: ch.encode('euc_jp'); return False
    except UnicodeEncodeError: pass
    try: ch.encode('gb2312'); return True
    except UnicodeEncodeError: return False
def is_ja(t):
    k = sum(1 for c in t if 0x3040 <= ord(c) <= 0x30ff)
    return k >= 10
S = collections.Counter(); ex = []
for mf in sorted(glob.glob(sys.argv[1] + '/*.meta.json')):
    metas = json.load(open(mf))
    if not metas: continue
    u = np.fromfile(mf[:-10] + '.u32', dtype=np.uint32); off = 0
    for m in metas:
        p, o = m['plen'], m['olen']; out = u[off + p: off + p + o]; off += p + o
        idx = np.where(out == TE)[0]
        ct = out[idx[0] + 1:] if len(idx) else out[:0]
        txt = tok.decode(ct.tolist(), skip_special_tokens=False)
        j = txt.find('<｜DSML｜'); txt = txt if j < 0 else txt[:j]
        txt = re.sub(r'```.*?```', '', txt, flags=re.S)
        if not is_ja(txt): continue
        z = [c for c in txt if zh_only(c)]
        S['ja_resp'] += 1; S['chars'] += len(txt); S['zh_chars'] += len(z)
        if z: S['resp_with_zh'] += 1
        if len(z) >= 3: S['resp_with_zh3'] += 1
        if z and len(ex) < 12: ex.append(''.join(z[:10]))
print(dict(S), 'rate(resp>=1)', S['resp_with_zh'] / max(1, S['ja_resp']), 'rate(>=3)', S['resp_with_zh3'] / max(1, S['ja_resp']))
print(ex)
