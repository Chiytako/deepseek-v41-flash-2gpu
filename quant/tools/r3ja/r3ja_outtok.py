import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import json, glob, re, collections, numpy as np
from tokenizers import Tokenizer
tok=Tokenizer.from_file(MODELS + '/DeepSeek-V4.1-Flash-1017/tokenizer.json')
te=tok.token_to_id('</think>'); print('think_end id',te, tok.token_to_id('<think>'))
REQ=DSV41 + '/work/x2/req'
def lang(t):
    k=c=n=0
    for ch in t:
        if ch.isspace(): continue
        n+=1;o=ord(ch)
        if 0x3040<=o<=0x30ff or 0xff66<=o<=0xff9f: k+=1
        elif 0x4e00<=o<=0x9fff: c+=1
    if n==0: return 'empty'
    if k>=3 and (k+c)/n>=0.10: return 'ja'
    if c/n>=0.2: return 'zh'
    return 'en'
S=collections.Counter()
for mf in sorted(glob.glob(REQ+'/*.meta.json')):
    metas=json.load(open(mf))
    if not metas: continue
    u=np.fromfile(mf[:-10]+'.u32',dtype=np.uint32); off=0
    for m in metas:
        p,o=m['plen'],m['olen']; out=u[off+p:off+p+o]; off+=p+o
        idx=np.where(out==te)[0]
        if len(idx): rs,ct=out[:idx[0]],out[idx[0]+1:]
        else: rs,ct=out,out[:0]
        S[('reason',lang(tok.decode(rs.tolist())))]+=len(rs)
        # split content into text vs DSML tool part
        txt=tok.decode(ct.tolist(),skip_special_tokens=False)
        j=txt.find('<｜DSML｜')
        body=txt if j<0 else txt[:j]
        nb=len(tok.encode(body,add_special_tokens=False).ids) if body else 0
        S[('content',lang(body))]+=nb; S[('toolcall','-')]+=max(0,len(ct)-nb)
T=sum(S.values())
for k,v in sorted(S.items()): print(k, v, round(100*v/T,1))
