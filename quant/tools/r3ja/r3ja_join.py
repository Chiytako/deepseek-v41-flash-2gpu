import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import orjson, collections, glob, os, re, numpy as np, json
from tokenizers import Tokenizer
tok=Tokenizer.from_file(MODELS + '/DeepSeek-V4.1-Flash-1017/tokenizer.json')
rows={}
for l in open(DSV41 + '/work/r3ja/rows.jsonl','rb'):
    r=orjson.loads(l); rows[(r['file'],r['ts'])]=r
REQ=DSV41 + '/work/x2/req'
def cat(r, _rules=[]):
    """要求の区分を返す。規則は R3JA_RULES (JSON の配列) で与える。公開版には利用者のクライアント固有の規則を入れていない。
    規則の例: [{"sys_h": ["<system prompt 先頭 400 字の md5 の先頭 10 桁>"], "fu_prefix": "<最初の発話の書き出し>", "cat": "A1_agent_review"},
              {"sys_h": ["d41d8cd98f"], "fu_contains": ["JSON"], "cat": "C2_plan_json"}]
    上から順に見て、書かれた条件 (sys_h / fu_prefix / fu_contains) をすべて満たした最初の規則の cat を返す。どれにも当たらなければ
    system prompt が空 (d41d8cd98f) なら C3_nosys_other、それ以外は H_other。"""
    if not _rules:
        import json as _json, os as _os2
        _p = _os2.environ.get("R3JA_RULES")
        _rules.append(_json.load(open(_p)) if _p else [])
    s=r['sys_h']; fu=r['fu']
    for rule in _rules[0]:
        if 'sys_h' in rule and s not in rule['sys_h']: continue
        if 'fu_prefix' in rule and not fu.startswith(rule['fu_prefix']): continue
        if 'fu_contains' in rule and not any(w in fu for w in rule['fu_contains']): continue
        return rule['cat']
    if s=='d41d8cd98f': return 'C3_nosys_other'
    return 'H_other'
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
agg=collections.defaultdict(lambda: collections.Counter())
olang=collections.defaultdict(collections.Counter)
cpt=collections.Counter()
samples=collections.defaultdict(list)
for mf in sorted(glob.glob(REQ+'/*.meta.json')):
    metas=json.load(open(mf))
    if not metas: continue
    u=np.fromfile(mf[:-10]+'.u32',dtype=np.uint32)
    off=0
    for m in metas:
        r=rows.get((m['file'],m['ts']))
        p,o=m['plen'],m['olen']
        out=u[off+p:off+p+o].tolist(); off+=p+o
        if r is None: c='Z_unmatched'
        else: c=cat(r)
        a=agg[c]; a['req']+=1; a['prompt']+=p; a['out']+=o
        a['glm_new']+=max(0,m['glm_prompt']-m['glm_cached'])
        if r is not None and r['ut_lang']=='ja': a['req_ja']+=1
        txt=tok.decode(out,skip_special_tokens=False)
        if '</think>' in txt: rs,ct=txt.split('</think>',1)
        else: rs,ct=txt,''
        ct=re.sub(r'<｜[^｜]*｜>','',ct)
        # strip DSML tool calls
        ct2=re.sub(r'<｜DSML｜.*','',ct,flags=re.S)
        lr,lc=lang(rs),lang(ct2)
        olang[c][('reason',lr)]+=1; olang[c][('content',lc)]+=1
        if lc=='ja' and len(ct2)>200:
            ids=tok.encode(ct2,add_special_tokens=False).ids
            cpt['ja_chars']+=len(ct2); cpt['ja_tok']+=len(ids)
        if lr=='en' and len(rs)>500:
            ids=tok.encode(rs,add_special_tokens=False).ids
            cpt['en_chars']+=len(rs); cpt['en_tok']+=len(ids)
        if len(samples[c])<3 and lc=='ja' and len(ct2)>80: samples[c].append(ct2[:200])
tot=collections.Counter()
for c,a in agg.items(): tot.update(a)
print('cat req req_ja% prompt(M) out(M) glm_newprefill(M)  share_req share_out share_new')
for c in sorted(agg):
    a=agg[c]
    print(c, a['req'], round(100*a['req_ja']/a['req'],1), round(a['prompt']/1e6,1), round(a['out']/1e6,2), round(a['glm_new']/1e6,2),
      round(100*a['req']/tot['req'],1), round(100*a['out']/tot['out'],1), round(100*a['glm_new']/tot['glm_new'],1))
print('TOTAL',dict(tot))
for c in sorted(olang): print(c, sorted(olang[c].items()))
print(cpt, 'ja chars/token', cpt['ja_chars']/max(1,cpt['ja_tok']), 'en chars/token', cpt['en_chars']/max(1,cpt['en_tok']))
for c,s in samples.items(): print(c, [x[:120] for x in s])
