import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
from tokenizers import Tokenizer, Regex, normalizers
import collections
tok=Tokenizer.from_file(MODELS + '/DeepSeek-V4.1-Flash-1017/tokenizer.json')
s=""
nm=normalizers.Sequence([normalizers.NFKC(),normalizers.NFD(),normalizers.StripAccents(),normalizers.Lowercase(),
  normalizers.Replace(Regex(r"[ \t\r\n]+")," "),normalizers.Replace(Regex(r"^ $"),s),normalizers.Strip(),normalizers.Replace(s," ")])
for w in ["が","か","ぱ","ば","は","ガ","カ","ｶﾞ","デ","テ","ご","こ","ぷ"]:
    print(w, repr(nm.normalize_str(w)))
V=tok.get_vocab_size(); print('vocab',V)
def isja(t):
    return any(0x3040<=ord(c)<=0x30ff or 0x4e00<=ord(c)<=0x9fff for c in t)
keys=collections.defaultdict(list); nja=0
for i in range(V):
    t=tok.decode([i],skip_special_tokens=False)
    if '�' in t: k=tok.id_to_token(i)
    else:
        k=nm.normalize_str(t) or t
    keys[k].append(i)
    if isja(t): nja+=1
print('compressed', len(keys), 'ja tokens', nja)
# collisions among ja tokens
col=0; ex=[]
for k,ids in keys.items():
    ts=[tok.decode([i]) for i in ids]
    if any(isja(t) for t in ts) and len(set(x.strip() for x in ts))>1:
        col+=len(ids)
        if len(ex)<25: ex.append(ts)
print('ja tokens in collapsed groups with different surface forms', col)
for e in ex: print(e)
# chars/token for ja whole-word tokens: length distribution
L=collections.Counter(len(tok.decode([i]).strip()) for i in range(V) if isja(tok.decode([i])))
print('ja token length dist', sorted(L.items())[:12])
