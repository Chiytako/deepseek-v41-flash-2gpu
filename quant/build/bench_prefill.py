import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import json,time,urllib.request,os,random
URL=os.environ.get("URL","http://localhost:8000/v1/chat/completions"); MODEL=os.environ.get("MODEL","deepseek-ai/DeepSeek-V4.1-Flash")
base=open(DSV41 + '/research/arch_spec.md').read()+open(DSV41 + '/research/r3_japanese.md').read()+open(DSV41 + '/research/PLAN.md').read()
for target_chars in (16000, 64000, 120000):
    doc=(base*4)[:target_chars]
    salt=f"[{random.random()}] "
    b={"model":MODEL,"messages":[{"role":"user","content":salt+doc+"\n\nこの文書の題名を一言で。"}],"max_tokens":1,"temperature":0}
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(URL,json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=1800)); dt=time.time()-t
    p=r["usage"]["prompt_tokens"]; print(f"prompt {p} tok: {dt:.2f}s -> {p/dt:.0f} tok/s",flush=True)
