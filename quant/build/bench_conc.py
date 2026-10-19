import json,time,urllib.request,sys,concurrent.futures as cf
import os
URL=os.environ.get("URL","http://localhost:8000/v1/chat/completions"); MODEL=os.environ.get("MODEL","deepseek-ai/DeepSeek-V4.1-Flash")
PROMPTS=["日本の四季について、それぞれの特徴を詳しく説明してください。","Pythonで簡単なTODOアプリのCLIを書いてください。","量子コンピュータの仕組みを高校生向けに説明してください。",
 "Explain how a hash map works and its time complexity.","Rustで並行に URL をフェッチする例を書いて、解説してください。","日本の戦国時代の主要な大名を5人挙げて、それぞれの業績を説明してください。",
 "Write a SQL query to find the top 3 customers by revenue per month, and explain it.","機械学習の過学習とその対策を具体例とともに説明してください。"]
def one(i,max_tokens):
    body={"model":MODEL,"messages":[{"role":"user","content":PROMPTS[i%len(PROMPTS)]+f" (#{i})"}],"max_tokens":max_tokens,"temperature":0.6,"top_p":0.95}
    t=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=1800))
    return time.time()-t, r["usage"]["completion_tokens"]
for c in map(int,sys.argv[1:]):
    n=max(c,4) if c<4 else c
    t=time.time()
    with cf.ThreadPoolExecutor(c) as ex: res=list(ex.map(lambda i: one(i,1024), range(n)))
    wall=time.time()-t; tok=sum(r[1] for r in res)
    per=[r[1]/r[0] for r in res]
    print(f"c={c}: {n} req, {tok} tok in {wall:.1f}s -> total {tok/wall:.0f} tok/s, per-stream mean {sum(per)/len(per):.1f} tok/s (min {min(per):.1f})",flush=True)
