import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
exec(open(DSV41 + '/tools/r3_alloc.py').read().split('Wtrue=weights(0.0)')[0])
for b in (2.15,2.20,2.25):
  cost,hist,avg,K=alloc(weights(0.2),b,[2,3])
  for B in (6,64,96):
    num=den=t=0
    for l in range(40):
      T=sum(c[l])
      for e in range(384):
        p=6*c[l][e]/T; q=1-(1-p)**B if p<1 else 1
        num+=q*K[l,e]; den+=q*2; t+=q
    print(f'b={b} B={B} touched/layer={t/40:.0f} expert-bytes ratio vs K2={num/den:.3f}')
