import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
exec(open(DSV41 + '/tools/r3_alloc.py').read().split('Wtrue=weights(0.0)')[0])
for b in (2.15,2.20,2.25):
  cost,hist,avg,K=alloc(weights(0.2),b,[2,3])
  dec=sum(sum(c[l][e]/sum(c[l])*K[l,e] for e in range(384)) for l in range(40))/40
  per=[sum(1 for e in range(384) if K[l,e]==3) for l in range(40)]
  print(f'b={b} decode-weighted K={dec:.3f}  K3 per layer: L0-9 {per[:10]} L10-19 {per[10:20]} L20-29 {per[20:30]} L30-39 {per[30:]}')
