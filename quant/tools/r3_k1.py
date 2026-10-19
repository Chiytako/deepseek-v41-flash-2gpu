import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
exec(open(DSV41 + '/tools/r3_alloc.py').read().split('Wtrue=weights(0.0)')[0])
eps[1]=float(__import__("os").environ.get("E1","0.10"))  # assumption: K1 rel err ~32%
W=weights(0.2); Wo=weights(0.0); Wu=weights(1.0); Wm=weights(0.5)
b2=lambda Wx: sum(Wx[k]*eps[2] for k in Wx)
for b in (2.20,):
  for tiers in ([2,3],[1,2,3]):
    # start from all-K1 when tiers include 1
    cost,hist,avg,K=alloc(W,b,tiers)
    ev=lambda Wx: sum(Wx[k]*eps[K[k]] for k in K)/b2(Wx)
    print(f'b={b} tiers={tiers} obs={ev(Wo):.3f} half={ev(Wm):.3f} uniform={ev(Wu):.3f} hist={hist}')
