import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
exec(open(DSV41 + '/tools/r3_alloc.py').read().split('Wtrue=weights(0.0)')[0])
eps[4]=0.0007  # K2 + K2 residual ~ (16.2%)^2 rel error  [assumption]
W=weights(0.0); base=sum(W[k]*eps[2] for k in W)
for name,tiers,b in (('dup K3 pool 1280 slots (17GB)',[2,3],2+1280/15360),('delta K2res pool 1920 (17GB)',[2,4],2+2*1920/15360),('static K2/K3 in-place 3840 (17GB)',[2,3],2.25)):
    cost,hist,avg,K=alloc(W,b,tiers)
    print(f'{name:36s} in-sample cost={cost/base:.3f} hist={hist}')
