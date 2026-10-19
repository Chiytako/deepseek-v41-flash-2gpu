import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
exec(open(DSV41 + '/tools/r3_alloc.py').read().split('Wtrue=weights(0.0)')[0])
Wtrue=weights(0.0); Wuni=weights(1.0)
bt=sum(Wtrue[k]*eps[2] for k in Wtrue); bu=sum(Wuni[k]*eps[2] for k in Wuni)
# leave-half-out: plan on even layers? no: plan with half the experts' counts perturbed -> use shift: evaluate on uniform routing and on 50/50 mix
Wmid=weights(0.5); bm=sum(Wmid[k]*eps[2] for k in Wmid)
for mix in (0.0,0.2,0.5):
  W=weights(mix)
  for b in (2.20,2.25):
    for name,tiers,lu in (('layer K2/K3',[2,3],True),('expert K2/K3',[2,3],False),('expert K2/K3/K4',[2,3,4],False)):
      cost,hist,avg,K=alloc(W,b,tiers,layer_uniform=lu)
      ev=lambda Wx,bx: sum(Wx[k]*eps[K[k]] for k in K)/bx
      print(f'planmix={mix} b={b} {name:16s} eval_obs={ev(Wtrue,bt):.3f} eval_half={ev(Wmid,bm):.3f} eval_uniform={ev(Wuni,bu):.3f}')
