import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import json, heapq, sys
c={int(k):v for k,v in json.load(open(DSV41 + '/research/tmp/r3/agent_counts.json')).items()}
alpha={int(k):v for k,v in json.load(open(DSV41 + '/ext/sudoingx_dsv41/data/alpha_sensitivity.json'))['alpha'].items()}
eps={2:0.026,3:0.0042,4:0.00058,5:9.6e-5}
PB=35389440*1/8/1e9  # GB per expert per bit
def weights(mix):
    W={}
    for l in range(40):
        T=sum(c[l]); 
        for e in range(384): W[l,e]=alpha[l]*((1-mix)*c[l][e]/T+mix/384)
    return W
def alloc(W,b_avg,tiers,floor=None,layer_uniform=False):
    lo=tiers[0]; K={k:lo for k in W}
    if floor:
        for (l,e) in K:
            if l in floor: K[l,e]=max(K[l,e],floor[l])
    budget=len(W)*b_avg-sum(K.values())
    if layer_uniform:
        # greedy over layers as units
        Kl={l:max(lo,(floor or {}).get(l,lo)) for l in range(40)}
        Wl={l:sum(W[l,e] for e in range(384)) for l in range(40)}
        while True:
            best=None
            for l in range(40):
                i=tiers.index(Kl[l]) if Kl[l] in tiers else None
                if i is None or i+1>=len(tiers): continue
                nk=tiers[i+1]; db=(nk-Kl[l])*384
                if db>budget: continue
                g=Wl[l]*(eps[Kl[l]]-eps[nk])/db
                if best is None or g>best[0]: best=(g,l,nk,db)
            if not best: break
            _,l,nk,db=best; Kl[l]=nk; budget-=db
        K={(l,e):Kl[l] for (l,e) in W}
    else:
        h=[]
        def push(k):
            i=tiers.index(K[k])
            if i+1<len(tiers):
                nk=tiers[i+1]; db=nk-K[k]; heapq.heappush(h,(-W[k]*(eps[K[k]]-eps[nk])/db,k,nk,db))
        for k in K: push(k)
        while h and budget>0:
            g,k,nk,db=heapq.heappop(h)
            if K[k]==nk or db>budget: continue
            if tiers.index(nk)!=tiers.index(K[k])+1: continue
            K[k]=nk; budget-=db; push(k)
    cost=sum(W[k]*eps[K[k]] for k in K)
    hist={t:sum(1 for v in K.values() if v==t) for t in tiers}
    return cost,hist,sum(K.values())/len(K),K
Wtrue=weights(0.0)   # evaluate on observed agent distribution
for mix in (0.0,0.2):
  W=weights(mix)
  base=sum(Wtrue[k]*eps[2] for k in Wtrue)
  print(f'== plan weights mix={mix} (evaluated on observed agent routing); uniform K2 cost=1.000')
  for b in (2.10,2.15,2.20,2.25,2.50):
    for name,tiers,lu in (('layer K2/K3',[2,3],True),('expert K2/K3',[2,3],False),('expert K2/K3/K4',[2,3,4],False)):
        cost,hist,avg,K=alloc(W,b,tiers,layer_uniform=lu)
        ev=sum(Wtrue[k]*eps[K[k]] for k in K)/base
        gb=sum(K.values())*PB
        print(f'b={b:.2f} {name:16s} avg={avg:.3f} GB={gb:6.1f} cost={ev:.3f} hist={hist}')
# reference: uniform K3
print('uniform K3 cost', sum(Wtrue[k]*eps[3] for k in Wtrue)/base, 'K3 GB', 15360*3*PB)
