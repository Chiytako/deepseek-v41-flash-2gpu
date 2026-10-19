import json,sys
c={int(k):v for k,v in json.load(open(sys.argv[1])).items()}
def stats(v):
    s=sorted(v,reverse=True); T=sum(s); n=len(s)
    top=lambda f: sum(s[:int(n*f)])/T
    bot30=sum(s[n-int(n*0.3):])/T
    acc=0;k90=0
    for i,x in enumerate(s):
        acc+=x
        if acc>=0.9*T: k90=i+1;break
    asc=sorted(v); g=sum((2*(i+1)-n-1)*x for i,x in enumerate(asc))/(n*T)
    zero=sum(1 for x in v if x==0)
    return dict(top10=top(.1),top25=top(.25),top50=top(.5),bot30=bot30,k90=k90,gini=g,zero=zero,maxmean=s[0]/(T/n))
rows=[]
for L in sorted(c):
    st=stats(c[L]); rows.append(st)
    print(L, ' '.join(f'{k}={v:.3f}' if isinstance(v,float) else f'{k}={v}' for k,v in st.items()))
import statistics as S
print('MEAN', ' '.join(f'{k}={S.mean(r[k] for r in rows):.3f}' for k in rows[0]))
print('MIN ', ' '.join(f'{k}={min(r[k] for r in rows):.3f}' for k in rows[0]))
print('MAX ', ' '.join(f'{k}={max(r[k] for r in rows):.3f}' for k in rows[0]))
# global (layer-normalized) coverage for tier fractions
for f in (0.1,0.2,0.25,0.3,0.4,0.5):
    cov=S.mean(stats(c[L])['top10'] if False else sum(sorted(c[L],reverse=True)[:int(384*f)])/sum(c[L]) for L in c)
    print(f'top{int(f*100)}% per-layer routing share mean={cov:.3f}')
