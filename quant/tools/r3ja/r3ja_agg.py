import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import orjson, collections, sys
rows=[orjson.loads(l) for l in open(DSV41 + '/work/r3ja/rows.jsonl','rb')]
N=len(rows); print('requests',N)
C=collections.Counter
for k in ['lu_lang','fu_lang','ut_lang','sys_lang','tool_lang','asst_lang']:
    print(k, C(r[k] for r in rows).most_common())
# by task
tasks=collections.defaultdict(list)
for r in rows: tasks[r['task_h']].append(r)
print('tasks',len(tasks))
tl=C(v[0]['fu_lang'] for v in tasks.values()); print('task lang',tl.most_common())
# ja via any user text
print('ut_lang by tools', C((r['ut_lang'], r['ntools']>0) for r in rows).most_common())
# system prompt clusters
sc=C(r['sys_h'] for r in rows)
for h,n in sc.most_common(15):
    ex=next(r for r in rows if r['sys_h']==h)
    jl=C(r['ut_lang'] for r in rows if r['sys_h']==h)
    print(n, h, ex['ntools'], ex['tnames'][:8], jl.most_common(3), repr(ex['sys_head'][:110]))
print('thinking',C(str(r['thinking']) for r in rows).most_common()); print('effort',C(str(r['effort']) for r in rows).most_common())
print('temp',C(str(r['temp']) for r in rows).most_common(6)); print('model',C(str(r['model']) for r in rows).most_common(6))
print('img', sum(1 for r in rows if r['nimg']>0))
# char shares
for lang in ['ja','en','zh']:
    s=[r for r in rows if r['ut_lang']==lang]
    print(lang, len(s), 'tools%', round(100*sum(r['ntools']>0 for r in s)/max(1,len(s)),1), 'mean nmsg', round(sum(r['nmsg'] for r in s)/max(1,len(s)),1))
