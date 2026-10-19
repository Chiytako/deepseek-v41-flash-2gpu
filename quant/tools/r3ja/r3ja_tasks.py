import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import orjson, collections
exec(open(DSV41 + '/tools/r3ja/r3ja_join.py').read().split("def lang")[0].split("REQ=")[0])  # rows dict
src=open(DSV41 + '/tools/r3ja/r3ja_join.py').read(); ns={}
exec(src[src.index('def cat'):src.index('def lang')], ns)
cat=ns['cat']
T=collections.defaultdict(set); R=collections.Counter(); J=collections.Counter(); E=collections.Counter()
for r in rows.values():
    c=cat(r); T[c].add(r['task_h']); R[c]+=1
    if r['ut_lang']=='ja': J[c]+=1
    E[(c, str(r['effort']), str(r['thinking']))]+=1
nt=sum(len(v) for v in T.values())
for c in sorted(T): print(c, 'tasks',len(T[c]), round(100*len(T[c])/nt,1),'%')
print('total tasks',nt)
for k,v in sorted(E.items(), key=lambda x:-x[1])[:15]: print(k,v)
# B_assistant subtopics by keyword
kw={'計算/見積':['円','何日','いくら','計算','見積','%','年利'],'説明/学習':['説明','教えて','とは','何をしている','理解度','勉強'],
    '翻訳':['英語に','翻訳','和訳','英訳'],'計画/リスト':['スケジュール','チェックリスト','計画','予定'],'ファイル/ログ解析':['log','.json','uploads/','exp/','分析','論文','pdf'],
    'リポジトリ/開発':['リポジトリ','GitHub','github','ビルド','実装','コード'],'比較/意見':['比較','どちら','思いますか','違い']}
B=[r for r in rows.values() if cat(r)=='B_assistant' and r['ut_lang']=='ja']
seen=set(); K=collections.Counter()
for r in B:
    if r['task_h'] in seen: continue
    seen.add(r['task_h']); hit=False
    for k,ws in kw.items():
        if any(w in r['fu'][:200] for w in ws): K[k]+=1; hit=True
    if not hit: K['その他']+=1
print('B_assistant ja unique tasks',len(seen),K.most_common())
