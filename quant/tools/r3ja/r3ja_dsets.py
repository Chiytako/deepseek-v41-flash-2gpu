"""日本語評価セットのトークン長を V4.1 tokenizer で測る (CPU)。usage: r3ja_dsets.py <data_dir>"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import sys, os, json
import pandas as pd
from tokenizers import Tokenizer
tok = Tokenizer.from_file(MODELS + '/DeepSeek-V4.1-Flash-1017/tokenizer.json')
D = sys.argv[1]
def n(t): return len(tok.encode(str(t), add_special_tokens=False).ids)
def rep(name, lens):
    s = pd.Series(lens); print(f"{name}: n={len(s)} mean={s.mean():.0f} p50={s.median():.0f} p95={s.quantile(.95):.0f} sum={s.sum()}")
df = pd.read_parquet(os.path.join(D, 'jcqa_val.parquet')); print(df.columns.tolist())
rep('JCQA q+5choices', [n(r['question'] + ' ' + ' '.join(str(r[f'choice{i}']) for i in range(5))) for _, r in df.iterrows()])
df = pd.read_parquet(os.path.join(D, 'jamcqa_test.parquet')); print(df.columns.tolist(), df.iloc[0].to_dict() if len(df) else '')
cols = [c for c in df.columns if c.startswith('choice')]
rep('JamC-QA q+choices', [n(str(r['question']) + ' ' + ' '.join(str(r[c]) for c in cols)) for _, r in df.iterrows()])
if 'category' in df.columns: print(df['category'].value_counts().to_dict())
df = pd.read_parquet(os.path.join(D, 'mmluprox_ja.parquet')); print(df.columns.tolist()[:20])
oc = [c for c in df.columns if c.startswith('option_')]
rep('MMLU-ProX-ja q+options', [n(str(r['question']) + ' ' + ' '.join(str(r[c]) for c in oc if r[c] is not None)) for _, r in df.iterrows()])
if 'category' in df.columns: print(df['category'].value_counts().to_dict())
df = pd.read_parquet(os.path.join(D, 'jhumaneval.parquet')); print(df.columns.tolist()); rep('JHumanEval prompt', [n(x) for x in df['prompt']])
df = pd.read_parquet(os.path.join(D, 'mgsm_ja.parquet')); print(df.columns.tolist()); rep('MGSM-ja q', [n(x) for x in df['question']])
df = pd.read_parquet(os.path.join(D, 'jsquad_val.parquet')); print(df.columns.tolist()); rep('JSQuAD ctx+q', [n(str(r['context']) + str(r['question'])) for _, r in df.iterrows()])
df = pd.read_csv(os.path.join(D, 'elyza100_test.csv')); print(df.columns.tolist()); rep('ELYZA input', [n(x) for x in df['input']]); rep('ELYZA output(ref)', [n(x) for x in df['output']])
L = [json.loads(l) for l in open(os.path.join(D, 'mifeval_ja.jsonl'))]; print(list(L[0].keys())); rep('M-IFEval-ja prompt', [n(x['prompt']) for x in L])
L = [json.loads(l) for l in open(os.path.join(D, 'jmtbench_q.jsonl'))]; print(list(L[0].keys()), {x['category'] for x in L}); rep('JMT turns', [sum(n(t) for t in x['turns']) for x in L])
