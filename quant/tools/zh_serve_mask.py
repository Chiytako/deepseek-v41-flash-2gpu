"""配信時の中国語抑制マスクの見積もり (CPU のみ、r5_serving_suppress.md 用)。

やること
  1. V4.1 語彙 129,280 を、復号したバイト列で分類する。
     - Z1: 簡体字専用字 (GB2312 にあり、EUC-JP / EUC-JIS-2004 / Shift_JIS-2004 のどれでも表せない CJK 字) を 1 字以上含む完全 UTF-8 トークン
     - P : 不完全な UTF-8 (バイト片) のトークン。Z1 の字をバイト片の連結で作れる「漏れ道」になる
  2. 簡体字専用字それぞれを単独で符号化し、1 トークンで出るか (= Z1 マスクで止まる) を調べる。
  3. 中国語コーパス (zh.wikipedia 抜粋、alpaca-zh) で、簡体字専用字の出現のうち Z1 トークン経由の割合を出す。
  4. 日本語コーパス (wiki_ja、aozora、oasst2_ja、dolly_ja) と GLM 実トラフィック (V4.1 トークン化済み) で、
     Z1 トークンの出現率 (誤抑制の上限) を出す。実トラフィックは prompt 側 (文脈に中国語があるか) と出力側を分ける。
  5. ds41rt の文法マスクと同じ形式 (bit=1 が許可、ceil(V/32) 語の uint32) で Z1 の拒否マスクを書き出す。

usage: tools/.venv/bin/python -I tools/zh_serve_mask.py [--req-files N]
出力: work/zh/serve/{summary.json, z1_ids.json, allow_mask_z1.u32}
"""
import os as _os  # publish: host paths -> env (DSV41 = workspace root, MODELS = model dir)
DSV41 = _os.environ.get("DSV41", _os.getcwd())
MODELS = _os.environ.get("MODELS", _os.path.join(DSV41, "models"))
GLM_REQLOG = _os.environ.get("GLM_REQLOG", "reqlog")  # private request log (not published; bring your own)
import sys, os, glob, json, collections, argparse
import numpy as np
from tokenizers import Tokenizer

ROOT = DSV41
TOK = MODELS + '/DeepSeek-V4.1-Flash-1017/tokenizer.json'
OUT = f'{ROOT}/work/zh/serve'
V = 129280
THINK_END = 128822
DSML = 128825  # '｜DSML｜' (ツール呼び出しの開始 '<｜DSML｜ calls>' の 2 トークン目)

ap = argparse.ArgumentParser()
ap.add_argument('--req-files', type=int, default=400)
ap.add_argument('--ja-chars', type=int, default=6_000_000)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

tok = Tokenizer.from_file(TOK)

# --- byte-level BPE: トークン文字列 -> 生バイト ---------------------------------
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
U2B = bytes_to_unicode()

def raw_bytes(i):
    s = tok.id_to_token(i)
    if s is None:
        return None
    try:
        return bytes(U2B[c] for c in s)
    except KeyError:          # special / added token (平文)
        return s.encode('utf-8')

def zh_only(ch):
    o = ord(ch)
    if not (0x4e00 <= o <= 0x9fff or 0x3400 <= o <= 0x4dbf):
        return False
    for enc in ('euc_jp', 'euc_jis_2004', 'shift_jis_2004'):
        try:
            ch.encode(enc); return False
        except UnicodeEncodeError:
            pass
    try:
        ch.encode('gb2312'); return True
    except UnicodeEncodeError:
        return False

def is_kana(ch):
    return 0x3040 <= ord(ch) <= 0x30ff

def is_cjk(ch):
    o = ord(ch)
    return 0x4e00 <= o <= 0x9fff or 0x3400 <= o <= 0x4dbf

# --- 1. 語彙の分類 ---------------------------------------------------------------
z1, partial, cjk_tok, kana_tok = [], [], 0, 0
z1_chars_in_vocab = set()
for i in range(V):
    b = raw_bytes(i)
    if b is None:
        continue
    try:
        s = b.decode('utf-8')
    except UnicodeDecodeError:
        partial.append(i); continue
    if any(is_cjk(c) for c in s): cjk_tok += 1
    if any(is_kana(c) for c in s): kana_tok += 1
    zc = [c for c in s if zh_only(c)]
    if zc:
        z1.append(i); z1_chars_in_vocab.update(zc)
z1set = set(z1)
z1_with_kana = [i for i in z1 if any(is_kana(c) for c in raw_bytes(i).decode('utf-8'))]

# --- 2. 簡体字専用字の単独符号化 ----------------------------------------------------
zh_chars = []
for o in list(range(0x4e00, 0xa000)):
    c = chr(o)
    if zh_only(c): zh_chars.append(c)
single, multi = 0, 0
multi_examples = []
for c in zh_chars:
    ids = tok.encode(c, add_special_tokens=False).ids
    if len(ids) == 1: single += 1
    else:
        multi += 1
        if len(multi_examples) < 15: multi_examples.append(c)

# --- 3. 中国語コーパスでの捕捉率 ----------------------------------------------------
def rows(path):
    d = json.load(open(path))
    for r in d['rows']:
        r = r['row']
        yield '\n'.join(str(r.get(k, '')) for k in ('text', 'instruction', 'input', 'output') if r.get(k))
zh_texts = []
for p in sorted(glob.glob(f'{ROOT}/work/zh/data/*.json')):
    zh_texts.extend(rows(p))
zh_stat = collections.Counter(); zh_count = collections.Counter(); zh_sent_ids = []
for t in zh_texts:
    enc = tok.encode(t, add_special_tokens=False)
    zh_count.update(enc.ids)
    for sent in t.replace('！', '。').replace('？', '。').split('。'):
        if sum(1 for c in sent if is_cjk(c)) >= 6:
            zh_sent_ids.append(tok.encode(sent, add_special_tokens=False).ids)
    for i, (a, bnd) in zip(enc.ids, enc.offsets):
        frag = t[a:bnd]
        n = sum(1 for c in frag if zh_only(c))
        if n == 0: continue
        zh_stat['zh_char_tokens'] += 1
        if i in z1set: zh_stat['in_z1'] += 1
        else: zh_stat['not_z1'] += 1
    zh_stat['tokens'] += len(enc.ids)
    zh_stat['z1_tokens'] += sum(1 for i in enc.ids if i in z1set)
    zh_stat['zh_chars'] += sum(1 for c in t if zh_only(c))
    zh_stat['cjk_chars'] += sum(1 for c in t if is_cjk(c))

# --- 4a. 日本語コーパスでの Z1 出現 ---------------------------------------------------
ja_stat = collections.Counter(); ja_examples = collections.Counter(); ja_count = collections.Counter()
budget = args.ja_chars
for p in ['wiki_ja', 'oasst2_ja', 'dolly_ja', 'aozora']:
    path = f'{ROOT}/work/calib/src/{p}.jsonl'
    if not os.path.exists(path): continue
    used = 0
    for line in open(path):
        try: r = json.loads(line)
        except Exception: continue
        t = r.get('text') or '\n'.join(str(v) for v in r.values() if isinstance(v, str))
        if not t and isinstance(r.get('conversations'), list):
            t = '\n'.join(str(c.get('content', '')) for c in r['conversations'] if isinstance(c, dict))
        ids = tok.encode(t, add_special_tokens=False).ids
        ja_count.update(ids)
        ja_stat[f'{p}_tokens'] += len(ids)
        hits = [i for i in ids if i in z1set]
        ja_stat[f'{p}_z1'] += len(hits)
        for i in hits[:3]: ja_examples[tok.decode([i])] += 1
        used += len(t)
        if used > budget // 4: break
    ja_stat[f'{p}_chars'] = used

# --- 4a'. T2 (候補): 簡体字専用字は無いが中国語の語として使われ、日本語コーパスに 1 度も出ないトークン ---
def cjk_only(i):
    b = raw_bytes(i)
    try: s_ = b.decode('utf-8').strip()
    except Exception: return False
    return len(s_) >= 2 and all(is_cjk(c) for c in s_)
t2 = [i for i in range(V) if i not in z1set and zh_count[i] >= 3 and ja_count[i] == 0 and cjk_only(i)]
t2set = set(t2)
t2_examples = sorted(t2, key=lambda i: -zh_count[i])[:40]
# 中国語の文 (CJK 6 字以上) のうち Z1 / Z1∪T2 を含む割合と、最初に当たるまでのトークン数
sent = collections.Counter(); first_pos = {'z1': [], 'z1t2': []}
for ids in zh_sent_ids:
    sent['n'] += 1
    for key, S_ in (('z1', z1set), ('z1t2', z1set | t2set)):
        hit = next((k for k, i in enumerate(ids) if i in S_), None)
        if hit is not None:
            sent[key] += 1; first_pos[key].append(hit)
sent_cov = {'sentences': sent['n'], 'with_z1': sent['z1'] / max(1, sent['n']), 'with_z1_or_t2': sent['z1t2'] / max(1, sent['n']),
            'median_first_hit_tok_z1': float(np.median(first_pos['z1'])) if first_pos['z1'] else None,
            'median_first_hit_tok_z1t2': float(np.median(first_pos['z1t2'])) if first_pos['z1t2'] else None}
t2_np = np.zeros(V, dtype=bool); t2_np[t2] = True

# --- 4b. GLM 実トラフィック (V4.1 トークン化済み) ----------------------------------------
z1_np = np.zeros(V, dtype=bool); z1_np[z1] = True
tr = collections.Counter(); top_prompt = collections.Counter(); top_answer = collections.Counter()
ans_examples = []; t2_hits_ja = collections.Counter()
files = sorted(glob.glob(f'{ROOT}/work/x2/req/*.meta.json'))[: args.req_files]
for mf in files:
    metas = json.load(open(mf))
    if not metas: continue
    u = np.fromfile(mf[:-10] + '.u32', dtype=np.uint32); off = 0
    for m in metas:
        p, o = m['plen'], m['olen']
        pr = u[off: off + p]; out = u[off + p: off + p + o]; off += p + o
        tr['req'] += 1
        pz = int(z1_np[pr[pr < V]].sum())
        tr['prompt_tokens'] += p; tr['prompt_z1'] += pz
        if pz > 0:
            tr['req_prompt_has_z1'] += 1
            ids_, cnt_ = np.unique(pr[z1_np[np.minimum(pr, V - 1)] & (pr < V)], return_counts=True)
            for a_, c_ in zip(ids_.tolist(), cnt_.tolist()): top_prompt[a_] += c_
        if pz >= 20: tr['req_prompt_z1_ge20'] += 1
        idx = np.where(out == THINK_END)[0]
        ans = out[idx[0] + 1:] if len(idx) else out
        az = int(z1_np[ans[ans < V]].sum())
        a_ = ans[ans < V]
        if len(a_) and sum(1 for c in tok.decode(a_[:400].tolist()) if is_kana(c)) >= 10:   # 日本語の回答 (held-out)
            tr['ja_answer_req'] += 1; tr['ja_answer_tokens'] += len(a_)
            tr['ja_answer_z1'] += int(z1_np[a_].sum()); tr['ja_answer_t2'] += int(t2_np[a_].sum())
            if t2_np[a_].any():
                tr['ja_answer_req_t2'] += 1
                for i_ in a_[t2_np[a_]].tolist()[:2]: t2_hits_ja[i_] += 1
        tr['answer_tokens'] += len(ans); tr['answer_z1'] += az
        if az > 0:
            tr['req_answer_has_z1'] += 1
            for a_ in ans[(ans < V)][z1_np[ans[ans < V]]].tolist(): top_answer[a_] += 1
            if pz == 0 and len(ans_examples) < 12:
                k = int(np.where(z1_np[np.minimum(ans, V - 1)])[0][0])
                ans_examples.append(tok.decode(ans[max(0, k - 12): k + 6].tolist()))
        if az > 0 and pz == 0: tr['req_answer_z1_without_prompt_z1'] += 1
        if len(idx):
            tr['think_tokens'] += int(idx[0]); tr['think_z1'] += int(z1_np[out[:idx[0]][out[:idx[0]] < V]].sum())
        if m.get('tools'): tr['req_tools'] += 1
        if az > 0:
            # 回答内の位置: 自由文 / コードフェンス内 / DSML ツール呼び出し内
            in_tool, in_fence = False, False
            for t_ in ans.tolist():
                if t_ == DSML: in_tool = True
                b_ = raw_bytes(t_) if t_ < V else b''
                if b_ and b'```' in b_: in_fence = not in_fence if b_.count(b'```') % 2 else in_fence
                if t_ < V and z1_np[t_]:
                    tr['answer_z1_' + ('tool' if in_tool else 'fence' if in_fence else 'free')] += 1

# --- 5. マスクの書き出し (bit=1 が許可) ------------------------------------------------
words = (V + 31) // 32
allow = np.full(words, 0xFFFFFFFF, dtype=np.uint32)
for i in z1:
    allow[i // 32] &= ~np.uint32(1 << (i % 32))
allow[-1] &= np.uint32((1 << (V % 32)) - 1) if V % 32 else np.uint32(0xFFFFFFFF)
allow.tofile(f'{OUT}/allow_mask_z1.u32')
json.dump(sorted(z1), open(f'{OUT}/z1_ids.json', 'w'))
json.dump(sorted(t2), open(f'{OUT}/t2_ids.json', 'w'))

summary = {
    'vocab': V,
    'cjk_tokens': cjk_tok, 'kana_tokens': kana_tok,
    'z1_tokens': len(z1), 'z1_tokens_with_kana': len(z1_with_kana),
    'z1_with_kana_examples': [tok.decode([i]) for i in z1_with_kana[:10]],
    'partial_utf8_tokens': len(partial),
    'zh_only_chars_total(gb2312,no-JIS)': len(zh_chars),
    'zh_only_chars_single_token': single, 'zh_only_chars_multi_token': multi,
    'multi_token_examples': ''.join(multi_examples),
    'zh_corpus': dict(zh_stat),
    'zh_corpus_capture_rate': zh_stat['in_z1'] / max(1, zh_stat['zh_char_tokens']),
    'ja_corpus': dict(ja_stat),
    'ja_corpus_z1_examples': ja_examples.most_common(15),
    't2_tokens': len(t2), 't2_examples': [tok.decode([i]) for i in t2_examples],
    'zh_sentence_coverage': sent_cov,
    't2_hits_in_glm_ja_answers': [(tok.decode([i]), c) for i, c in t2_hits_ja.most_common(25)],
    'traffic': dict(tr), 'traffic_files': len(files),
    'traffic_prompt_top_z1': [(tok.decode([i]), c) for i, c in top_prompt.most_common(30)],
    'traffic_answer_top_z1': [(tok.decode([i]), c) for i, c in top_answer.most_common(30)],
    'traffic_answer_z1_examples_without_prompt_z1': ans_examples,
    'mask_bytes': int(words * 4),
}
json.dump(summary, open(f'{OUT}/summary.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
