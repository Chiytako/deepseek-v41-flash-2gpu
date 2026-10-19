# DeepSeek-V4.1-Flash-1017 アーキテクチャ仕様書（推論エンジン実装用）

作成日: 2026-10-18 / 対象: `deepseek-ai/DeepSeek-V4.1-Flash-1017`（MIT）
一次資料: 公式参照実装 `ref/`（`model.py`, `kernel.py`, `engram.py`, `vision.py`, `generate.py`, `convert.py`, `config.json`, `inference_config.json`, `headers.json`）。
`image_processor.py` は ref/ に無く、ダウンロード先 `$MODELS/DeepSeek-V4.1-Flash-1017/inference/image_processor.py` を参照した（他の .py は ref/ と `cmp` で一致を確認済み）。

凡例:
- `model.py:123` は `ref/model.py` の 123 行目を指す。他のファイルも同様。
- 【実測】は headers.json などを機械集計した値。【計算】は式から出した理論値。【推測】は根拠が間接的なもの。
- 集計スクリプトは `<scratchpad (非公開)>/{h1,h2,calc,primes}.py` に置いた（`python3 -I script headers.json` で再現できる）。

---

## 0. 要点（先に結論）

1. **層の構成は 43 層（本体 40 + DSpark 3）**。`compress_ratios` の値は層ごとに 0 / 2 / 1 で、意味は 0=スライディング窓のみ、2=2 トークンを 1 latent に圧縮した KV を追加、1=圧縮なし（1:1 の射影）の KV を追加。どの層も窓 128 の生 KV を持ち、ratio>0 の層はそれに加えて索引器が選んだ top-512 の圧縮位置も見る（`model.py:78-81`, `model.py:775-781`）。
2. **圧縮 KV と索引キーを書き込むのは 4 層（2/8/14/20）だけ**で、他の層はそれを読むだけ。索引器（top-k を選ぶ処理）が動くのは 8 層（2/8/14/20/24/28/32/36）だけで、その間の層は直前の索引器の top-k 結果をそのまま使う（`model.py:82-84`, `model.py:726-727`, `model.py:747-749`）。
3. **KV キャッシュは非常に小さい**。学習時に模した量子化形式のまま保存すると、文脈に比例する部分は **890 B/token**、文脈に依存しない部分（窓リング）は 1 本あたり約 2.9 MB【計算】。
   - 1M 文脈 1 本で 0.94 GB、並列 16 で 15.0 GB。
   - 参照実装は bf16 で保持するため 3,200 B/token になり、1M×16 で 53.9 GB。
4. **デコード 1 トークン（バッチ 1）で読む重みは約 13.0 GB**【実測+計算】。内訳は密層 7.20 GB、routed top-6 で 4.51 GB、lm head（bf16）1.32 GB、Engram 行 12.7 KB。
   - FLOPs は本体 34.0 GFLOP/token と head 1.32 GFLOP。これに文脈長に比例する索引器が加わり、1M では最大 +22〜56 GFLOP。
   - アクティブパラメータは約 16.1B。
5. **Engram のハッシュは入力トークン ID だけで決まり、隠れ状態には依存しない**。したがって先読みできる。
   - 1 トークンあたり 2 層×24 行 = 48 行で、1 行は 256 B（FP8）+ 8 B（UE8M0 スケール）= 264 B。合計 12,672 B/token。
   - 表は 2 層合計 202.8 GB で、safetensors の 47・48 番の 2 ファイルだけに入っている（各 101.5 GB）。
6. **routed experts の FP4 は OCP MXFP4 そのもの**（E2M1 を K 方向に 2 値/byte で詰め、低ニブルが先。32 要素ごとに E8M0 スケール 1 個）。1 エキスパート内のバイト配置は gpt-oss と同じ。
   - 違いは 3 点。活性化が `silu(min(gate,10)) * clamp(up,±10)`（gpt-oss の swigluoai とは別物）、bias が無い、w1/w3 が別テンソル。
   - 参照実装は活性化を MXFP8 に量子化する W4A8 で動く。
7. **参照実装の tilelang カーネルは H100 系（228 KB smem）向けの寸法**になっている。`sparse_attn` は kv バッファ（64×512 bf16 を 2 段）だけで 128 KB を使うので、SM120 の 1 ブロック上限（約 99 KB、【推測: 一般的な仕様値】）に収まらない見込みが高い。再チューニングか別カーネルが必要。
8. **LIL vLLM イメージ（glm53v で稼働中）には V4.1 用の実装が既にある**。`vllm/models/deepseek_v41/`、SM120 向け `vllm/models/deepseek_v4_1/`（b12x 系）、`EngramConfig.table_memory = device|ram|disk`（ディスクは io_uring）。詳細は §9。この調査では読むだけで、実行はしていない。

---

## 1. 計算フロー

### 1.1 全体（`Transformer.forward`, `model.py:1244-1275`）

```text
input_ids[b,s], start_pos, (images, token_types)
image_mask  = token_types >= 0            # 画像スパン（START/IMAGE/NEWLINE/END すべて）  model.py:1252
engram_mask = ~image_mask                 # 画像位置は Engram 対象外                      model.py:1254
hashes = NgramHashState(input_ids, start_pos, engram_mask)  # [b,s,2(層),24]            model.py:1255
h = embed(input_ids)                      # bf16 [b,s,5120]  vocab-parallel             model.py:1256
merge_image_embeddings(h)                 # 画像スパンを ViT/aligner 出力と学習済み区切り埋め込みで上書き  model.py:1231-1242
h = repeat(h, 4 copies)                   # Hyper-Connections: [b,s,4,5120]             model.py:1261
pre_mix = onehot(copy0)                   # [b,s,4] = (1,0,0,0)                         model.py:1162-1166,1263
for i in 0..39:
    if i in {1,14}: h = Engram_i(h, hashes[:,:,idx_i], engram_mask)   # Block の前に実行     model.py:1265-1266
    if i in {37,38,39}: main_hiddens.append(h.mean(dim=copies))      # DSpark 用（層の入力） model.py:1268-1269
    h, pre_mix = Block_i(h, start_pos, pre_mix, image_mask)
h = sum_c pre_mix[c] * h[c]               # 最後の FFN が出した pre で 4 コピーを畳む     model.py:1271
logits = head(RMSNorm(h)) (fp32, 最終位置のみ)  # model.py:1272, 1010-1019
main_hidden = concat(main_hiddens) [b,s,15360]
```

### 1.2 Block（Hyper-Connections の pre/post, `model.py:909-996`, `kernel.py:406-474`）

```text
def hc_mixes(X[b,s,4,5120], fn[24,20480] fp32, scale[3], base[24]):     # model.py:950-957
    x = flatten(X).float()                       # 20480
    m = (x @ fn.T) * rsqrt(mean(x^2) + 1e-20)    # 学習済み重み無しの RMS 正規化
    pre[c]   = sigmoid(m[c]*scale0 + base[c]) + hc_eps                     # 合計 1 には正規化しない  kernel.py:426-427
    post[c]  = 2*sigmoid(m[4+c]*scale1 + base[4+c])                        # 値域 (0,2)              kernel.py:428-429
    comb[j,k]= m[8+4j+k]*scale2 + base[8+4j+k]
    comb = softmax_k(comb) + eps; comb /= (colsum_j + eps)                 # kernel.py:436-448
    repeat 19 回: 行で正規化 → 列で正規化（Sinkhorn 計 20 回, eps=1e-6）    # kernel.py:450-458
    return pre, post, comb

hc_pre(X, pre)                 = Σ_c pre[c] * X[c]                          # model.py:959-962
hc_post(f, R, post, comb)[k]   = post[k]*f + Σ_j comb[j,k] * R[j]           # model.py:964-968

Block.forward(X, pre_mix_in):                                              # model.py:970-996
    R = X
    attn_pre, attn_post, attn_comb = hc_mixes(X, hc_attn_*)
    y = Attention(RMSNorm_attn(hc_pre(X, pre_mix_in)))       # ★ 畳み込みに使う pre は「1 つ前のサブ層」が出したもの
    X = hc_post(y, R, attn_post, attn_comb)
    R = X
    ffn_pre, ffn_post, ffn_comb = hc_mixes(X, hc_ffn_*)
    y = MoE(RMSNorm_ffn(hc_pre(X, attn_pre)), image_mask)    # ★ attn_pre は Block に入った時点の X から計算済み
    X = hc_post(y, R, ffn_post, ffn_comb)
    return X, ffn_pre                                        # ffn_pre は次の層の Attention が使う
```

pre だけが 1 サブ層遅れて使われる（`model.py:916`, `model.py:978-981` の docstring）。post と comb はそのサブ層自身の入力から計算する。
残差ストリームは 4×5120 の bf16（40 KB/token）で、prefill では活性化メモリが通常の 4 倍になる。

### 1.3 Attention（`model.py:614-791`）

```text
qr = RMSNorm_q(wq_a(x))                 # 5120→1280 FP8          model.py:771
q  = wq_b(qr) → [64 heads, 512]         # 1280→32768 FP8 (列並列) model.py:772
RoPE(q[..., -64:])                      # RoPE は末尾 64 次元。隣り合う 2 要素を複素数として回す（interleaved）  model.py:392-406,773
# --- スライディング窓（全層） ---
kv = RMSNorm_kv(wkv(x))                 # 5120→512, MQA 1 ヘッド、K=V で同じベクトル  model.py:706
RoPE(kv[..., -64:]); fake-quant FP8 (1x32, UE8M0)  # RoPE 部分も含む 512 全体    model.py:707-708
ring[pos % 128] = kv                    # decode。prefill ではチャンク内の kv をそのまま使う  model.py:709-721
idx_win = 自分を含む直近 128 位置
# --- 圧縮 KV（ratio>0 の層） ---
if layer is kv_source: latent = Compressor(x)        # 下記 1.4
if layer is index_source: idx_c = Indexer(x, qr, latent)   # 下記 1.5（latent は RoPE 前）
else: idx_c = 直前の index_source が出した結果                      model.py:726-727
if latent: RoPE(latent[..., -64:]) を群の先頭位置 j*ratio で掛ける
           fake-quant FP4 (16 要素ごと, E4M3 スケール) → compress_kv_cache   model.py:752-762
KV  = concat(window_kv, shared compress_kv[: (pos+1)//ratio])
idx = concat(idx_win, idx_c + offset)   # offset は窓 KV の長さ（prefill: s、decode: 128）
o = sparse_attn(q, KV, attn_sink, idx, scale=512^-0.5)   # K=V。sink は分母にだけ加わる  kernel.py:382-385
RoPE^-1(o[..., -64:])                   # ★ V に RoPE 済み部分が混ざるので出力から逆回転で外す  model.py:782
o → 8 群 [b,s,8,4096]; fake-quant FP8; einsum(wo_a[8,1024,4096])   # 群ごとの低ランク射影  model.py:786-789
out = wo_b(flatten) 8192→5120 (行並列 + all_reduce)                 model.py:790
```

RoPE の周波数（`model.py:681-699`）:
- ratio>0 の層: q・窓 KV・圧縮 KV・索引器のすべてに `compress_rope_theta=160000` と YaRN（factor 16、original 65536、beta 32/1）を使う。
- ratio=0 の層（0, 1, DSpark 40-42）: `rope_theta=10000` で YaRN は無効。
- YaRN の mscale（softmax スケールの補正）は掛けていない。scale は `head_dim^-0.5 = 512^-0.5` のまま（`model.py:652`）。

### 1.4 Compressor（`model.py:429-485`）。kv_source 層 2/8/14/20 にだけある

- **ratio 1**（層 20）: `latent = RMSNorm(wkv_c(x))`。プーリングもゲートも無く、wkv は bf16（`model.py:461-462`）。
- **ratio 2**（層 2/8/14）: 演算は fp32 で行い、重みも fp32 に昇格する（`model.py:444-448`）。
  - `kv = wkv_c(x)`、`score = wgate(x)`（どちらも 5120→512）。
  - 群 g の 2 トークン t について、次元ごとに `latent_g[d] = Σ_t softmax_t(score[t,d]) * kv[t,d]` を計算し、最後に RMSNorm（`model.py:473-475`, `model.py:482`）。
  - 群がまだ埋まっていない間は `kv_state` / `score_state`（[ratio,512] fp32 を 2 本）に溜め、出力は None（`model.py:476-484`）。
  - decode では `(pos+1) % 2 == 0` のステップでだけ latent を 1 個出す。
- 圧縮位置 j は「群の最初のトークン位置 j*ratio」の RoPE を受ける（`model.py:753-758`）。
- 位置 p のクエリから見える圧縮位置は `(p+1)//ratio` 個（`model.py:564-568`, `model.py:745`）。群の最後のトークンを過ぎた時点で見えるようになる。

### 1.5 Indexer（`model.py:488-581`）と candidate blocks（`model.py:584-611`）

```text
# 索引キー（owns_k = kv_source の層: 2/8/14/20）
k = RMSNorm(wk(latent))          # 512→128 bf16、RoPE 前の latent から作る     model.py:544
RoPE(k[..., -64:]); fake-quant FP4 (32 要素ごと, UE8M0) → k_cache[pos//ratio]   model.py:545-547
# 照会（index_source の層: 2/8/14/20/24/28/32/36）
q_I = wq_b_I(qr) → [32 heads, 128]; RoPE; fake-quant FP4(32, UE8M0)          model.py:551-553
w   = weights_proj(x) [32] * 128^-0.5 * 32^-0.5                                model.py:556
score[t] = Σ_h relu(q_I[h]·k[t]) * w[h]          # softmax 無し。TP ではヘッドを分割して all_reduce  model.py:557-560
score[未到達の位置] = -inf
if layer == 20 (candidate source):
    blk = max over 8 consecutive positions; 最新の（埋まりきっていない）ブロックは +inf で必ず残す
    candidates = top-2048 blocks → 位置マスク（2048×8 = 16,384 位置）            model.py:597-611
elif layer > 20 (24/28/32/36):
    score = score.masked_fill(~candidates, -inf)     # 2 段目: 自分の重みで、候補ブロックの中だけを採点  model.py:574-576
top-k = min(512, (pos+1)//ratio); 位置の昇順に並べ直す; 範囲外は -1              model.py:579-581
```

参照実装は 24〜36 層でも全位置を採点してからマスクしている（`model.py:557`, `model.py:576`）。エンジンでは候補 16,384 位置だけを採点すれば同じ結果になる（以後「最適化版」と呼ぶ）。

### 1.6 層ごとの役割一覧（`config.json:60-123` と headers.json の重みの有無で確認）

| 層 | ratio | 窓 KV | 圧縮 KV | 索引器 | 備考 |
|---|---|---|---|---|---|
| 0 | 0 | 自前 | なし | なし | theta 1e4、YaRN なし |
| 1 | 0 | 自前 | なし | なし | **Engram（Block の前）** |
| 2 | 2 | 自前 | **書き込み元**（Compressor + wgate） | **書き込み元**（own k + own q） | |
| 3-7 | 2 | 自前 | 層 2 のものを読む | 層 2 の top-k を再利用 | |
| 8 | 2 | 自前 | 書き込み元 | 書き込み元 | |
| 9-13 | 2 | 自前 | 層 8 を読む | 層 8 を再利用 | |
| 14 | 2 | 自前 | 書き込み元 | 書き込み元 | **Engram** |
| 15-19 | 2 | 自前 | 層 14 を読む | 層 14 を再利用 | |
| 20 | 1 | 自前 | **書き込み元（1:1, wgate なし）** | 書き込み元 + **candidate source** | |
| 21-23 | 1 | 自前 | 層 20 を読む | 層 20 を再利用 | |
| 24,28,32,36 | 1 | 自前 | 層 20 を読む | own q + 層 20 の k + candidates | |
| 25-27,29-31,33-35,37-39 | 1 | 自前 | 層 20 を読む | 直前の 24/28/32/36 を再利用 | 37-39 は DSpark の入力を取る層 |
| 40-42（mtp.0-2） | 0 | 自前 | なし | なし | DSpark。experts 128 個で top-3 |

headers の実測: `compressor.wkv` は層 [2,8,14,20]、`compressor.wgate` は [2,8,14]、`indexer.wq_b` は [2,8,14,20,24,28,32,36]、`indexer.wk` は [2,8,14,20]、`engram.embed` は [1,14]（`h2.py` の出力）。

**「共有」の意味**:
- **kv_source_layer_ids による共有**: 同じ ratio の連続する層が、1 本の圧縮 KV（source 層の Compressor が source 層の入力 x から作ったもの）を共有する。消費側の層は、自分の q と自分の窓 KV に対して、source 層の圧縮 KV を引く（`model.py:619-621`, `model.py:740-764`）。
- **index_source_layer_ids による共有**: 選ぶ位置（top-k の idx）を共有する。層 3〜7 は「層 2 のクエリが選んだ 512 位置」をそのまま attention する。
- 共有の受け渡しは `SharedAttentionRuntime` の 1 スロットだけで行う。層を順番に実行することが前提（`model.py:1169-1183`）。

### 1.7 MoE（`model.py:794-906`）

```text
s = sqrt(softplus(x.float() @ W_gate.float().T))            # 384 個、fp32           model.py:813,819
bias = image_mask ? bias_vl : bias                          # 画像トークンには別の補正バイアス  model.py:820-822
idx = topk(s + bias, 6)                                     # バイアスは「選ぶ」ためだけ  model.py:824
w = s[idx] / (Σ s[idx] + 1e-20) * 1.5                       # model.py:825-828
Expert(x) = w2( w * silu(min(w1 x, 10)) * clamp(w3 x, -10, 10) )   # w は w2 の「前」に掛ける  model.py:845-853
y = Σ_routed + SharedExpert(x)                              # shared の重みは 1、同じ clamp   model.py:888-889,905
```

- グループ制限付きのルーティング（n_group / topk_group）は無い。
- 参照実装の TP は「エキスパート単位の分割（rank r が 384/TP 個を持つ）+ fp32 all_reduce」（`model.py:869-872`, `model.py:903-904`, `convert.py:129-134`）。
- shared expert は各 rank が全量を重複して計算している（`model.py:905`）。

### 1.8 Engram 層（`model.py:296-365`, `engram.py`）

```text
rows = Table_l[hash_ids]              # [b,s,24,256] FP8 × UE8M0(1x32) → bf16      model.py:317-320
kv   = wkv_e(flatten(rows))           # 6144→25600 FP8（TP 分割なし）              model.py:353
key[4,5120], value[5120] = split(kv)
for c in 0..3:                        # Hyper-Connections のコピーごと
    rstd = rsqrt(mean(h_c^2)+1e-20) * rsqrt(mean(key_c^2)+1e-20)                 model.py:359
    dot  = Σ_d h_c[d]*q_w[c,d]*k_w[c,d]*key_c[d] * rstd / sqrt(5120)            model.py:356,360
    gate_c = sigmoid( sign(dot) * sqrt(max(|dot|, 1e-6)) )                     # 符号付き平方根  model.py:362
    gate_c = 0 if image token                                                  model.py:363-364
    h_c += gate_c * value             # 4 コピー共通の value をゲートで足す            model.py:365
```

---

## 2. KV キャッシュとトークンあたりバイト数

### 2.1 保存形式（参照実装が学習時の量子化を「模擬」している形式）

| 種類 | 持つ層 | 1 エントリ | エンジンで保存する形式（ロスなし） | 参照実装の実体 |
|---|---|---|---|---|
| 窓 KV | 全 43 層 | 512 次元 | FP8 E4M3 512 B + UE8M0 スケール 16 B = **528 B** | bf16 1,024 B（`model.py:664-669`）|
| 圧縮 KV | 2,8,14（ratio 2）/ 20（ratio 1） | 512 次元 | FP4 256 B + E4M3 スケール（16 要素ごと）32 B = **288 B** | bf16 1,024 B（`model.py:670-680`）|
| 索引 K | 2,8,14 / 20 | 128 次元 | FP4 64 B + UE8M0 スケール（32 要素ごと）4 B = **68 B** | bf16 256 B（`model.py:521-525`）|
| Compressor 状態 | 2,8,14 | [2,512]×2 本 fp32 | 8 KB/本/層 | `model.py:452-456` |
| Engram ハッシュ履歴 | — | 直近 3 トークンの圧縮 ID | 実質 24 B/本（参照実装は全長を int64 で持つ）| `engram.py:155-157` |

参照実装は値を fake-quant（量子化して即座に逆量子化）した bf16 で持つ（`model.py:708`, `model.py:761`, `model.py:546`）。したがって上の詰めた形式で保存しても、参照実装と完全に同じ値になる。

### 2.2 トークンあたりのバイト数【計算】

| 層 | 圧縮 KV B/token | 索引 K B/token |
|---|---|---|
| 2 / 8 / 14（各） | 288/2 = 144 | 68/2 = 34 |
| 20 | 288 | 68 |
| その他の層 | 0 | 0 |
| **合計** | **720** | **170** |

- 文脈に比例する部分の合計は **890 B/token**（参照実装の bf16 では 3,200 B/token）。
- 文脈に依存しない部分は、窓リング 128×528 B×43 層 = **2.91 MB/本**（bf16 なら 5.64 MB）と、Compressor 状態 24.6 KB/本。

### 2.3 文脈長ごとの合計【計算】（`calc.py` の出力）

| 文脈 | 1 本（詰めた形式）| 並列 16（詰めた形式）| 1 本（参照 bf16）| 並列 16（参照 bf16）|
|---|---|---|---|---|
| 128k (131,072) | 0.120 GB | **1.91 GB** | 0.426 GB | 6.82 GB |
| 256k (262,144) | 0.236 GB | **3.78 GB** | 0.847 GB | 13.5 GB |
| 1M (1,048,576) | 0.936 GB | **14.98 GB** | 3.37 GB | 53.9 GB |

- デコード 1 トークンが読む KV は、文脈長に関係なくほぼ一定で、約 8.5 MB/本【計算】。内訳は窓 43 層×67.6 KB と、圧縮 38 層×512 位置×288 B（top-k は層ごとに gather する）。
- 文脈に比例して増えるのは索引 K の読み込みで、詳しくは §3.3。

---

## 3. デコードの重みバイト数と FLOPs、prefill の FLOPs

### 3.1 重みの内訳【実測: headers.json】

| 区分 | GB | 備考 |
|---|---|---|
| routed experts（本体） | 288.78 | 1 エキスパート 18,800,640 B（w1/w3: [2304,2560] I8 + [2304,160] E8M0、w2: [5120,1152] I8 + [5120,72] E8M0）|
| Engram 表 | 202.76 | `layers.{1,14}.engram.embed.{weight,scale}`。**ファイル 47 と 48 だけに入っている** |
| DSpark routed | 7.22 | 128 個×3 層 |
| 注意（本体） | 5.15 | 1 層 126.7 MB（wq_a 6.6、wq_b 41.9、wkv 2.6、wo_a 33.6、wo_b 41.9 MB）と indexer・compressor |
| embed と head | 2.65 | どちらも bf16 [129280,5120] |
| shared experts | 1.42 | FP8 で 35.4 MB/層 |
| ViT と aligner | 0.97 | bf16 |
| DSpark その他 | 0.71 | main_proj FP8 [5120,15360]、markov embed/head bf16 [129280,256]×2 |
| Engram 射影 | 0.315 | wkv FP8 [25600,6144]×2 |
| gate / hc | 0.157 / 0.157 | gate は bf16、hc_* は fp32 |
| **合計** | **510.29** | |

> 訂正: 依頼文の「その他 ≈ 15〜20GB」に対し、experts と Engram 表を除いた実測値は **11.5 GB** だった。

### 3.2 デコード 1 ステップで読む重み【計算】

バッチ 1 の場合:
- 密層 7.20 GB（attn 5.07、shared 1.42、engram 射影 0.315、gate 0.157、hc 0.157、indexer 0.045、compressor 0.037）
- routed 6×18.80 MB×40 = 4.51 GB
- head 1.32 GB（bf16）
- Engram 48 行 12.7 KB
- **合計 13.04 GB/token**

バッチ B の場合、1 層あたりで触れるエキスパート数の期待値は `384·(1-(1-6/384)^B)`。

| 1 ステップのトークン数 B | 触れるエキスパート数/層 | 読み込み/step | 1 トークンあたり |
|---|---|---|---|
| 1 | 6.0 | 13.0 GB | 13.0 GB |
| 16 | 85.5 | 72.8 GB | 4.55 GB |
| 64 | 243.8 | 191.9 GB | 3.00 GB |
| 96（16 本×6 トークンの検証） | 299.3 | 233.6 GB | 2.43 GB |
| 128 | 332.8 | 258.8 GB | 2.02 GB |

注意点:
- 参照実装は `wo_a` を bf16 に展開し（`convert.py:157-173`、+1.34 GB）、gate と head を fp32 で計算する（`model.py:813`, `model.py:1008`）。エンジンでは FP8 / bf16 のまま扱うべき。
- TP2 では wq_a・wkv・compressor・gate・hc・engram wkv が両 GPU に複製される（参照実装の分割方法）。

### 3.3 FLOPs/token【計算】（2×MAC）

| 項目 | MFLOP/層 | 備考 |
|---|---|---|
| 注意の射影 | 253.2 | wq_a + wq_b + wkv + wo_a（8 群）+ wo_b |
| 注意の本体（ratio>0 の層） | 83.9 | 64 ヘッド×512×(128+512)×4。**文脈長に依存しない** |
| 注意の本体（窓のみの層） | 16.8 | |
| routed MoE（top-6） | 424.7 | |
| shared | 70.8 | |
| gate / HC | 3.9 / 2.0 | |
| Engram 射影（層 1, 14） | 314.6 | |

- **本体 34.03 GFLOP/token**（索引器を除く）と **head 1.32 GFLOP**。アクティブパラメータは 16.07B。
- 索引器（デコード 1 トークン、位置 p）:
  - FLOP = `8192·(3·p/2 + p + 4·min(p,16384))`（最適化版）。参照実装の全採点方式では最後の項が `4·p`。
  - 索引 K の読み込み = `68·(3p/2 + p + 4·min(p,16384))` B。

| 文脈 | 索引器 GFLOP（最適化版 / 参照）| 索引 K 読み込み/token/本（最適化版 / 参照）|
|---|---|---|
| 128k | 3.3 / 7.1 | 26.7 MB / 57.9 MB |
| 256k | 6.0 / 14.0 | 49.0 MB / 115.9 MB |
| 1M | 22.1 / 55.9 | 183 MB / 463 MB |

- 1M×16 本では索引 K の読み込みだけで 1 ステップ 2.9 GB（最適化版）になる。q と k はどちらも FP4 なので、FP4 テンサーコア（mxf4 / mxf8f6f4）が使える。
- **prefill の平均 FLOPs/token**（長さ L のプロンプト全体の平均）:

| L | 合計（最適化版） | 合計（参照） |
|---|---|---|
| 8k | 34.3 | 34.3 |
| 32k | 34.9 | 35.0 |
| 128k | 36.0 | 37.6 |
| 256k | 37.3 | 41.1 |
| 1M | 45.4 | 62.0 |

  - head は最終位置だけ（+1.3 GFLOP/リクエスト）。
  - 6k tok/s の prefill には約 216 TFLOPS の実効性能が必要【計算】。実測の FP8 は 576 TFLOPS/GPU。
- DSpark の 1 ドラフトステップ（1 本、5 位置）:
  - 3 層×5 位置×約 0.556 GFLOP + head 5×1.32 + markov 0.33 + main_proj 0.16 ≈ **約 15 GFLOP**【計算】。
  - 読む重みは、DSpark の密層 0.71 GB + experts（B=1 で最大 15 個/層 → 0.85 GB、16 本では約 109 個/層 → 6.1 GB）+ head 1.32 GB。

---

## 4. Engram

### 4.1 ハッシュの決定性（`engram.py`）

1. **トークン ID → 圧縮 ID**（`engram.py:17-61`）
   - tokenizer の backend で 1 トークンずつ decode し、正規化した文字列をキーにする。正規化は NFKC→NFD→StripAccents→Lowercase→空白の連続を 1 個に→Strip（1 文字だけの空白は残す）。
   - `�` を含む部分 UTF-8 トークンは、生のトークン名をキーにする。
   - 圧縮後の語彙数は **99,092 でなければならない**（`engram.py:146`）。
2. **lookback**（`engram.py:164-175`）
   - 位置 p で shift 0..3 のトークンを並べる。系列の先頭より前と DEAD（画像位置。`engram.py:166`）は pad（= token 2 の圧縮 ID）に置き換える。
   - `blocked` は累積する。一度途切れたら、それより古い位置もすべて pad になる。
3. **乗数**（`engram.py:64-83`）
   - `np.random.default_rng(10007*layer_id).integers(0, bound, 4)*2+1`、`bound=(int64max//99092)//2=46,539,438,283,891`。
   - 実際の値（numpy 2.1.0 で計算）:
     - 層 1: `[76632096046245, 4839876093313, 35959672319349, 73987337458391]`
     - 層 14: `[67716810739261, 51510806800915, 30921347202721, 82619226485591]`
   - エンジンでは定数として埋め込めばよい。積は int64 の範囲に収まる（最大でも 9.2e18 未満）。
4. **ハッシュ**（`engram.py:177-184`）
   - `r = t0*m0`。n=2..4 について `r ^= t_{n-1}*m_{n-1}` を順に計算し、`id = r % prime[l][n][head] + offset`（head 0..7）。
   - prime は 16,000,000 より大きい素数を順番に 48 個取り、層 1 に 24 個、層 14 に 24 個を割り当てる（`engram.py:108-118`）。
   - 検証: 素数の合計は層 1 が 384,006,168、層 14 が 384,016,682 で、config の `engram_num_embeddings` と**完全に一致**（`primes.py`）。
5. **結論: ハッシュ ID は「入力トークン ID 列（と画像マスク）」だけの関数**。隠れ状態にも層の計算結果にも依存しない。
   - 同じ 4-gram なら、どのリクエストでもどの位置でも同じ行になる。
   - 画像位置は DEAD として扱い、ゲートも 0（`model.py:1252-1255`, `model.py:363-364`）。画像位置の行は読む必要がない。

### 4.2 1 トークンあたりのアクセス

- 1 層 = 3 種類の n-gram（2/3/4）×8 ヘッド = **24 行**。それぞれが約 1,600 万行の別々の部分表から 1 行ずつ選ばれる（`engram.py:90-92`）。2 層で **48 行**。
- 1 行の重みは 256 B（FP8）、スケールは 8 B（UE8M0。別テンソルなのでディスク上の位置も別）。
- 論理的な量は **12,672 B/token**。
- TP の通信量は、参照実装の all_reduce で bf16 [24,256] = 12 KB/token/層。

### 4.3 TP での分割

- **参照実装**: 行を ceil(N/TP) ずつ連続で分割する（`model.py:303-305`, `convert.py:135-141`）。各 rank が 24 個すべてを引き、範囲外は 0 にしてから bf16 で all_reduce する（`model.py:312-325`）。
- **TP2 では、行分割の境目が部分表の境目とほぼ重なる**。
  - 層 1 では最初の 12 部分表の合計が 192,001,740、ceil(N/2) が 192,003,084 で、差は 1,344 行（層 14 では差 1,325）。
  - したがって、各 rank は 1 トークンあたり実質 12 行を引くだけでよい。部分表単位で分けて all_gather しても同じことになる。
- wkv（6144→25600）を K 方向で分割すると、出力の all_reduce が 51 KB/token になる。行を集める方式（12 KB）の方が通信量は少ない。
- **ハッシュは token のみで決まるので、CPU やホスト側で行を読み、GPU には 12.7 KB/token を送るだけ、という方式も取れる**（6k tok/s でも 76 MB/s）【計算】。

### 4.4 先読み

- **prefill**: リクエストを受けた時点で全位置のハッシュが決まる。チャンク k を計算している間にチャンク k+1 の行を読める。
- **decode**: トークンをサンプリングした直後にハッシュが決まる（直近 3 トークンは既知）。
  - 層 1 で行が必要になるまでの猶予は embed と層 0 の分だけ。層 14 までは 13 層分の猶予がある。
  - NVMe のランダムリードの遅延（約 80〜100 µs と想定【推測】）なら、層 1 にもぎりぎり間に合うと見込む【推測】。
- **投機デコード**: ドラフトが決まった時点で d1..d5 のハッシュが決まり、検証の forward より前に読み始められる。棄却されたドラフトの分も読む必要がある（無駄な IO になる）。

### 4.5 ランダムリードの要求量【計算】

前提:
- 1 行 = 1 IO（4 KiB）。
- 「スケール常駐」= 8 B スケール 6.14 GB を RAM に常駐させる場合（LIL vLLM の `disk_resident_scales` 相当。その説明にも約 5.72 GiB とある）。
- 「スケールもディスク」= 1 行あたり 2 IO。

| 負荷 | 行/s | 論理 MB/s | IOPS（スケール常駐） | 4K 読み GB/s | IOPS（スケールもディスク） |
|---|---|---|---|---|---|
| prefill 6,000 tok/s | 288,000 | 76 | **288k** | 1.18 | 576k（2.36 GB/s）|
| prefill 3,000 tok/s | 144,000 | 38 | 144k | 0.59 | 288k |
| decode 16 本×4 tok×20 step/s | 61,440 | 16 | 61k | 0.25 | 123k |
| decode 16 本×6 tok×20 step/s | 92,160 | 24 | 92k | 0.38 | 184k |
| decode 16 本×6 tok×50 step/s | 230,400 | 61 | 230k | 0.94 | 461k |

- DSpark の block 5 では、1 ステップの検証は「直前の確定トークン t + ドラフト 5 個 = 6 トークン」になる（§5）。依頼文の「1+3」に合わせた 4 トークンの場合も併記した。
- Little の法則から、288k IOPS×約 80 µs で必要なキュー深さは約 23【計算】。S880 のランダム 4K IOPS は**未実測**。fio（`--rw=randread --bs=4k --iodepth=64 --direct=1`）で確かめる必要がある【推測: PCIe4 のコンシューマ SSD なら 500k〜1M IOPS 級】。
- n-gram の頻度は Zipf 的に偏るはずなので、ホットな行を RAM や GPU にキャッシュすれば IOPS を大きく減らせる【推測】。ヒット率は、コーパスをこの tokenizer とハッシュにかけて CPU だけでオフライン測定できる。
- 表 202.8 GB は RAM 128 GB より大きい。そのため NVMe に置くか、ホットな行だけを RAM に置くかのどちらかになる。

---

## 5. DSpark（MTP 3 層）

構成（`model.py:1103-1159`, headers の実測）:
- `mtp.0`: `main_proj`（FP8 15360→5120）と `main_norm` を持つ。
- `mtp.2`: `norm`、`markov_head`（`embed` bf16 [129280,256] と `head` bf16 [129280,256]）、`confidence_head.proj`（bf16 [1,5376] → fp32 で計算）を持つ。
- 3 層すべて: 注意は ratio 0 の窓のみ（theta 1e4）。MoE は experts 128 個で top-3（`model.py:142-149`）。shared 1、gate と bias_vl はあるが VL バイアスは使わない（`model.py:1129`）。
- embed と head は本体と共有している（`model.py:1215-1216`、`convert.py:109-110` で mtp 側の embed/head を読み飛ばす）。

`forward_spec` の手順（`model.py:1277-1285`）:

```text
# 本体の forward で層 37,38,39 の「入力」を 4 コピー平均して連結 → main_hidden[b,s,15360]   model.py:1268-1269
main_x = RMSNorm(main_proj(main_hidden))                    # [b,s,5120]  model.py:1133
draft_ids = [t, NOISE, NOISE, NOISE, NOISE]  (NOISE=128799)  # t = 本体がサンプルした次トークン  model.py:1134-1135
x = repeat(embed(draft_ids), 4 copies); pre_mix = onehot
for stage in mtp[0..2]:
  prefill (start_pos==0):  # 窓 KV の種まきだけ。x は変えない
     main_kv = RoPE(RMSNorm(wkv_stage(main_x)))[pos 0..s) → 直近 128 を ring に入れる   model.py:1041-1054,1126-1128
  decode (start_pos=p):
     ring[p%128] = main_kv(main_x at p)                                               model.py:1067
     q, kv_blk = 5 個のドラフト位置 p+1..p+5（RoPE も p+1..p+5）                        model.py:1057-1064
     attend over [ring(<=128), kv_blk(5)]  # ★ ブロック内は因果マスク無し（全 5 位置が互いを見る）  model.py:1022-1031,1068-1069
     → hc_post → MoE(128 個, top-3) → hc_post  (Block.forward と同じ)
forward_head:                                                                          model.py:1140-1159
  x = hc_pre(x, pre_mix); logits = head(RMSNorm(x))   # 5 位置ぶん、本体の head を使う
  out[0] = t
  for i in 0..4:                                       # 逐次処理だがとても軽い（256 次元）
     bias_i, e_i = markov.head(markov.embed(out[i])), markov.embed(out[i])
     logits[i] += bias_i ; out[i+1] = sample(logits[i])  # 直前のトークンを条件にした bigram 補正
  confidence = proj(concat(x, e))  → [b,5] fp32 スカラー（何に使うかは参照実装に無い）
return out[1..5]（ドラフト 5 個）, logits, confidence
```

エンジン実装で注意すること:
- **参照実装には検証ループが無い**（`model.py:129-130` に「nothing calls forward_spec」とある）。本体側も 2 トークン以上の decode（検証）に対応していない。具体的には `_window_kv` の `squeeze(1)`（`model.py:719`）、Compressor の decode 経路（`model.py:476-482`）、`get_window_topk_idxs` の decode 経路（`model.py:421-424`）が 1 トークン前提になっている。
- エンジンで実装が必要なもの:
  1. 検証用の複数トークン forward
  2. 棄却時に窓リング・Compressor 状態・圧縮 KV・索引 K・Engram の履歴を巻き戻す処理
  3. 確定した全位置の main_kv を DSpark の窓に入れる処理（参照実装は 1 ステップ 1 位置だけ）
- 1 ステップの検証は、t（p+1）と d1..d5（p+2..p+6）の 6 トークン。確定するのは 1〜6 トークン。confidence head は、受理確率を予測してドラフトを途中で打ち切るためのものと考えられる【推測】。
- prefill で DSpark が必要とするのは、窓に入る直近 128 位置の `main_x` だけ（`model.py:1046-1053`）。そのため main_proj は最後の 128 位置だけ計算すればよい。

---

## 6. 量子化形式と既存カーネルとの互換性

| 対象 | 形式 | 根拠 | 標準形式との対応 |
|---|---|---|---|
| 密層の重み（attn, shared, engram wkv, main_proj, indexer wq_b） | FP8 E4M3、**32×32 の 2 次元ブロック**、UE8M0 スケール [ceil(N/32), ceil(K/32)] | `model.py:225-233`, headers（例: `wq_b.scale [1024,40]`）| スケールを行方向に 32 回複製すれば **MXFP8（1×32, E8M0）と完全に同じ**になる |
| 密層の活性化 | FP8 E4M3、トークンごとに 1×32、スケールは 2 のべき乗（ceil）| `model.py:187,197`, `kernel.py:36-37,76-80` | **MXFP8** |
| routed experts | I8 に 2 値/byte で詰める（低ニブル = 偶数要素、`convert.py:31-34`）、[N,K/2]、スケール [N,K/32] UE8M0 | `model.py:219-224`, headers | **OCP MXFP4 そのもの**。値の表は `convert.py:13-15` |
| experts の GEMM | 活性化 MXFP8 × 重み MXFP4（W4A8）。参照実装は FP4→FP32→FP8 に変換して FP8 mma で計算 | `model.py:186-195`, `kernel.py:538-540` | SM120 の `kind::mxf8f6f4`（e4m3×e2m1, ue8m0, block 32）でそのまま計算できる形 |
| 窓 KV | FP8、512 全体を 1×32 / UE8M0 で fake-quant | `model.py:708` | MXFP8 |
| 圧縮 KV | FP4 E2M1、**16 要素ごと、E4M3 スケール**、amax の下限は 6·2^-9 | `model.py:760-761`, `kernel.py:160-163` | **NVFP4 からグローバルスケール（FP32）を除いた形**（グローバルスケールを 1 とみなせば同じ）|
| 索引の q / k | FP4 E2M1、32 要素ごと、UE8M0 | `model.py:546,553`, `kernel.py:164-166` | MXFP4 |
| Engram 表 | FP8 E4M3 で 1 行 256、1×32 UE8M0 で 8 スケール/行 | `model.py:307-320`, headers | MXFP8 の行 |
| wo_a | 保存は FP8 32×32、`convert.py` が bf16 に逆量子化、活性化は FP8 で fake-quant | `convert.py:157-173`, `model.py:787` | FP8 のまま MXFP8 の grouped GEMM にすれば厳密に同じ |
| head / embed / gate / compressor / weights_proj / indexer wk / markov | bf16（head・gate・compressor(r=2)・confidence は fp32 で計算）| `model.py:446-448,813,1008,1096` | — |

**gpt-oss 用の MXFP4 MoE カーネル（vLLM / SGLang）が使えるか**:
- **使える点**:
  - 1 エキスパート内の重みのバイト配置は gpt-oss と同じ。[N, K/2] は gpt-oss の blocks [N, K/32, 16] と同じで、scales [N, K/32] は E8M0（バイアス 127）。
  - ニブルの順序も低ニブルが先で一致する。
- **使えない・注意が要る点**:
  1. 活性化の式が違う。gpt-oss は `(up+1)*gate*sigmoid(1.702*gate)`（clamp 7、swigluoai）だが、本モデルは `silu(min(gate,10))*clamp(up,±10)`。
  2. bias が無い。
  3. w1（gate）と w3（up）が別テンソル。gpt-oss は gate と up を交互に並べた fused テンソルなので、ロード時に結合方法（packed か interleave か）を合わせる必要がある。
  4. 参照実装は routing 重みを w2 の**前**（中間活性化）に掛ける（`model.py:851-852`）。数学的には同じだが、中間活性化を FP8 量子化するときの値が変わる。誤差は小さいと見込む【推測】。
  5. 活性化を MXFP8 にする（A8）と参照実装と一致する。Marlin などの W4A16 経路では一致しない（精度は少し上がるはず）【推測】。
- **LIL vLLM（コンテナ内、読み取りのみ）の状況**:
  - `DeepseekV41FP8Config` は routed experts を `Mxfp4MoEMethod` に、Linear を `B12xFP8LinearMethod` に割り当てる（`/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4_1/quant_config.py`）。
  - fused_moe には `swiglu_limit` の処理（`silu(clamp(gate))*clamp(up)`）がある（`vllm/model_executor/layers/fused_moe/activation.py:181-183, 305-309`）。
  - MXFP4-CSF 版では SILU と A16 / A8（MXFP8）を選べる（`vllm/models/deepseek_v4_1/mxfp4_csf.py` の `Mxfp4CsfMoEMethod`、`VLLM_B12X_MOE_FP4_FORCE_A16`）。
- **DeepSeek-V3 系の「128×128 ブロック FP8」経路（DeepGEMM など）**: ブロック形状が違うため、そのままでは使えない。128 に寄せて量子化し直すと誤差が出る。MXFP8 経路を使うべき。

---

## 7. tilelang カーネルの SM120 互換性（`kernel.py`）

1. **pass_configs**（`kernel.py:9-12`）: `TL_DISABLE_WARP_SPECIALIZED` と `TL_DISABLE_TMA_LOWER` が True になっている。Hopper 専用の TMA と warp specialization を使わず、cp.async と mma.sync 系にコンパイルされるはず。SM120（wgmma が無い）にとってはむしろ都合がよい【推測】。tilelang 0.1.12 が sm_120 を target にできるか、FP4 型（cuda_fp4.h、CUDA ≥12.8）をコンパイルできるかは要確認。LIL vLLM イメージには `tilelang-0.1.12` が入っている（`pip` の dist-info を確認）。
2. **fp8_gemm**（`kernel.py:207-274`）: block 32×128×32、2 段パイプライン。smem は約 13 KB で問題ない。ただし K=32 ごとに fp32 でスケールを掛けて累積し直すので、mma の効率は低い（block-scaled MMA を使わない書き方）。
3. **fp4_gemm**（`kernel.py:477-559`）: FP4 を smem で要素ごとに FP32→FP8 に変換してから FP8 mma にかける（`kernel.py:538-540`）。FP4 のテンサーコアは使わないので、FP8 の mma 速度に変換コストが上乗せされる。性能が出る実装ではない。
4. **sparse_attn**（`kernel.py:310-403`）:
   - **shared memory が足りない**【計算】。
     - q_shared [h,512] bf16、kv_shared [64,512] bf16×2 段、o_shared [h,512]、acc_s_cast [h,64]。
     - h=64（TP1）で約 264 KB、h=32（TP2）で約 196 KB、h=16（TP4 以上、ヘッドは 16 に詰め足し）でも約 162 KB。
     - SM120 の 1 ブロック上限は約 99 KB（101,376 B）【推測: 一般的な仕様値。vLLM の Triton コードでは同じ値を Ada の上限として使っている（`vllm/v1/attention/ops/triton_decode_attention.py:531`, `vllm/third_party/flash_linear_attention/ops/utils.py:180`）。実機では `torch.cuda.get_device_properties().shared_memory_per_block_optin` で確認する】。
     - H100（227 KB）+ README の既定 MP=8（ヘッド 8 を 16 に詰め足し）なら収まる寸法になっている。
   - レジスタ: acc_o [h,512] fp32 を 256 スレッドで持つと、h=32 で 64 reg/thread、h=64 で 128 reg/thread になり、spill が起きやすい。
   - KV の gather は、要素ごとの `if_then_else` ロード（`kernel.py:361-362`）で、cp.async も TMA も使わない。bf16 に逆量子化した KV を読むので、帯域も 2〜3.5 倍になる。
   - 対策の方向: block 16〜32、1 段パイプライン、o_shared を q_shared と共用、KV を FP8 / FP4 のまま読んでカーネル内で逆量子化。または LIL vLLM の `sparse_mla.py`（`supports_compute_capability: capability.major == 12`、`vllm/models/deepseek_v4_1/sparse_mla.py:304-305`）を使う。
5. **act_quant / fp4_quant / hc_split_sinkhorn**: 寸法が小さいので互換性の懸念は少ない。sinkhorn は 1 トークンを 64 スレッドで処理し 20 回反復するので、デコードでは起動遅延の方が目立つと見込む【推測】。
6. **参照実装の構造上の制約**（カーネル以前の問題）:
   - prefill は start_pos=0 の 1 チャンクしか扱えない（`generate.py:67-80`）。
   - 索引スコアを [s, T] の密行列で作る（`model.py:557`）。64k の prefill では層 20 だけで約 17 GB になり【計算】、長文脈の prefill は現実的でない。
   - 画像スパンは最初のチャンクの中に収まっている必要がある（`model.py:1258`, `generate.py:62`）。

---

## 8. ハマりどころ

1. **`rms_norm_eps = 1e-20`**（`config.json:37`, `model.py:75`）。RMSNorm、HC の mixes の正規化（`model.py:955`）、Engram の rstd（`model.py:359`）のすべてで使う。
   - fp16 で計算したり eps を fp16 / bf16 に落としたりすると、0 にアンダーフローする。そうなると、CUDA graph の詰め物の行などゼロ行で `rsqrt(0)=inf`、`inf×0=NaN` になる。
   - NaN が MoE の top-k に入るとエキスパート数の集計（`bincount`）が壊れる。eps は fp32 のまま扱い、詰め物の行の入力は 0 以外にしないか、出力を必ず捨てる。
   - HC の Sinkhorn だけは `hc_eps=1e-6` を使う。gate の正規化は norm_eps ではなく **1e-20**（`model.py:827`）。
2. **swiglu_limit=10 は非対称**: gate は上側だけ（下限なし）、up は ±10 で clamp する（`model.py:847-849`）。shared expert と DSpark の experts にも同じように掛かる（`model.py:889`）。
3. **sqrtsoftplus**: `sqrt(softplus(logit))`。バイアス（`e_score_correction_bias` → `bias`、`convert.py:115`）は選ぶときにだけ使い、重みはバイアス無しのスコアを正規化してから ×1.5（`model.py:823-828`）。
4. **bias_vl**: 画像スパンのトークン（`token_types >= 0`、START/END/NEWLINE も含む）だけに使う、別の補正バイアス（`model.py:795-796`, `model.py:809`, `model.py:820-822`）。学習時の `noaux_tc_for_vl` に対応する。本体 40 層すべてと DSpark にあるが、DSpark では使わない。
5. **attn_sink**: ヘッドごとの fp32 の logit で、softmax の分母にだけ `exp(sink - max)` として加わる（`kernel.py:382-383`）。gpt-oss の sinks と同じ考え方。TP ではヘッド方向に分割する（`convert.py:62`）。
6. **K=V の MQA（head_dim 512）と出力の逆 RoPE**: V に RoPE 済みの 64 次元が含まれるので、出力 o の末尾 64 次元を逆回転してから wo_a にかける（`model.py:782`）。FlashMLA 系のように「nope と rope を分けて V は nope だけ」という構造ではない。
7. **RoPE は interleaved で末尾 64 次元**: 隣り合う 2 要素を複素数として扱う（`model.py:392-397`）。ViT は前半と後半で分ける NeoX 方式（`vision.py:18-21`）。
8. **theta と YaRN は層の種類で変わる**: HF config の `rope_theta=1e4 + yarn` をそのまま全層に当てるのは誤り。ratio>0 の層は theta 1.6e5 + YaRN、ratio=0 の層は theta 1e4 で YaRN なし（`model.py:681-688`）。YaRN の mscale は softmax スケールに入れない。
9. **索引器のスコア**: ReLU → ヘッドごとの重み（`weights_proj(x)·128^-0.5·32^-0.5`）→ ヘッドの総和で、softmax は無い（`model.py:556-558`）。TP でヘッドを分割すると、スコア [T] の all_reduce が必要になる（`model.py:559-560`）。1M では層 20 だけで 4 MB/本/step。**索引器の重みは小さい（wq_b 5 MB/層）ので、全ヘッドを両 GPU に複製して all_reduce をやめる方がよい**【推測】。
10. **top-k の idx の扱い**: 位置の昇順に並べ直し、範囲外は -1、圧縮側の idx には窓の長さ（offset）を足す（`model.py:579-581`, `model.py:777-779`）。有効な idx が 1 つも無い行の出力は 0（`kernel.py:352-355`）。
11. **Compressor は ratio 2 だけ fp32**（重みは bf16 から昇格）。ratio 1 は bf16（`model.py:444-446`）。ratio 2 の softmax は「次元ごと」に取る。
12. **HC の pre は 1 サブ層遅れ**（§1.2）。最後の畳み込みは、最後の層の ffn_pre で行う（`model.py:1271`）。
13. **DSpark の入力は層 37-39 の「入力」を 4 コピー平均したもの**（hc_pre ではない、`model.py:1267-1269`）。ブロック内の注意は双方向（`model.py:1031`）。
14. **Engram の tokenizer 依存**: 圧縮 ID の表を tokenizer から作り直す必要がある（`engram.py:43-59`）。圧縮後の語彙数 99,092 の assert（`engram.py:146`）が外れると、表全体のハッシュがずれ、エラーにならないまま出力品質だけが落ちる。乗数は numpy PCG64 の出力なので、他の言語で実装する場合は §4.1 の定数を使う。
15. **Engram 表の詰め物**: convert は TP 分割で端数の行を、weight=0、scale=1 で埋める（`convert.py:136-141`）。実際に参照されることはない。
16. **画像トークン**: 画像スパンのすべての位置が id 129264 を持ち、token_types で区別する（`image_processor.py:1-10, 132-137, 161-172`）。START/END/NEWLINE は学習済みベクトル（`model.py:1222-1225`）。Engram 用のマスクは DEAD とゲート 0 の 2 段で掛かる。
17. **gate と head を fp32 で計算**（`model.py:813`, `model.py:1008-1014`）。bf16 にすると routing が参照実装と変わる可能性がある【推測】。
18. **MoE の all_reduce と RowParallel は fp32**（`model.py:274`, `model.py:895`）。PCIe で TP2 にするなら、bf16 にするか独自の allreduce が必要になる。デコードでは 1 層あたり 2〜3 回の集団通信があり、40 層で 80 回超。遅延が支配的になる【推測】。
19. **DSpark 側の embed/head は checkpoint に無い（本体と共有）**: `convert.py:109-110`。

---

## 9. エンジン実装への含意（多くは【推測】）

- **メモリの置き場所**: 合計 510 GB に対し、VRAM 192 GB と RAM 128 GB を足しても 320 GB しかない。少なくとも約 190 GB はどこにも常駐できない。
  - アクセスが「1 行 264 B を疎に読む」形の Engram 表（202.8 GB）は NVMe 向きで、表が 47・48 の 2 ファイルにまとまっているのも都合がよい。
  - 一方、routed experts 288.8 GB は VRAM に入りきらない。VRAM から密層・KV・作業領域（約 30 GB）を引くと、experts に使えるのは約 160 GB。
  - 残る約 129 GB を RAM 128 GB に置くことはできない。
  - そのため、experts を何らかの方法で圧縮するか、CPU（AVX-512、DDR5 約 50 GB/s）で計算するかの検討が必要になる。LIL vLLM には「Losslessly compressed DS4.1 routed experts」として MXFP4-CSF がある（`vllm/models/deepseek_v4_1/mxfp4_csf.py:3`）。圧縮率は未確認。
- **LIL vLLM の V4.1 実装**（`glm53v` コンテナ内 `/opt/venv/lib/python3.12/site-packages/vllm/`。grep と cat で読んだだけ）:
  - レジストリ: `DeepseekV41ForCausalLM` と DSpark 用のドラフトモデル（`model_executor/models/registry.py:383-385, 658-661`）。
  - `config/engram.py:76-86`: `table_memory: "device"|"ram"|"disk"`（disk は「SSD io_uring staging」）、`disk_resident_scales`（約 5.72 GiB）、`projection_tp`。
  - `models/deepseek_v4_1/common/engram.py`: 「global ceil-row TP shards」（参照実装と同じ分割）と、b12x のネイティブ lookup・DiskTable。
  - `models/deepseek_v41/nvidia/engram.py`: DP 共有テーブル向けに、ハッシュヘッド単位で分割して all_gather する方式と、非同期プリフェッチ。
  - SM120 専用の入口: `models/deepseek_v4_1/nvidia/vl_model.py`（「SM120 DeepSeek V4.1 with native vision」）、`nvidia/b12x_attention.py`。
- **TP2 の分割の提案**:
  - 注意は q ヘッド（32/rank）と wo_a の群（4/rank）で分割する。MQA なので窓 KV と圧縮 KV は複製（KV は小さいので複製しても問題ない）。
  - 索引器は複製する。
  - routed experts は 192/rank の EP 型（参照実装と同じ）か、intermediate 方向に TP で分割する。
  - Engram は行（部分表）単位で分割する。
  - 1M×16 本でも、KV は複製して 1 GPU あたり 15 GB に収まる。

---

## 10. 未解決・要確認

- SM120 の smem 上限の実測値と、tilelang 0.1.12 が sm_120 と FP4 型をコンパイルできるか（GPU が空いたときに確認する）。
- S880 のランダム 4K IOPS と遅延（fio）。Engram の行ヒット率（コーパスで CPU だけでオフライン測定）。
- Engram 表がファイル内で 4 KiB 境界に揃っているか（safetensors のヘッダ長 + data_offsets。ダウンロードが終わったら先頭 8 バイトから計算する）。256 B の行が 4 KiB ページをまたがないための条件になる。
- DSpark の検証の受理率と、confidence の使い方（参照実装には無い）。
- MXFP4-CSF の圧縮率と、experts を CPU / RAM / NVMe にどう分けるか。
- `encoding/`（チャットテンプレートの Python 実装）と tokenizer.json はまだ読んでいない。Engram の圧縮 ID 表は tokenizer から作る。
