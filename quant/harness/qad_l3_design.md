# L3: 全体 KL による scale-only QAD の実装設計

作成日: 2026-10-18
対象: `DeepSeek-V4.1-Flash-1017` の routed experts (40 層 x 384、SwiGLU、hidden 5120、inter 2304) を EXL3 K2 にした学生
骨格コード: `harness/qad_l3.py` (CPU テスト 45/45 通過、GPU 経路は未検証)
前提資料: `research/SPEC_QUANT_JA.md` (以下 SPEC、§7 反証レビューを含む)、`research/r3_qad.md` (QAD)、`research/r3_japanese.md` (JA)、`results/QUALITY.md`、`research/r3_routing_skew_measured.md`
凡例: 【実測】は手元で確かめた値、【資料】は出典 (ファイル:行) 付きの値、【計算】は式から出した値、【推測】は根拠が間接的な値。
この作業では GPU を使っていない。稼働中のコンテナには触れていない (CPU だけの使い捨てコンテナでテストした)。`stream_ref.py` は変更していない (更新時刻 10:11 のまま)。

---

## 0. 結論

1. **L3 はこの機材で実行できる。** 学生の trellis (K2、1 ステージ 63.3GiB) を 2 枚の VRAM に 20 層ずつ常駐させ、forward と backward で触れる expert の内側行列 Q をその場で復元する。1 枚あたりのメモリは 89.0 / 86.6GiB (容量 95.6GiB) と見積もった【計算、§7.2】。
2. **学習するのは routed experts の suh/svh (342M) だけを既定とする。** `ffn_norm` は router に入るので固定する (SPEC §7 R-6)。最終 `norm` と `attn_norm` は効果を見る追加実験 (アブレーション) に回す。Engram、router、mHC 係数、shared expert、注意の重みは固定する (§2)。
3. **1 step (4,096 トークン 1 系列) は単独で約 11 秒、1F1B で流したときの実効は約 7 秒 (中央の見込み、範囲 4〜14 秒)** になる。1M トークン x 4 epoch の学習は約 2.0 h (1.1〜3.8 h)、教師の取得と評価を含めた全体は約 5.3 h (4.3〜7.3 h) と見込む【計算 + 推測、§7.1】。r3_qad の「学習 7〜10 h」より短いのは、1 step の律速を復元と Python の起動費用に分けて数え直したためで、この差は GPU での実測 (§9 の G2) で確かめる必要がある。
4. **日本語は、データ (日本語が主の系列 40%)、損失の重み (日本語の位置 x2)、区分別の早期終了の 3 つで守る。** L3 で得られる「向上」は一律量子化からの劣化の縮小であり、1017 を超える日本語性能は得られない (SPEC §0.4)。
5. **L2 の後に L3 を足すかは、L2 後の版で 64 本 (約 25 万トークン) x 1 epoch の試験学習 (パイロット) を回して決める。** 教師の取得を含めて約 1 h で済む。 区分を平均した KL が 5% 以上下がれば本番 (1M トークン) に進み、本番では「L2 からの KL の改善が 10% 以上 (SPEC X-7) かつ、どの区分も悪化しない」ことを採用の条件にする (§8)。

---

## 1. L3 の位置づけ

SPEC §2.5 の段階は次のように並ぶ。
L0 (Viterbi 再符号化 + refit) と L1 (Hessian 校正) は trellis を作り、L2 と L3 は trellis を固定して連続の scale だけを合わせる。

| 段 | 合わせる量 | 誤差の見方 | 学習対象 |
|---|---|---|---|
| L2 | 層ごとの expert 出力 (教師 MXFP4 と同じ入力) | 局所 (層内) | suh/svh |
| L3 | 最終 logits (教師 top-32) | 全体 (層をまたぐ誤差の伝わり方を含む) | suh/svh (+ 任意で norm) |

L3 が L2 で取れない分を取れるのは、層をまたぐ誤差の打ち消しと、logits に効く方向への誤差の寄せ方を学べるからである。
L2 は各層の出力誤差 (二乗) を等しく扱うので、後段で増幅される誤差と減衰する誤差を区別できない。
一方 L3 は 40 層すべてを通して backward するので、1 step の費用は L2 の数十倍になる。

出発点は v1 (`DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1`、15:15 頃に完成予定) か、v1 に L2 をかけたパック (`qad_l2.py apply --mode shards` の出力) である。
骨格はパックの suh/svh を初期値 (base) として読むので、どちらからでも始められる (`qad_l3.py:135-176` の `Exl3ResidentSource`)。
L2 の後に回す場合、L3 は L2 の値からの相対変化を学習し、正則化も L2 の値に向かって掛かる。

---

## 2. 学習対象の選定

| 候補 | 数 | 推論で読む形 | L3 での扱い | 理由 |
|---|---:|---|---|---|
| routed の suh/svh | 342M (40 x 384 x 22,272) | fp16、値のまま (`v41_exl3_residency.rs:233-290`、SPEC §0.1) | **学習する (既定)** | 推論の追加費用が無い。L2 と同じ自由度 |
| `ffn_norm` | 40 x 5120 | bf16 | 固定 | 出力が router に入り、top-6 の選択が直接変わる (`ref/model.py:993-994`、SPEC §7 R-6) |
| router (`gate.weight`、`bias`) | 40 x 384 x 5120 | bf16 / fp32 | 固定 | 同上。DSpark の受理率と負荷分布にも響く (SPEC §2.5) |
| 最終 `norm` | 5120 | bf16 | アブレーション A | logits にだけ効き、router と DSpark (層 37〜39 の入力を読む、`ref/model.py:1268-1269`) を通らない。書き戻しは shard 43 (head と同居) の実体コピーになる |
| `attn_norm` | 40 x 5120 | bf16 | アブレーション B | router には直接入らないが、indexer の top-k (離散) を変える (`ref/model.py:986`、`:527-582`)。非 expert の shard の書き換えが要る |
| `q_norm` / `kv_norm` | 40 x (1280 + 512) | bf16 | 固定 | `q_norm` の出力 qr は indexer の入力でもある (`ref/model.py:771`、`:740-764`) |
| mHC 係数 (`hc_*`) | 約 2M / 層 | fp32 | 固定 | 次の FFN の入力の混ぜ方を決めるので router への経路に乗る。効果も小さい見込み (QAD §2.3) |
| shared expert | 1.4B (FP8) | FP8 e4m3、32x32 ブロック | 固定 | e4m3 の丸め幅 (相対 6〜12%) で小さな更新が消える (QAD §2.3)。Adam の状態が 17GB 要る。局所の L2' で扱う |
| 注意の重み、embed、head | 数 B | FP8 / bf16 | 固定 | 量子化していない部分で、教師と同じ値。自由度が大きく過学習しやすい |
| Engram | 2 x 98GB (FP8) | FP8 | 固定 | ユーザーの方針。濁点を区別しない圧縮語彙は学習時からの仕様 (JA §4.4) |
| trellis | 離散 | int16 | 固定 | 勾配が流れない。PV-Tuning 型の離散更新は研究扱い (QAD §6) |

suh/svh は `su = base * exp(theta)` と対数スケールで学習する (`qad_l3.py:644-690`)。
こうすると符号 (Hadamard の符号反転を含む) が base のまま保たれ、expert ごとに大きさの違う scale を同じ学習率で動かせる。
v1 の |suh| は約 0.017 で (qad_l2.py の docstring、`qad_l2.py:16-17`)、絶対値で lr 5e-4 を掛けると相対 3% の一歩になってしまう。
up の svh と down の suh は、up の clamp (±10) が効かない範囲では積しか出力に効かない (SPEC §7.3)。
theta の L2 正則化 (`reg()`) はこの冗長な自由度の値を決める役も持つ。

---

## 3. 学生の計算

### 3.1 W を作らない scale-only の expert

EXL3 の復元は `w = had_l(inner); w *= suh[:, None]; w = had_r(w); w *= svh[None, :]` である (`build/exllamav3_151/exllamav3/modules/quant/exl3.py:239-250`)。
`had_l` と `had_r` は 128 次の Sylvester 型 Hadamard をブロックごとに掛ける (`exl3_lib/quantize.py:401-418`、`util/hadamard.py:108-121`)。
したがって W = diag(su) Hb Q Hb diag(sv) で、x W は W を作らずに次の順で計算できる。

```
a = had(x * su)      # 入力側: 要素積のあと 128 ごとに H を右から掛ける
c = a @ Q            # Q は trellis を su=sv=1 で復元した内側の行列 [in, out]
y = had(c) * sv      # 出力側
```

su と sv について y は線形なので、勾配は要素積の縮約 (`dsv = sum_t had(c)_t * dy_t` など) で求まる。
Q は定数で、勾配も保存も要らない。
`scaled_proj` / `expert_scaled` (`qad_l3.py:180-201`) がこの式で、clamp は gate が上側だけ、up が ±10 である (`ref/model.py:847-849`、SPEC §7 R-7)。

### 3.2 Q を事前に復元して持てるか

持てない。
Q を全部 bf16 で持つと 40 x 384 x 3 x 5120 x 2304 = 543.6G 要素 x 2B = 1,087GB (1 枚あたり 544GB) になる【計算】。
課題文の「543GB」は要素数で、バイト数はその 2 倍である。
FP8 に詰めても 1 枚 272GB で、そもそも MCG の値 (fp16 2 つの和) は FP8 で表せない。
ホスト RAM (128GB) にも入らず、trellis そのもの (2 ステージ合計 136GB) すら RAM に収まらない。
残った VRAM (約 6GiB) で熱い expert の Q を持っても 1 枚約 90 本 (全体の 0.6%) にとどまる。

そこで trellis を VRAM に常駐させ、触れた expert だけを `ext.reconstruct` で Q に戻す (`qad_l3.py:135-176`。呼び方は `exl3.py:225-230` の `_reconstruct_inner` と同じ)。
1 expert の trellis は w1/w3 が [320, 144, 32] int16、w2 が [144, 320, 32] int16 で、計 8.85MB である【実測、v0 パックのヘッダ】。
20 層 x 384 本で 67.95GB (63.3GiB) になり、ds41rt の 0-spark 構成の 64.38GiB/枚 (SPEC §3.1) とほぼ同じ量である。

### 3.3 forward と backward の構成

公式 `ref/model.py` の `Block`、`Attention`、`Compressor`、`Engram`、`Expert` を out-of-place に書き直し、公式 module のパラメータをそのまま使う (`qad_l3.py:575-642`、`:829-971`)。
公式の forward は in-place の RoPE (`apply_rotary_emb`) と in-place の量子化 (`act_quant(..., True)`) を使い、`@torch.inference_mode` で包まれているので、そのままでは backward できない。

| 部品 | forward | backward |
|---|---|---|
| FP8 の Linear (注意、shared、Engram の wkv) | 公式 `linear()` (act_quant + fp8_gemm、`ref/model.py:181-207`) | dX = dY @ W_deq (活性の量子化は STE)、`Fp8LinearSTE` (`qad_l3.py:477-503`) |
| `act_quant` / `fp4_act_quant` (in-place の量子化) | 公式 kernel を複製に掛ける | 恒等 (STE、`_STEFn` `qad_l3.py:405-414`) |
| sparse attention | `stream_ref.sparse_attn_torch` と同じ計算 (bf16 でビット一致、T6c) | chunk ごとに gather をやり直し、lse から P を再構成 (`SparseAttnFn` `qad_l3.py:532-572`) |
| indexer の top-k、router の top-6 | 公式 module を no_grad で呼ぶ | 離散なので勾配なし。router の重み (softmax 側) には勾配が流れる |
| `hc_split_sinkhorn` | torch 版 (`qad_l3.py:417-427`) | autograd |
| routed MoE | `MoEScaleOnlyFn` (`qad_l3.py:276-356`) | expert ごと (bucket ごと) に Q を復元し直し、局所 recompute で dx、route weight、su/sv の勾配を出す |
| 損失 | chunk ごとに logits を作る (`head_kl_loss` `qad_l3.py:710-742`) | chunk ごとの checkpoint |

MoE は token 数の多い順に 16 本ずつ束ね (bucket)、padding した bmm で処理する (`make_buckets` `qad_l3.py:219-225`)。
4,096 トークンの系列では 1 expert あたり平均 64 トークンしか来ない (4,096 x 6 / 384) ので、expert ごとに演算を起動すると 1 ステージで 7,680 回 x 数十 kernel になり、起動費用が律速になる【推測、§7.1】。

### 3.4 離散の判定が学習に与える影響

学生の forward には微分できない判定が 3 種類ある。
router の top-6、indexer の top-k (圧縮 KV のどの位置を見るか)、FP8/FP4 の丸めの境目である。
CPU のテストでは、量子化を外すと学生の logits は公式 Block と相対 1.6e-6 で一致した (fp32 の丸め誤差だけ) 【実測、T12a】。
量子化を入れると、上流の 1e-6 程度の差で境目が反転し、位置によっては相対 7% の差が出た (中央値は 1e-4 以下) 【実測、T12a'】。
同じ理由で、tiny model の bf16 と fp32 の勾配の cos は量子化なしで 0.94、量子化ありで 0.72 だった【実測、T16】。
勾配は離散の判定を固定したときの値であり、step が大きいと判定が変わって損失が予測どおり下がらないことがありうる。
学習率を小さめ (theta で 1e-3、§6) にし、損失の実測が勾配の予測に沿うかを G5 (§9) で確かめる。

---

## 4. 2 GPU のパイプライン

### 4.1 層の分け方

stage0 = 層 0〜19 (embed、Engram の層 1 と 14 を含む)、stage1 = 層 20〜39 (最終 norm、head を含む) とする。
公式の実装では、圧縮 KV と index 用の key が直前の kv_source の層から、top-k 位置が直前の index_source の層から、候補が candidate_source (層 20) から後続の層へ渡る (`ref/model.py:1169-1183`、`:723-764`)。
境目をこれらの供給元と使い手のあいだに置くと、ステージ間で余分な状態を送ることになる。
`valid_splits()` (`qad_l3.py:1317-1340`) で実 config を調べると、使える境目は層 1、2、8、14、20 だけだった【実測、T17】。
20 は条件を満たし、trellis の量も 2 枚で等しくなる。
v2/v3 の K3 を混ぜたパックでは K3 の expert が前半の層に集まる (層 0 に 316 本、層 30〜39 は計 11 本、SPEC §3.3) ので、stage0 のメモリが足りない場合は `--split 14` を使う (§11)。

### 4.2 1F1B と checkpoint

微小バッチ (microbatch) は 1 系列 (4,096 トークン以下) とする。
公式の Compressor は prefill の途中から続きを計算できない (`start_pos > 0` の経路は 1 トークンずつ、`ref/model.py:476-484`) ので、系列をトークン方向に切ることはできない。
`--accum 4` 系列ごとに optimizer を 1 回進め、その 4 本を非 interleaved の 1F1B で流す (`schedule_1f1b` `qad_l3.py:975-988`)。
2 ステージ 4 本の bubble は (4 + 1) / 4 = 1.25 倍である。

activation は層単位で checkpoint し、層の入力 h ([1, T, 4, 5120] bf16、168MB) だけを保存する。
そのうえで、recompute しても結果が同じで費用だけかかる 2 つを保存して使い回す。

- **MoE の出力**：初回の forward の値を保存し、recompute ではそれを返す (`qad_l3.py:286-299`)。routed MoE の再計算 (Q の復元 7,680 本分) を 1 回省ける。
- **indexer の top-k 位置**：初回の値を保存し、recompute では indexer を呼ばない (`qad_l3.py:861-881`)。1F1B では stage0 が B(i) の前に F(i+1) を行い、そのとき kv_source の層 (2、8、14) の `k_cache` が次の系列の値で上書きされる。保存しないと recompute が別の系列の index key を拾い、勾配が相対 53% ずれた【実測、T12c の陰性対照】。

ステージ間で送るのは h (168MB) と pre_mix ([1, T, 4] fp32) で、PCIe P2P (20.6GB/s) で片道約 8ms である。
送信は `isend` にする (`DistLink` `qad_l3.py:1056-1090`)。
blocking の send にすると、stage0 の F(i+1) の送信と stage1 の B(i) の送信が同時に起き、互いに相手の受信を待って止まる。

### 4.3 TP2 にしない理由

ds41rt と同じく expert を中間次元で 2 枚に割る構成 (TP2) も考えられる。
この場合 bubble は無くなる一方で、注意を両方の GPU で重複して計算するか注意も分割する必要があり、前者では FLOP の約 4 割 (注意 337M / 層の 838M) を二重に払う【計算】。
公式のコードを層単位でそのまま使えるパイプラインのほうが実装の差分が小さいので、こちらを採る。

---

## 5. 教師データと損失

### 5.1 教師の取り方

教師は既存のハーネスで取る。
`stream_ref.py --teacher-in <calib>.jsonl --teacher-out <calib>_teacher.pt` に、jsonl の `score_from` を 1 にした系列を渡せば、全位置の top-32 (logprob fp16 と id int32) が保存される (`stream_ref.py:435-473`)。
コードの変更は要らない。
1 位置 192B なので 1M 位置で 192MB である【計算】。
ハーネスの教師は 4,096 トークン 1 本に 52〜61 秒かかる【実測、`work/teacher/teacher.log`】ので、1M トークン (244 本) は GPU 1 枚で約 3.9 h、2 枚で約 2.0 h になる【計算】。

既存の `agent24_teacher.pt` と `ja16_teacher.pt` は評価用で、採点位置も応答側の約 2 万位置だけである (`ja16` は 16 系列 9,014 位置【実測、T13】)。
これらは学習に使わず、最終の比較 (QUALITY.md と同じ指標) にだけ使う。

### 5.2 データの構成

SPEC §2.1 の calib_set.v1 (日本語が主の系列 40%) を 1M トークンに増やして使う。
L1/L2 と同じ校正集合を使ってよいが、評価集合 (agent24、ja16、JA-KL 70 本、MCQ) と同じタスクを含めない (SPEC §2.1 の保留の分離)。
日本語の on-policy 系列 (V4.1 自身の推論と回答) 5% は L3 で特に意味がある。
teacher-forced の L3 は学生が通る系列の分布に合わせて scale を寄せるので、GLM の生成文だけで学習すると V4.1 の文体での誤差が残りやすいと考えられる【推測】。

### 5.3 損失

教師 top-32 の上の KL(P||Q) に、top-32 の外を 1 つの桶にまとめた項を足す (`kl_topk` `qad_l3.py:692-707`)。

```
KL_t = sum_{i in top32} p_i (log p_i - log q_i) + p_tail (log p_tail - log q_tail)
p_tail = 1 - sum p_i,  q_tail = 1 - sum_{i in top32} q_i
loss = sum_t c_t KL_t / sum_t c_t  +  lambda * mean(theta^2)
```

ハーネスの採点 (`stream_ref.py:452-458`) は第 1 項だけで、top-32 の外に学生が確率を流しても罰しない。
学習でその近似を使うと、学生が教師の top-32 外の token に確率を移しても損失が増えないので、桶の項を足した。
報告する指標はハーネスと同じ第 1 項の平均と top-1 一致にそろえる (`head_kl_loss` の戻り値)。

c_t は日本語の位置で 2.0、それ以外で 1.0 とする (SPEC §2.5)。
予測先の token を含む 32 token の窓を decode し、CJK と仮名が 25% 以上で仮名を含む窓を日本語とする (`classify_ja` `qad_l3.py:745-754`)。
L2 の前処理 (`capture_moe.py:83` の `tag_tokens`、窓 16、仮名 3 字以上) と判定の規則が少し違うので、本番前にどちらかに揃える。

---

## 6. 最適化の設定 (初期値)

| 項目 | 値 | 根拠 |
|---|---|---|
| 最適化 | Adam (beta 0.9 / 0.98) | QuIP# の FT (QAD §2.2、arXiv 2402.04396 §F.6) |
| 学習率 | theta に 1e-3、warmup 5%、cosine | QuIP# の su/sv の 5e-4 (±1 の符号ベクトルに対する値) を相対変化の尺度に読み替え、step 数が少ない (§7.1) 分を上げた【推測】 |
| optimizer の 1 step | 4 系列 (16,384 トークン) | 1F1B の bubble を 1.25 倍に抑える。1M トークンで 61 step/epoch |
| 正則化 lambda | 1e-2 (theta^2 の平均) | 冗長な自由度の固定と、冷たい expert の過学習対策 |
| 勾配の clip | 全体ノルム 1.0 (2 ランクで all_reduce) | `clip_grad_global` `qad_l3.py:1413-1426` |
| epoch | 4 (最大 5) | QuIP# の 5 epoch (QAD §1.1) |
| 検証 | 学習データの 5% を区分別に、1 epoch に 4 回 | `train_main` `qad_l3.py:1460-1530` |
| 早期終了 | どの区分も自分の最良値から 2% 以上悪化していない時点のうち、重み付き KL が最小のもの | 日本語の改善のために code / agent を悪化させない (G-JA10) |

冷たい expert は 4,096 トークンの系列 1 本で数回しか通らず、勾配の出る step が少ない。
Adam の moment は勾配が 0 の step でも theta を動かし続けるので、冷たい expert の theta は正則化で 0 に引き戻される側に寄る。
これは「データの無い expert は L2 の値のまま」という望ましい振る舞いだが、日本語でだけ熱い expert (1 層あたり平均 6.2 本、`research/r3_routing_skew_measured.md`) にも同じことが起きないよう、日本語 40% のデータ構成が要る。

---

## 7. 時間とメモリの見積もり

### 7.1 1 step と全体の時間

`python3 qad_l3.py estimate` (`qad_l3.py:1242-1314`) の出力である【計算 + 推測】。
FLOP は config から数え、1 トークン 1 層で forward 0.838 GFLOP、backward 側 (recompute を含む) 1.763 GFLOP になった。
backward 側には、層の recompute (MoE を除く)、注意の dX (sparse attention は forward の 2 倍)、MoE の局所 recompute と dX (routed で forward の 2 倍) が入る。
重みの勾配の GEMM は無い (学習するのは scale だけ)。

時間は 4 つの項の和とし、場合ごとに仮定を変えた。

| 項 | 楽観 | 中央 | 悲観 |
|---|---|---|---|
| bf16 の実効 (小さな GEMM と torch の演算を含む) | 150 TFLOPS | 80 | 40 |
| Q の復元 1 tensor (5120 x 2304) | 15us | 30us | 60us |
| bucket 1 回の起動と Python (forward / backward) | 0.15 / 0.6ms | 0.3 / 1.2ms | 0.6 / 2.5ms |
| sparse attention の gather の実効帯域 | 1.5TB/s | 1.2TB/s | 0.8TB/s |

当機の bf16 GEMM のピークは 299 TFLOPS、W4A16 の MoE kernel は M=2048 で 157 TFLOPS だった【実測、GLM の計測メモ】。
4,096 トークンの系列では expert 1 本あたり平均 64 行の GEMM になり、ピークの数分の 1 しか出ないと見て、中央を 80 TFLOPS とした【推測】。
Q の復元は 23.6MB の fp16 書き込みが主で、1.79TB/s なら 13us の計算になる【計算】。

| 場合 | s0 fwd | s0 bwd | s1 fwd | s1 bwd | 1 step 単独 (s) | 1F1B の実効 (s / 4,096 tok) | tok/s | 1M tok / epoch | 1M tok x 4 epoch |
|---|---|---|---|---|---|---|---|---|---|
| 楽観 | 1.06 | 2.10 | 1.07 | 2.16 | 6.4 | 4.1 | 1,009 | 0.28 h | 1.1 h |
| 中央 | 1.92 | 3.70 | 1.96 | 3.80 | 11.4 | 7.2 | 567 | 0.49 h | 2.0 h |
| 悲観 | 3.74 | 7.12 | 3.81 | 7.33 | 22.0 | 13.9 | 294 | 0.95 h | 3.8 h |

「1 step 単独」は 1 系列を流して片方の GPU が待つ時間を含めた値で、`--accum 1` で回すとこの時間になる。
epoch を変えたときの学習時間は中央で 1 epoch 0.49 h、3 epoch 1.5 h、5 epoch 2.5 h である。

ジョブ全体 (1M トークン x 4 epoch、中央) の内訳は次のとおりである。

| 段 | 時間 |
|---|---|
| 教師 (ハーネス、2 枚に分けて 244 本) | 2.0 h |
| 起動 (trellis 136GB と非 expert の読み込み) | 約 0.1 h |
| 学習 | 2.0 h (1.1〜3.8) |
| 学習中の検証 (12 本 x 16 回、forward のみ) | 約 0.2 h |
| 書き出し (expert shard 40 本の実体コピー) | 約 0.1 h |
| 最終評価 (パイプライン内で agent24 + ja16、ハーネスで確認) | 約 0.9 h |
| 計 | 約 5.3 h (4.3〜7.3) |

r3_qad §2.1 は 1 step を 15〜25 秒と見ていた。
差の主因は、r3_qad が全 expert の復元を 300G 重み/秒の 1 回で見積もったのに対し、本設計では MoE の出力の保存で復元を forward と backward の 2 回に限り、復元の速度を書き込み帯域から見積もった点にある。
どちらも未測定なので、G2 (§9) で 1 層の forward と backward を測り、`estimate()` の仮定を置き換える。

### 7.2 メモリ (1 枚あたり、GiB)

| 項目 | GPU0 (層 0〜19) | GPU1 (層 20〜39) | 内訳 |
|---|---:|---:|---|
| trellis (常駐) | 63.28 | 63.28 | 7,680 本 x 8.85MB |
| base の su/sv (fp32) | 0.64 | 0.64 | 7,680 x 22,272 x 4B |
| theta (fp32) | 0.64 | 0.64 | |
| 勾配 (fp32) | 0.64 | 0.64 | |
| Adam の m, v | 1.27 | 1.27 | |
| 非 expert の重み | 5.37 | 6.29 | 注意 (wo_a は bf16 に戻すので 2 倍)、shared、router、mHC、Engram の wkv (GPU0)、embed (GPU0)、head fp32 (GPU1) 【実測、checkpoint のヘッダから集計】 |
| 保存する activation | 9.69 (2 系列) | 4.84 (1 系列) | 1 系列 1 層で h 168MB + MoE 出力 84MB + top-k 8MB |
| 作業領域 | 5.00 | 6.50 | 1 層の recompute と backward (sparse attention の chunk、hc の fp32、bucket 16 本の Q 1.1GB)、GPU1 は logits の chunk (1,024 x 129,280 x 4B x 約 4) |
| CUDA context と断片化 | 2.50 | 2.50 | |
| **計** | **89.02** | **86.60** | 容量 95.59GiB |

「1 step (4,096 トークン) あたりのメモリ」と言える増分は、保存する activation の 4.84GiB/ステージである。
GPU0 の余裕は 6.6GiB しかない。
足りない場合は、MoE 出力の保存を bf16 にする (-0.8GiB)、embed を CPU に置く (-1.2GiB)、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を付ける (既存の起動例と同じ、`build/chain_g1.sh:6`)、の順で詰める。

ホスト RAM は、各ランクが非 expert の重みを読む一時領域 (約 6GB)、教師の top-32 (192MB)、Engram の行の保存 (stage0 のみ) で 40GB 以内に収まる。
Engram の行は系列だけで決まる定数なので、1 epoch 目に pread で集めたものをホスト RAM に置き、2 epoch 目以降に使い回す (`StudentStage.prepare` `qad_l3.py:936-948`)。
骨格は bf16 に戻した行 (1 系列 2 層で約 100MB) を持つので 1M トークンで約 25GB になり、FP8 の生の行で持てば約 13GB に減らせる【計算: 4,096 x 24 行 x 264B x 2 層 x 244 本】。

---

## 8. L2 の後に L3 を足す価値の判断基準

判断は 4 段で行い、前の段を通らなければ先に進まない。

**(1) 実装の正しさ (前提)。**
θ=0 のパイプラインの評価が、同じパックをハーネスで採点した KL と top-1 を再現すること (KL の差 ±0.002 または ±1% 以内) を条件にする (§9 の G4)。
再現しない状態で学習した改善は、評価の差か学習の効果かを区別できない。

**(2) L2 の後に残る誤差の大きさと性質。**
L2 の後の版 (v3 か v1 + L2) で、次の 2 つを測る。

- 区分別の KL と G-JA1 (日本語の KL / en-neutral の KL)。G-JA1 が 1.2 を超えるなら、日本語に重みを置ける L3 を試す理由が強い。全ゲートを余裕を持って満たしているなら L3 は省いてよい。
- 誤差の相互作用の比 rho = KL(全層を量子化) / Σ_g KL(群 g の層だけを量子化)。群は SPEC §2.2 (4) の α 測定と同じ 5 層 x 8 群を使い、`--exl3-layers` の改修で測れる。rho が 1.2 を大きく超えるなら、層をまたぐ誤差の増幅が大きく、層ごとに閉じた L2 では取りきれない分が残っていると考えられる。rho が 1 に近い場合でも L3 が効かないとは言えない (L3 は logits に効く方向へ誤差を寄せられる) ので、rho は (3) の優先度を決める材料にとどめる。

**(3) パイロット (教師を含めて約 1 h)。**
校正集合から 64 本 (約 25 万トークン) を 1 epoch 学習し、検証集合で区分別の KL を測る。
本番用の教師は 64 本ずつに分けて取り、最初の 64 本 (2 枚で約 30 分) が揃った時点でパイロットを始める。
学習は中央の見込みで 8 分、起動と検証を含めて約 30 分である【計算】。
区分を平均した KL が 5% 以上下がり、どの区分も 1% を超えて悪化しないなら本番に進む。
2% 未満の改善なら、L3 はこの自由度 (suh/svh) では効かないと判断して打ち切る。
その間なら、アブレーション A (最終 norm を加える) を同じ条件で試してから決める。

**(4) 本番の採用条件。**
本番 (1M トークン、最大 5 epoch) の結果を、L2 の版と同じ評価集合で比べる。

| 条件 | 基準 | 由来 |
|---|---|---|
| KL の改善 | agent24 + ja16 (+ JA-KL) の区分平均で 10% 以上 | SPEC X-7 (`SPEC_QUANT_JA.md:542`) |
| 統計的な差 | 系列単位の対応のある bootstrap で、改善の 95% 区間が 0 を含まない | 評価集合が 40 系列と小さいため |
| 悪化の禁止 | どの区分も L2 の版から 3% を超えて悪化しない。en-neutral と code は G-JA10 (v1 から 10% 以内) | SPEC §4.2 |
| 日本語 | G-JA1 が L2 の版より悪化しない | 同上 |
| 生成 | ds41rt の 2 段目で G-JA5 (日本語で答える率)、G-JA6 (簡体字の混入)、DSpark の 1 round のトークン数 (G-JA9) が L2 の版より悪化しない | KL に出にくい尾の変化を見る |

採用しない場合も、L3 の費用は GPU 約 5 h と書き出し用のディスク 137GB で、パックの形式は変わらないので、L2 の版に戻すだけで済む。
VRAM を K3 に使い切った後 (SPEC §3.2 の案 B など) にまだ日本語のゲートを満たさない場合は、追加の VRAM を要しない L3 が残る唯一の手段になるので、(3) の結果が境界的でも本番まで回す価値がある。

---

## 9. GPU での検証手順

GPU が空いてから、上から順に行う。
いずれも `dsv41-quant:151` で、`build/chain_g1.sh:5-7` と同じマウント (`/ref`、`/model`、`/h`、`/t`) に学生のパック (`/pack`) を加える。

| ID | 中身 | 合格の目安 | 所要 |
|---|---|---|---|
| G0 | CPU テストをもう一度回す: `python3 /h/qad_l3.py selftest` | 45/45 | 1 分 |
| G1 | `Exl3ResidentSource.inner` + `dense_from_inner` と `LinearEXL3(...).get_weight_tensor()` を、層 0 と 39 の expert 数本で比べる | 最大差が fp16 の丸め程度 (相対 1e-3 以下) | 10 分 |
| G2 | 1 層 (層 0) の MoE で、4,096 トークンの forward と backward を bucket 1/8/16/32 で測る。`ext.reconstruct` 1 回の時間も測る | `estimate()` の仮定を実測に置き換え、§7 を更新する | 30 分 |
| G3 | `KernelSet("official")` の `act_qdq` / `fp4_qdq` / sinkhorn を、エミュレーション (`qdq_fp8_emul` など) と実データで比べる | act_quant はビット一致。sinkhorn は fp32 の丸め程度 | 10 分 |
| G4 | θ=0 で、v1 パックの agent24 と ja16 をパイプラインの `forward_only` で採点し、同じパックのハーネス採点 (`stream_ref.py --exl3-pack`) と比べる | KL の差が ±0.002 か ±1% 以内、top-1 の差が 0.3 点以内 | 1.5 h (ハーネス側が 1 h) |
| G5 | 256 トークンの系列 2 本で、theta の数方向に ±eps を振った損失の差分と、勾配から予測した変化を比べる。あわせて 1 step 進めたときの損失の実測と予測を比べる | 方向微分の相対差 1e-2 以下。1 step の損失低下の比が 0.5〜1.5 | 30 分 |
| G6 | 8 系列 x 2 step で `train` を回し、メモリの最大値 (`torch.cuda.max_memory_allocated`)、1 step の時間、ログの KL を確かめる | GPU0 で 92GiB 以下、時間が G2 からの予測の 1.5 倍以内 | 20 分 |
| G7 | §8 (3) のパイロット | §8 (3) の基準 | 30 分 |
| G8 | `export` で書いたパックを、ハーネス (`--exl3-pack`) と ds41rt の `--dry-run` で読む | 学習ログの最良の検証値とハーネスの採点が一致。ds41rt が拒否しない (非有限値と fp16 の範囲は書き出し時に検査済み) | 1 h |

G1 から G3 は GPU 1 枚、G4 から G7 は 2 枚を使う。
L2 の実行と同時にはできないので、L2 の作業が終わってから回す。

起動の例 (G6 以降、未実行):

```
sudo docker run --rm --gpus all --network none --ipc host -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $DSV41/ref:/ref:ro -v $MODELS/DeepSeek-V4.1-Flash-1017:/model:ro \
  -v $MODELS/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1:/pack:ro \
  -v $DSV41/harness:/h:ro -v $DSV41/work/l3:/w \
  --entrypoint bash dsv41-quant:151 -c 'cd /h && torchrun --nproc_per_node 2 qad_l3.py train \
    --pack /pack --teacher /w/ja_teacher.pt --teacher /w/code_teacher.pt --teacher /w/agent_teacher.pt --out /w/run1 --epochs 4'
python3 qad_l3.py export --pack <v1> --theta run1/theta_r0_best.pt --theta run1/theta_r1_best.pt --out <新パック>
```

`train_main` は `--teacher` のファイル名の最初の `_` より前を区分名として、検証の集計と早期終了に使う。
そのため教師は区分ごとにファイルを分け、`ja_teacher.pt` のように区分名を先頭に置く。

---

## 10. 骨格コードの構成とテスト

`harness/qad_l3.py` (CPU で import とテストが通る範囲)。
GPU を要する所 (`Exl3ResidentSource`、`KernelSet("official")`、`build_stage_real`、`DistLink`、`train_main`) は import 時に CUDA 拡張を読まず、呼んだときだけ読む。

| 節 | 行 | 中身 |
|---|---|---|
| 1 | 53-95 | Hadamard (Sylvester)、`dense_from_inner` / `inner_from_dense` |
| 2 | 97-176 | Q の供給元 (テスト用の dense 版、GPU 用の常駐 trellis 版) |
| 3 | 179-352 | scale-only の expert、bucket、`MoEScaleOnlyFn` |
| 4 | 354-503 | FP8/FP4 のエミュレーション、STE、sinkhorn、FP8 Linear |
| 5 | 505-572 | sparse attention の autograd 版 |
| 6 | 574-641 | 公式 Block の部品の out-of-place 版 |
| 7 | 643-689 | `ScaleParams` (base x exp(theta)、正則化、fp16 の書き出し) |
| 8 | 691-739 | top-32 + 桶の KL、chunk 化した head |
| 9 | 740-815 | 日本語の重み、教師 rec からの microbatch |
| 10 | 817-972 | `StudentStage` (層の forward、checkpoint、top-k と MoE 出力の保存) |
| 11 | 974-1109 | 1F1B の順序、`StageRunner`、`LocalLink` / `DistLink` |
| 12 | 1110-1225 | safetensors の読み書き、パックの書き出し (対象 shard だけ実体コピー) |
| 13 | 1226-1315 | 見積もり |
| 14 | 1316-1561 | GPU 経路 (未検証)、`export` (CPU で動く) |
| 15 | 1562-2109 | CPU テスト (`selftest`) |

テストは `dsv41-quant:151` と `dsv41-quant:5be8` の両方で、GPU を割り当てない使い捨てのコンテナ (`--network none`) で回し、どちらも 45/45 だった【実測】。

| テスト | 確かめたこと |
|---|---|
| T1, T2 | Sylvester の H128 が exllamav3 の `get_hadamard(128)` と一致。`dense_from_inner` が exllamav3 の `preapply_had_l/r` の手順と一致 (fp32 の精度)。`scaled_proj` = x @ W (double で 1e-13) |
| T3 | `MoEScaleOnlyFn` の forward が dense の参照 (stream_ref の bf16 経路の式) と一致 (bucket 1 と 3)。gradcheck (x、route weight、6 本の scale、clamp が効く入力)。checkpoint 用の保存が recompute で使われる |
| T4, T5 | FP8/FP4 のエミュレーションの性質、STE、sinkhorn の gradcheck |
| T6 | `SparseAttnFn` の forward が `stream_ref.sparse_attn_torch` と bf16 でビット一致、gradcheck |
| T7, T8 | FP8 Linear の STE。KL 損失の gradcheck、chunk + checkpoint の損失と勾配が一括計算と一致 |
| T9 | 日本語の判定と位置の重み |
| T10, T15 | パックの書き出し (対象 shard は新しい inode、他は hardlink、元のファイルは不変)。`export` が base x exp(theta) を fp16 で書き、trellis を変えない |
| T11 | theta=0 で scale = base、書き出しの名前と値 |
| T12 | 公式 `ref/model.py` の tiny model (6 層、圧縮比 0/2/1、kv/index/candidate の供給元を含む) に対し、学生の logits が一致 (量子化なしで相対 1.6e-6)。2 ステージ 1F1B + checkpoint の theta の勾配が一括計算と完全に一致。top-k を保存しない陰性対照では勾配がずれる。`engram_apply` が公式 `Engram.forward` と一致。評価経路の KL が一括計算と一致 |
| T13 | 実ファイル `work/teacher/ja16_teacher.pt` の形式 (16 系列、9,014 位置、top-32 fp16) |
| T16 | bf16 (GPU と同じ dtype 構成) でも 1F1B = 一括、勾配は有限 |
| T17 | 実 config で使えるステージの境目が [1, 2, 8, 14, 20] |

CPU のテストで確かめられないのは、CUDA 拡張 (`ext.reconstruct`、tilelang の kernel、NCCL) を通る部分と、実寸での時間とメモリである。
これらは §9 の G1〜G6 で確かめる。

---

## 11. リスクと未確認事項

| リスク | 根拠 | 対処 |
|---|---|---|
| 1 step が見積もりより遅い | Q の復元と Python の起動費用は未測定 (§7.1) | G2 で測る。遅ければ bucket を大きくする、forward だけ EXL3 の MoE kernel (W を作らない) に替える、`reconstruct_batch` (exllamav3 の batch 復元 kernel) を使う |
| GPU0 のメモリ不足 | 余裕 6.6GiB (§7.2) | §7.2 の順で詰める |
| K3 を混ぜたパック (v2/v3) では、さらに不足する | K3 は前半の層に集まり、1 本あたり +4.42MB (SPEC §3.1)。案 B の 2,960 本が層 0〜19 に寄ると GPU0 に約 +10GiB【推測】 | `--split 14` にする。それでも足りない場合は冷たい expert の trellis をホスト RAM に置き、触れたときに PCIe で送る (8.85MB で 0.4ms)。`Exl3ResidentSource` は今は (層、射影) ごとに K が同じことを前提に積んでいるので、expert ごとの K に対応させる改修が要る |
| 離散の判定で損失が勾配どおりに下がらない | §3.4 の測定 | 学習率を小さめにし、G5 で確かめる |
| 冷たい expert の過学習 | 1 系列で数回しか通らない expert がある | 正則化、区分別の早期終了 |
| 教師との数値のずれ | 学生の torch 版 sinkhorn、einsum の hc_post (加算順) は公式 kernel と fp32 の丸め程度違う | G3 と G4 で、θ=0 の評価がハーネスと一致することを確かめる |
| ds41rt で svh が clamp の前に掛かっているか | SPEC §7-2 の未確認事項。L2 と同じ | 層 0 の参照比較 (SPEC §2.5)。L3 で動かした scale でも同じ確認をする |
| ディスク | 空きは 229GB (v1 の作成中、28/40 shard)【実測】。v1 の完成で約 190GB、L2 のパック (shards) と L3 のパックがそれぞれ 137GB | L3 の前に v0 の expert shard (137GB) を消すか、ユーザー許可済みの `MiMo-V2.6-Flash-RL` (166GB) か `GLM-5.3-Flash-W4A16` (166GB) を消す。学習中の theta は最良と最新だけを残す (1 ランク 0.68GB x 2) |
| DSpark の受理率の変化 | DSpark は層 37〜39 の入力を読む (`ref/model.py:1268-1269`) | §8 (4) の生成の条件で監視する |
| 日本語の判定の不一致 | L2 の `capture_moe.tag_tokens` と規則が違う (§5.3) | 本番前に揃える |
