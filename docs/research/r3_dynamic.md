# r3: 「動的量子化」の選択肢と、ds41rt / LIL b12x / EXL3 で実装できる形

作成日: 2026-10-18 / 対象: `deepseek-ai/DeepSeek-V4.1-Flash-1017` の routed experts (40 層 × 384、1 expert = 3×5120×2304 = 35.39M 重み)
前提: `PLAN.md`、`DECISION.md`、`r2_quant_tools.md` (以下 QT)、`r2_ds41rt.md` (DR)、`r2_lil_lowbit.md` (LL)、`r2_engram_hit.md` (EH)、`arch_spec.md` (AS)
方針: GPU は使っていない (量子化と教師取得が動作中)。docker コンテナには触れていない。重みは取得していない。
凡例: 【実測】はこの調査で手元で出した値。【資料】はコードか公開資料の記載。【計算】は式から出した値。【推測】は根拠が間接的な値。

この文書の範囲は「動的量子化」である。QAD と TensorFold の仕様は別の文書で扱う。ただし、両者と関わる点は §5.6 に 1 行ずつ書いた。

**スクリプトと出力**

- スクリプト: `tools/r3_*.py`
  - `r3_load_counts.py`: torch を使わずに教師 .pt から routing 回数を取り出す
  - `r3_skew.py`: 偏りを集計する
  - `r3_alloc.py`: 配分をシミュレーションする
  - `r3_robust.py`、`r3_dyn.py`、`r3_k1.py`、`r3_dec.py`、`r3_bytes.py`
- 出力: `research/tmp/r3/`
  - routing 回数: `agent_counts.json`
  - 各表の元データ: `*_out.txt`、`skew_agent.txt`

---

## 0. 結論

1. **「動的量子化」のうち、この機材で効果があり、すぐ実装できるのは「静的な expert 別混合精度」(Unsloth Dynamic と同じ発想) だけである。** 実装先は ds41rt である。
   - ds41rt の staged manifest は、projection (w1/w3/w2) ごとに K を持てる (`v41_exl3.rs:440-466`)。
   - 1 チェックポイントに、隣り合う 2 段の K (k23 = K2/K3、k34 = K3/K4) を混ぜて 1 回の起動で動かせる (`v41_exl3_residency.rs:197-209`、`native/include/ds41rt_native.h:210-221, 916-923`)。
   - 現行の一律 K2 も、実は「空の K3 段を持つ混合カーネル」で動いている (`v41_exl3.rs:179-183`)。このため K2/K3 を混ぜても、カーネル切り替えの費用は増えない。増えるのは読むバイト数だけである【資料: `docs/release-v7-exl3-k2-homogeneous.md:122-140`】。
2. **1017 の実トラフィックでは、routing の偏りが公開値より強い。**【実測】
   - データ: エージェント型の要求 17 本 (68,308 トークン) の教師実行。
   - 結果 (40 層の平均): 上位 25% の experts が routing の 69.3%、下位 30% が 3.7%、Gini 0.60。
   - 公開値は 0909 の混合コーパスで、上位 25% が約 56%、下位 30% が 7.2%、Gini 0.46 だった。
   - したがって、expert 別配分の効果は 0909 での見積もりより大きい。
3. **配分のシミュレーション (代理指標。§2.4)【計算 + 推測】**
   - 平均 2.20bpw (experts 149.5GB) で expert 別に K2/K3 を配ると、代理損失は一律 K2 の **0.38 倍**になる。層別 K2/K3 では 0.59 倍で、一律 K3 (204GB、VRAM に入らない) は 0.16 倍である。
   - K4 を足して 3 段 (k234) にしても、0.383 が 0.375 になるだけである。**k23 で十分**と判断する。
   - ただし、計画を立てた分布と評価する分布が違うと劣化する。routing が一様な分布に変わる最悪の場合、0.38 倍は 0.67〜0.69 倍まで悪化し、層別配分 (0.59) よりも悪くなる。
   - **日本語の routing を測らずに agent の頻度だけで配分すると、日本語が最も損をする構図になる。** 日本語 (ja16) の routing 回数を取ってから配分すること。
4. **実行時に動的な方式 (ホット expert の高ビット版の差し替え、残差の追加) は、この構成では静的配分に勝てない。**【計算】
   - VRAM の余り (2 枚で約 13〜17GB) をどれに使うかで比べた (代理損失、計画と評価が同じ分布の場合)。
     - 静的に K2→K3 へ置き換える: 3,840 本、損失 0.33。1 本あたり 4.42MB で済むため。
     - 高ビット版を重複して持つ動的プール: 1,280 本、損失 0.58。
     - K2 の残差を加える方式: 1,920 本、損失 0.41。
   - PCIe x8 (約 20GB/s/枚) では、1 step 内の差し替えは不可能である (c1 で 1 step に約 0.9GB 必要)。秒〜分単位の追従しかできない。
   - したがって、動的方式は「静的配分を入れたあと、セッション単位の routing のずれが大きいと実測で分かった場合」だけ検討する。
5. **活性化の動的量子化 (per-token FP8) は品質面の利点が無い。**
   - 公式の参照実装は、experts の活性化を 1×32 ブロックの MXFP8 にしている (`ref/model.py:186-197`)。ds41rt の EXL3 経路は A16 で、参照実装より精度が高い側にある。
   - 速度のために A8 にするなら、行ごとの単一 scale ではなく 1×32 ブロック (参照と同じ) を使うこと。日本語への影響は無いと見る【推測】。
6. **日本語を守るための要点は 3 つ。**
   - (i) 配分に使う routing 統計に日本語を入れ、重みを付ける。案は π_ja = 0.30。GLM の実トラフィックでの日本語は 9〜11%。
   - (ii) cold な experts を K1 に落とさない。K2 が下限である。ds41rt はもともと K1 を持たない。
   - (iii) 評価を日本語で別に行う。
   - 公開資料では、量子化による劣化は多言語・会話・作文で大きく、コードで小さい。同じパックで multilingual の KLD がコードの 2.8〜4.5 倍だった【資料: sudoingX `data/kld/*.json`】。日本語は、自動指標の劣化が 1.7% でも人手評価では 16% 落ちた例がある【資料: Marchisio et al. 2024】。
   - **元の品質を超える「向上」は、量子化の配分だけでは見込めない。** 可能性があるのは QAD (別文書) と、言語別の router 補正 (§3.4、実験的) の 2 つだけである。
7. **推奨仕様の要約 (§5)**
   - 形式: ds41rt staged manifest、k23 family。
   - 配分: experts の平均 2.19〜2.25bpw。K3 にするのは 3,000〜3,900 本で、大半は層 0〜19 に集まる。DSpark の experts は K3 にする。
   - 実装: 配分ツール、K3 パック、組み立てツールの 3 つ。
   - 工数: 計 4〜6 日。GPU は K3 パック作成で約 2.7 時間 (2 枚)、感度測定で約 3 時間 (2 枚)【推測】。

---

## 1. 前提と、今回手元で測った値

### 1.1 サイズの基礎値【計算】

| 項目 | 値 |
|---|---|
| 1 expert の重み数 | 3 × 5120 × 2304 = 35,389,440 |
| 1 expert の trellis (K2 / K3 / K4 / K5) | 8.85 / 13.27 / 17.69 / 22.12 MB (2 rank の合計。1 rank はその半分) |
| K2→K3 の 1 本あたりの増分 | 4.42MB (1 rank あたり 2.21MB = 2.11MiB)。**1 rank あたり 1GiB で約 485 本** |
| 全 experts を K2 にした場合 | ds41rt の実測で 1 rank あたり 69,122,129,920B = 64.38GiB (回転を含む、`scripts/select-release-gpus.py:36-48`) |
| 全 experts を K3 にした場合 | 203.8GB (trellis のみ) |
| suh/svh | 1 expert あたり 44.5KB。全体で約 0.68GB (QT §6) |

### 1.2 1017 の routing の偏り (エージェント型トラフィック)【実測】

- 出典: `work/teacher/agent24_teacher.pt` の `counts`。教師実行の途中経過で、17 本、計 68,308 トークン。層ごとに 409,848 回選ばれている (68,308 × 6)。
- 取り出しには `tools/r3_load_counts.py` を使った。torch も numpy も使わずに pickle を読む。
- 集計: `tools/r3_skew.py`。出力は `research/tmp/r3/skew_agent.txt`。

| 指標 (層ごとの値の平均。括弧内は最小〜最大) | 1017 agent (今回) | 0909 公開 (sudoingX 85 万トークン、QT §4.2) |
|---|---|---|
| 上位 10% の experts が持つ routing | 0.457 (0.315〜0.577) | 0.25〜0.37 |
| 上位 25% | **0.693** (0.556〜0.777) | 約 0.56 |
| 上位 50% | 0.886 (0.792〜0.937) | 0.76〜0.84 |
| 下位 30% | **0.037** (0.013〜0.088) | 0.072 |
| routing の 90% を覆う experts 数 | 201 (161〜260) | 247 |
| Gini | **0.60** (0.44〜0.69) | 0.46 |
| 1 度も選ばれない expert | 層あたり 0〜7 本 | — |

- 層による違い: 層 0〜1 が最も平ら (上位 25% が 0.56)。層 20〜23 が最も偏っている (0.74〜0.76)。
- **注意点**
  - 標本は 17 本で、同じセッション系統の要求が多い。偏りは実際より強めに出ている可能性がある。
  - diffbot の報告では、長いエージェント型セッションでは下位 30% に当たる routing が 2.4%、合成プロンプトでは 11〜24% だった (QT §4.2)。今回の 3.7% はその中間にあたる。
  - **日本語 (`work/teacher/ja16.jsonl`、16 本、61,402 トークン) の教師実行はまだ行われていない。** 日本語の routing は未測定である。

---

## 2. 課題 1: 静的な混合精度 (Unsloth Dynamic 方式)

Unsloth Dynamic 2.0 は、モデルごとに「層 (テンソル) ごとの量子化型」を変える方式である。校正データは 150 万トークン以上で、評価指標は KLD である。選択は量子化の時点で固定され、実行時には変わらない【資料: https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs 】。MoE では、この「テンソル単位」を「expert 単位」まで細かくできるかどうかが効果を決める。

### 2.1 各エンジンがどの単位まで混合 K を扱えるか (コードで確認)

| エンジン | 混合の単位 | K の範囲 | 段の数 | 根拠 (ファイル:行) | 備考 |
|---|---|---|---|---|---|
| **ds41rt** (staged manifest) | **projection 単位 (expert × w1/w3/w2)** | **K2〜K5** (整数) | release image に入っているのは **2 段 (k23 / k34)** | `rust/crates/ds41rt-loader/src/v41_exl3.rs:440-466` (projection ごとの `bits_per_weight`)、`:171-185` (`decoder_family`: 使われている K の集合。1 種類なら隣の空の段を足す)、`v41_exl3_residency.rs:197-209` (descriptor = `tier<<9 \| local`、段ごとの slab)、`native/include/ds41rt_native.h:210-221` (tier0 / tier1 の slab)、`native/cmake/v41_exl3.cmake:15-20` (family は "2,3;3,4")、`scripts/release-common.sh:149` (k23 / k34 しか許さない) | raw publication (`config.json` だけの形式) は一律 K のみで、`layer_bits` も拒否する (DR §2.1)。混合にするには `quantize_config.json` (`meta.ds41rt.schema=ds41rt.v41-routed-exl3.v1`、`tensor_storage` に 40×384×3 + 3×128×3 = 47,232 projection) を書く。**staged では DSpark の experts も EXL3 にする必要がある** (`v41_exl3.rs:141-151, 432-438`) |
| ds41rt (カーネル層、sparkinfer) | 同上 | mcg K2〜K6 (`third_party/sparkinfer/b12x/moe/_shared/trellis_codebooks.py:59-60`) | **2〜4 段**を 1 回の launch で処理できる (`mixed_trellis.py:114`、`mixed_trellis4.py:55`) | `python/tools/export_b12x_v41_exl3_aot.py:145-158`: 2〜4 段の export。3 段以上は packed routing が必須 (direct は 2 段のみ)。`v41_experts/exl3.rs:22-37`: daemon は `exl3-k{段を連結した名前}` のディレクトリを探す (例 `exl3-k234`) | k234 は **AOT パッケージを自分で build すれば使える見込み**【推測】。検証用ツール `python/tools/qualify_v41_exl3_four_tier.py` がある。release では未検証 |
| LIL b12x (`exl3-v1`) | **層単位** (manifest 1 つにつき bits 1 つ。層ごとに manifest を分ける改修が要る) | lut_e4m3 K2〜K4、mcg K3〜 | 1 | LL §2.2 (`checkpoints/exl3.py:239-240` は uniform のみ。`per_expert_pair` は Kimi の TP10 専用で P22 が無い) | expert 単位はできない。K2 には自作の量子化器が要る |
| vcruz305 vllm-exl3 (AGPL) | **expert 単位** (`p2b_fused_moe_mk`、`exl3_moe_mixedk`)、層単位 (`layer_bits`) | K1〜 (SAGE は K1〜K6) | 多段 | `ext/vcruz305_vllm-exl3/README.md:39-41, 168`、`CHANGELOG.md:23-26, 57, 82` | SM120 TP2 での実績は無い (QT §1.6)。K1 を使える唯一の経路 |
| exllamav3 (量子化器) | テンソル単位 | 1〜8 | — | 当機の量子化器 `build/qrecipe/tools/quantize_experts_exl3_sm120.py:38-46, 532` は `--bits` と `--layer-bits` (shard、すなわち層単位) を持つ。bits のガードは無い | expert 単位の K は「K2 パックと K3 パックを別々に作り、expert ごとに選んで組み立てる」ことで実現できる (§5.3)。校正なし (単位 Hessian) なので、experts は互いに独立に符号化される。組み合わせても数値は変わらない |

**判断**: expert 単位の混合を、既製のカーネルのまま実行できるのは ds41rt だけである。LIL b12x は層単位、vllm-exl3 は AGPL で SM120 TP2 の実績が無い。

### 2.2 感度指標

| 指標 | 中身 | 取得の費用 (当機) | 長所 | 短所 |
|---|---|---|---|---|
| routing 頻度 `count` | 層 × expert ごとの選択回数 | 教師実行のついでに取れる。追加費用 0 (`harness/stream_ref.py:300-303`) | 安い。実トラフィックを反映する | 出力への寄与の大きさを見ない |
| `tshare` / 重み和 | Σ routing 重み (1 トークン内で正規化) | `moe_forward` に 1 行足すだけ | sudoingX の採用値。super expert を過大評価しない (`FINDINGS.md:50-54`) | 層ごとの感度は別に要る |
| 出力エネルギー | Σ‖w·E(x)‖² | 同上 | **super expert** (層 1 で 1 本がエネルギーの 92.3%、per-token share は 1.4%) を見つけられる | これだけで配分すると、他の experts のビットが足りなくなる (同上) |
| 入力 Hessian の trace | tr(H_e) = Σ w‖x‖² | 同上 (x のノルムだけ) | 校正なし EXL3 の出力誤差は ≈ ε_K × ‖W_e‖²_F × E[w‖x‖²] に比例するので、理屈に合う | 層をまたぐ誤差の伝播は見ない |
| **層感度 α_l** | 層 l の routed 出力に相対ノイズ σ を入れたとき、最終 KL がどれだけ増えるか | 0909 の値が公開済み (sudoingX `data/alpha_sensitivity.json`: 層 0 で 4.71、層 39 で 0.019、**約 250 倍の差**)。1017 で測り直すなら、harness で層ごとに K2 へ差し替えて KL を測る (§5.4) | 層の差が最も大きい要因になる | 0909 のもので、日本語で測ったものではない |
| Fisher (E[g²]) | 損失の勾配の 2 乗 | 層ストリーミングで逆伝播が要る。harness に backward は無い | 理論上は最も直接的 | **費用に見合わない** (順伝播の 3 倍以上で、活性化の保存も要る)。採らない |
| 直接 KL 寄与 | expert (群) を K2 に差し替えたときの ΔKL | 1 構成 = 教師系列の再実行 (約 60〜100 秒/4096 トークン) | 正解そのもの | 15,360 本を個別に測るのは不可能。層単位か、super expert の候補だけに使う |

**採用する重要度**【推測。sudoingX の方式に、言語の重みと Hessian trace を加えた形】

```
I[l,e] = α_l × Σ_d π_d × S_d[l,e] / Σ_e' S_d[l,e']
  S_d[l,e] = Σ_{t∈domain d} w_t[l,e] × ‖x_t‖²     (Hessian trace 型。当面は count で代用)
  π_d   = ドメインの重み (§4.3: ja 0.30 / code 0.35 / other 0.35)
  安全項: S を (1−m)·S + m/384 で一様分布と混ぜる (m = 0.2〜0.5)
```

- 目的は、Σ I[l,e] × ε(K[l,e]) を予算のもとで最小にすることである。
- ε は sudoingX が実測した相対 MSE を使う: K2 0.026、K3 0.0042、K4 0.00058、K5 0.000096 (`plans/plan_v3_270.json`。相対誤差の中央値は K2 16.2%、K3 6.5%、K4 2.4%、`FINDINGS.md:56`)。
- ε が凸なので、1 bit あたりの改善量が大きい順に上げていく貪欲法で最適解になる。

### 2.3 VRAM の予算表 (ds41rt 0-spark の実測をもとに)【資料 + 計算】

- 根拠: DR §2.1 の起動直後の内訳。
  - RTX0: experts 64.38、KV 7.34、実行 2.44、その他 16.79GiB。
  - RTX1: experts 64.38、KV 4.90、実行 2.68、その他 19.59GiB。DSpark の MXFP4 experts 7.22GB は RTX1 側にある。
  - 1 枚の容量は 97,887MiB = 95.59GiB。余裕として 0.8GiB を残す。
- KV は auto で残りを埋める設計なので、KV の値は「最低限必要な量」に置き換えて計算した。
- **TP2 では、どの expert も 2 枚に半分ずつ載る。したがって追加できる量は、空きの少ない方の GPU (多くの場合 RTX1) で決まる。**

| KV/枚 | DSpark experts | RTX1 の追加枠 (GiB) | RTX0 の追加枠 (GiB) | K3 にできる本数 | 平均 bpw | experts 合計 | 備考 |
|---|---|---|---|---|---|---|---|
| 4.90 (現行) | MXFP4 7.22GB | 3.24 | 7.18 | 1,571 (10%) | 2.10 | 145GB | raw のままでは混合にできない。参考値 |
| 4.0 | MXFP4 | 4.14 | 8.08 | 2,008 (13%) | 2.13 | 147GB | staged では DSpark の EXL3 化が必須なので、この組み合わせは成り立たない |
| 4.0 | **K3 (5.1GB)** | 6.11 | 8.08 | 2,963 (19%) | **2.19** | 151GB | **推奨の基準** |
| 3.0 | K3 | 7.11 | 9.08 | 3,448 (22%) | 2.22 | 153GB | |
| 2.0 | K3 | 8.11 | 10.08 | 3,933 (26%) | 2.26 | 155GB | KV が小さく、1M 文脈を何本も保持できない (DR §10 では 1M で約 12GiB と推測) |
| 2.0 | K2 (3.4GB) | 9.84 | 10.08 | 4,770 (31%) | 2.31 | 159GB | DSpark の受理率が落ちる危険がある (sudoingX は drafter の experts を K3 にしても受理率は保たれたと報告、`FINDINGS.md:63-65`) |

- 本数の計算式: 本数 = 追加枠 (GiB) × 2^30 / 2.21MB。例: 6.11GiB × 485 本/GiB ≈ 2,963 本。
- **KV を 1GiB/枚 減らすと、K3 にできる本数が約 485 本 (平均 +0.032bpw) 増える。** 文脈の容量と品質の交換になるので、§6 でユーザーが決める。
- 共有 expert (FP8)、attention、head、Engram は ds41rt の契約で公式の精度に固定されている (`v41_exl3.rs:250-262`)。課題文の「共有専門家は高精度」は、何もしなくても満たされる。

### 2.4 配分のシミュレーション (代理損失)【計算 + 推測】

- スクリプト: `tools/r3_alloc.py`。入力は §1.2 の 1017 agent の回数と、0909 の α_l。損失 = Σ I × ε(K) で、一律 K2 を 1.000 とした。
- この代理損失は「誤差が足し算で効き、MSE が KL に比例する」と仮定している (sudoingX と同じ仮定)。
  - 目安の換算: 一律 K3 の 0.16 に diffbot の一律 K2 の KL 0.36 を掛けると 0.058 になり、校正あり 3.0bpw の公開値 0.0545 (QT §5) とほぼ合う。偶然の可能性もあるので、あくまで目安である。

**予算別 (計画と評価に同じ agent 分布を使った場合。出力は `research/tmp/r3/alloc_out.txt`)**

| 平均 bpw | experts | 層別 K2/K3 | **expert 別 K2/K3** | expert 別 K2/K3/K4 | (換算 KL の目安【推測】) |
|---|---|---|---|---|---|
| 2.10 | 142.7GB | 0.774 | 0.545 | 0.543 | 約 0.20 |
| 2.15 | 146.1GB | 0.678 | 0.452 | 0.447 | 約 0.16 |
| 2.20 | 149.5GB | 0.594 | **0.383** | 0.375 | 約 0.14 |
| 2.25 | 152.9GB | 0.518 | 0.331 | 0.319 | 約 0.12 |
| 2.50 (参考、VRAM に入らない) | 169.9GB | 0.271 | 0.200 | 0.157 | — |
| 一律 K3 (参考) | 203.8GB | 0.162 | | | 約 0.058 |

- expert 別は、層別より約 35% 損失が小さい。公開事例の傾向 (Krishando: expert 別 2.41bpw の方が層別 2.63bpw より KL が小さい、QT §1.5) と合う。
- K4 を混ぜても、2.2bpw の予算では 0.383 が 0.375 になるだけである。理由は、K3→K4 の 1 bit あたりの改善 (0.0036) が K2→K3 (0.0218) の 1/6 しかないためである。k234 の AOT を自分で build する価値は小さい。

**分布がずれたときの頑健性 (`research/tmp/r3/robust_out.txt`、2.20bpw)**

| 計画に使う分布 (m = 一様分布を混ぜる割合) | 方式 | agent で評価 | agent と一様の半々で評価 | 一様で評価 |
|---|---|---|---|---|
| m=0 | 層別 K2/K3 | 0.594 | 0.594 | 0.594 |
| m=0 | expert 別 K2/K3 | 0.383 | 0.534 | 0.685 |
| m=0.2 | expert 別 K2/K3 | 0.385 | 0.528 | 0.671 |
| m=0.5 | expert 別 K2/K3 | 0.403 | 0.522 | 0.640 |

- **計画の分布から外れたトラフィックでは、expert 別配分の利点が縮む。完全に外れると、層別より悪くなる。**
- 日本語の routing が agent とどれだけ違うかは未測定である。そのため、**日本語の回数を計画に入れること (§4) が品質を左右する。**
- 安全項 m は 0.2〜0.5 が妥当と見る。手元の分布での損失は +0.5〜5% 増える代わりに、外れたときの劣化が 2〜7% 減る。

**K3 が置かれる場所 (2.20bpw、m=0.2、`dec_out.txt`)**

| 層 | 0〜9 | 10〜19 | 20〜29 | 30〜39 |
|---|---|---|---|---|
| K3 の本数 (384 本中) | 316, 261, 209, 180, 228, 205, 141, 158, 166, 138 | 156, 115, 114, 86, 85, 62, 64, 51, 57, 88 | 29, 33, 20, 17, 20, 17, 11, 11, 15, 8 | 4, 4, 1, 0, 1, 0, 1, 0, 0, 0 |

- K3 は層 0〜19 (CED の encoder 側。prefill でも全トークンで計算される層) に集まる。α_l の 250 倍の差がそのまま出ている。
- 層 37〜39 は DSpark の入力を取る層である。sudoingX は 37〜39 に K3 の下限を設けた (`plan_v3_270.json` の `floor_layers`)。ここで 3 層すべてを K3 にすると 5.1GB かかり、0.075bpw に相当する。**τ (DSpark の受理長) を測ってから決める。**

**読み出し量 (速度) への影響 (`bytes_out.txt`)【計算 + 推測】**

| 2.20bpw の配分 | 1 step のトークン数 B=6 (c1) | B=64 (c16) | B=96 |
|---|---|---|---|
| 1 層あたりに触れる experts (偏りを考慮) | 32 | 170 | 206 |
| expert の読み出しバイト (一律 K2 との比) | ×1.22 | ×1.17 | ×1.16 |

- ホットな experts を K3 にするので、平均 bpw (2.20) よりも読み出し量の方が大きく増える。
- c1 では 1 round (約 23ms) のうち、expert の読み出しは約 4.9ms (5.7GB/枚 ÷ 1.45TB/s × 1.25)。増加は +1.1ms で、**c1 は約 5% 落ちる**【推測】。
- c16 では 1 round (約 39ms) のうち約 26ms が expert の読み出しで、増加は +4.4ms。**c16 は約 10% 落ちる**【推測】。
- 偏りのおかげで、B=64 で触れる experts は一様の場合の 244 本から 170 本に減っている。PLAN の c16 の見積もりは、むしろ悲観的だった。
- 速度を重視する場合は、sudoingX の `--decode-bpw` と同じ「routing で重み付けした bpw の上限」を制約に加える (例 ≤2.45)。

### 2.5 静的配分の推奨

- **ds41rt staged、k23 (K2/K3)、平均 2.19bpw** (KV 4GiB/枚、DSpark の experts は K3、K3 は約 2,960 本)。
  - KV を 3GiB/枚まで削れるなら、2.22bpw (約 3,450 本) にする。
- super expert の保護: 層ごとに、出力エネルギーが最大の expert (層 1 では 92%) を必ず K3 にする。k23 の上限が K3 なので、sudoingX の K5 は使えない。
  - 1 層に 1〜2 本、計 40〜80 本なので、予算への影響は小さい。
  - エネルギーは harness で取る (§5.4)。
- 層 37〜39 の下限は、τ を測ってから決める。
- 計画の入力: 1017 の agent + ja + code の回数 (§4)、α_l、安全項 m=0.3。

---

## 3. 課題 2: 実行時に動的な方式

### 3.1 (a) 2 段の専門家表現 (全 experts を低ビットで常駐させ、ホットな experts の高ビット版を差し替える)

**帯域と容量【計算】**

| 項目 | 値 |
|---|---|
| PCIe H2D | 約 20GB/s/枚 × 2 枚。TP2 なので各 rank が自分の半分を受け取る → 合計約 40GB/s |
| K3 の全体版 1 本 (2 rank の合計) | 13.27MB → 0.33ms (40GB/s) |
| c1 の 1 round | 約 23ms (C1 code 225 tok/s、1 round 5.19 トークン、DR §5) |
| c1 の 1 round で触れる expert | 32 本 × 40 層 = 約 1,300 |
| 1 round の中で差し替える場合 (触れる expert の 5% を高ビットに入れ替える) | 65 本 × 13.27MB = 0.86GB → 21ms。**1 round の時間がほぼ倍になる** |
| 背景で PCIe の 10% を使う場合 | 1 round あたり 92MB = K3 で 7 本。**約 300 本/秒** |
| ホスト RAM に置ける高ビット版 | RAM 128GB の内訳: Engram のページキャッシュ 48〜64GB (EH §0-4)、エンジン 15GB、OS 8GB、pinned 4〜16GB。残りは約 40GB → K3 版で 3,000 本、残差版 (8.85MB) で 4,500 本 |
| VRAM 上のプール | §2.3 の追加枠と同じ 13〜17GB。K2 版を残したまま高ビット版を**重複して持つ**と、1 本 13.27MB → **1,000〜1,280 本**にしかならない |

- **1 token や 1 step の単位で差し替える方式 (HOBBIT のように、キャッシュに無い expert を低精度版で代用する方式) は成り立たない。**
  - HOBBIT はオフロード環境で、読み込み待ちを減らすために低精度版で代用する (https://arxiv.org/abs/2411.01433 )。
  - 当機は全 experts が VRAM に常駐しているので、この方式で得るものが無い。
- 成り立つのは、**秒〜分単位でホットな集合を入れ替える**方式だけである。ホットな集合の推定には、直近の窓の routing 回数の指数移動平均を使う。
- **静的配分との比較 (`dyn_out.txt`、同じ 17GB の枠で、計画と評価が同じ分布という理想の条件)**

| 方式 | 本数 | 代理損失 |
|---|---|---|
| 静的 K2→K3 (その場で置き換え) | 3,840 | **0.331** |
| K3 を重複して持つ動的プール | 1,280 | 0.584 |
| K2 の残差を加える動的プール (§3.2) | 1,920 | 0.414 |

- 動的方式は、理想どおりに追従できても静的方式に負ける。勝てるのは、静的計画の損失が分布のずれで 0.41〜0.58 を超える場合だけである。§2.4 の「agent と一様の半々」程度のずれでは静的が 0.53 なので、残差方式とほぼ互角になる。
- **ds41rt での実装の可否【資料 + 推測】**
  - descriptor (`tier<<9 | local`) は device 上のバッファなので、書き換えれば CUDA graph の再捕捉なしに段を切り替えられる見込みである (`v41_exl3_residency.rs:197-209`)。
  - ただし slab は段ごとに固定長で、ロード時に確保される。段を移すには「tier1 の slab に空きスロットを持たせる」改修が要る。
  - **K2 版を消して空きを作る方式は、可変長の slot を持たない現在の形式ではできない。** 結果として、重複方式になる。
  - 工数は 2〜3 週【推測】。

### 3.2 (b) 残差補正 (低ビットの本体 + ホットな experts だけ高精度の差分)

- 形: Ŵ = Q_K2(W) + Q_K2(W − Q_K2(W))。
  - 相対誤差は約 16.2% × 16.2% ≈ 2.6% で、K4 (2.4%) に近い【推測。Hadamard 回転の後では残差もほぼガウス分布になると仮定】。
  - 残差を 1 本足すのに 8.85MB (2 bit 分) かかる。K4 版を丸ごと持つのと同じバイト数である。
- **非線形 (SwiGLU) の前後で、足し方が違う。**
  - w1 / w3 の残差は、活性化の**前**に足す必要がある (silu(min(W1x+ΔW1x, 10)))。別の expert として出力側で足すことはできない。FC1 のタイル内で 2 本の trellis を復号して足す形 (加法量子化と同じ形) の、カーネル改修が要る (sparkinfer の `mixed_trellis.py` の復号器)。
  - w2 の残差は線形なので、FC2 で足すだけで済む。
  - base と残差で suh / svh の回転を共有すれば、回転は 1 回で済む。
- 利点: 本体の K2 を書き換えずに、残差だけを付け外しできる。§3.1 の動的プールと組み合わせる形になる。
- 1 bit あたりの効き: 残差は (0.026 − 0.0007) / 2bit = 0.0127/bit で、静的な K2→K3 (0.0218/bit) より 4 割悪い。**静的配分の後に、分布のずれが大きいと実測で分かった場合だけ検討する。**
- 工数: カーネル (CuTe DSL、AOT の再 export)、ローダ、プール管理で 3〜4 週【推測】。

### 3.3 (c) 活性化の動的量子化 (per-token FP8)

- **現状【資料】**
  - 公式の非 expert 層は、活性化を 1×32 / UE8M0 の動的 FP8 にする (`activation_scheme=dynamic`、`ref/model.py:187, 197`)。ds41rt の契約もこれと同じである (`v41_exl3.rs:250-262`)。
  - 公式の experts も、MXFP8 の活性化を使う W4A8 である (AS §6)。
  - ds41rt の EXL3 経路は W4A16 (trellis を FP16 に復号し、活性化は BF16/FP16) で、**参照実装より精度が高い側**にある。
  - KV は契約どおり FP8 / FP4 の形式である (DR §7)。
- **A8 にした場合の影響【推測】**
  - 品質: 1×32 ブロックなら参照実装と同じ量子化になる。行ごとに 1 つの scale にすると、mHC の残差や super expert の大きな出力 (外れ値) で誤差が増える。**行ごとの単一 scale は採らない。**
  - 速度: prefill の FC1 / FC2 を FP8 の mma にすれば、計算律速の区間 (prefill 5.6k tok/s) が 1〜2 割速くなる余地がある。decode は帯域律速なので変わらない。
  - 日本語: 参照実装と同じ量子化であれば、特定の言語に偏った影響は考えにくい。
- 判断: 品質の目的では行わない。prefill の速度が課題になったときに、sparkinfer の W4A8 trellis 経路 (LL §1.1 #13) を検討する。

### 3.4 (d) トークンや言語に応じた精度の切り替え

| 方式 | 中身 | 可否と費用 | 判断 |
|---|---|---|---|
| d1: routing による暗黙の切り替え | 日本語のトークンは日本語でよく使われる experts に routing される。その experts を K3 にしておけば、**日本語のトークンだけが自動的に高精度になる** | §2 の静的配分に日本語の回数を入れるだけ。追加の実行費用は 0 | **採用** |
| d2: 要求単位で計画を切り替える | 「日本語重視」と「コード重視」の 2 つの snapshot を作り、運用で切り替える | ds41rt の起動は 17 秒 + ページキャッシュの温め。ディスクは 1 版 150GB | 両方の負荷が混ざる運用には向かない。保留 |
| d3: 言語別の router bias (`bias_ja`) | 画像トークンには別の bias (`bias_vl`) を使う仕組みが既にある (`ref/model.py:820-822`、ds41rt `native/cuda/kernels/v41_router.cu`、`rust/crates/ds41rt-ffi/src/v41_router.rs:49`)。これを日本語トークンにも使い、中間層の routing を英語の task experts に寄せる。または K3 の experts に寄せる | 文献では、中間層の routing を英語に寄せると多言語の性能が 1〜2% 上がった (Qwen3-30B-A3B ほか 3 モデル、https://arxiv.org/abs/2510.04694 )。先頭と末尾の層に介入すると悪化した。ds41rt の改修 (トークンの言語判定と bias の選択) は約 1 週。探索と評価に GPU 数時間【推測】 | **実験扱い**。V4.1 で効く保証は無い。bias は「選ぶ」だけで重みは変えない (`ref/model.py:824`) ので、副作用の範囲は限られる。ただし 1 層あたり 384 本の bias の探索になる |
| d4: トークン単位で残差を付け外し | 難しいトークンだけ §3.2 の残差を通す | 残差カーネル + 判定器 | 費用に見合わない |

---

## 4. 課題 3: 日本語のトークンは別の experts を使うか

### 4.1 文献と公開データ【資料】

| 出典 | 内容 | V4.1 への含意【推測】 |
|---|---|---|
| Bandarkar et al., "Multilingual Routing in Mixture-of-Experts" (ICLR 2026, https://arxiv.org/abs/2510.04694 ) | Qwen3-30B-A3B、Phi-3.5-MoE、GPT-OSS-20B、OLMoE で調べた。**routing は先頭と末尾の層で言語ごとに分かれ、中間層では言語をまたいで揃う**。言語の性能は、その言語の routing が英語にどれだけ近いかと相関する | 先頭の層に日本語専用の experts があるとすると、そこは V4.1 で α_l が最も大きい層 (層 0 で 4.71) と重なる。**日本語専用の experts が agent の頻度で K2 に落とされると、損失が最も大きい場所で日本語が削られる** |
| Chen et al. (https://arxiv.org/abs/2601.14050 ) | routing は語族に沿って似る。高資源の言語は共有の experts に、低資源の言語は専用の experts に頼る | 日本語は DeepSeek では高資源と見られ、共有寄りと考えられる。ただし中国語と漢字を共有するので、CJK の専用 experts が存在する可能性がある |
| OpenMoE (https://arxiv.org/abs/2402.01739 ) | ja と ko が同じ expert (E14) を好み、zh-cn と zh-tw は E5 / E16 を好む。プログラミング言語には特化が見られなかった | 細かい experts の DeepSeek でも、CJK のまとまりができる可能性 |
| Koishekenov et al. (NLLB-200、https://arxiv.org/abs/2212.09811 ) | 言語別の expert を刈り込み指標で特定できた。80% を刈っても翻訳品質の低下は小さい | 言語専用の experts は実在し、指標で見つけられる |
| Marchisio et al. (EMNLP Findings 2024、https://arxiv.org/abs/2407.03211 ) | 量子化の影響は言語で差があり、**非ラテン文字の言語ほど大きい**。日本語は自動指標で −1.7% のとき、人手評価で −16.0% だった | 自動指標の KL だけで合格を決めると、日本語の劣化を見落とす |
| sudoingX の KLD パネル (`ext/sudoingx_dsv41/data/kld/*.json` の `by_domain`、V4.1 0909) | 同じパックでのドメイン別 KLD: 自作 2.70bpw は multilingual 0.069 / chat 0.061 / writing 0.058 / code 0.015。Mia 2.9bpw は 0.058 / 0.069 / 0.068 / 0.020。0xSero 2.68bpw は 0.089 / 0.090 / 0.070 / 0.021 | **multilingual の KLD はコードの 2.8〜4.5 倍**。V4.1 でも、量子化で最も損をするのは多言語、会話、作文である |
| sudoingX `FINDINGS.md:57-62` | 校正データで multilingual が 5% しか無かったとき、multilingual の KLD が Mia の 1.34 倍になった。30 言語の補充で解消した | 校正つき (E3-2) に進むときは、日本語を校正データに入れることが必須 |
| diffbot AtomicChat (QT §5) | 一律 2.0bpw の KL: neutral 0.361 / code 0.215 / agentic 0.077 | 散文は、エージェント型の約 5 倍劣化する |
| EH §0-6 (当機の X2) | 日本語は 8k チャンク内で重複しにくく (unique 0.79、コードは 0.65)、Engram の IO が多い | Engram を FP8 のまま残す判断は、日本語の知識の保持にも効く |

- **DeepSeek V4.1 について、言語別の routing 統計を公開したものは見つからなかった** (検索の範囲で)。手元で測る必要がある。

### 4.2 手元での測り方 (GPU が空いたら、追加の実装はほぼ不要)

- harness は 1 回の実行ごとに `STORE.counts` を保存する (`stream_ref.py:472`)。したがって、**`ja16.jsonl` を別の教師実行にすれば、日本語の routing 回数がそのまま得られる** (16 本、61,402 トークン。所要は約 15〜25 分)。
- 比べる量 (`tools/r3_skew.py` を 2 ファイル比較用に拡張する)
  - 層ごとの分布の距離: Jensen-Shannon 距離、上位 25% の集合の Jaccard 係数。
  - 日本語の上位 experts が、agent の配分で K2 になってしまう割合。
  - 層ごとに見て、先頭と末尾の層で分かれ、中間層で揃うという Bandarkar の形が V4.1 でも出るか。
- コードと英語の散文も同じ方法で分けて取る。agent24 は tool JSON と英語の推論文とコードが混ざっているので、x2 と同じ 256 トークン窓の分類 (CJK と仮名が 25% 以上なら ja) で分けてもよい。その場合は `moe_forward` にドメインのラベルを渡す改修が要る (約 30 行)。

### 4.3 日本語に寄せた配分の効果の見込み【推測】

- 重み π_d の案: **ja 0.30 / code 0.35 / other (英語の推論文、tool JSON、エージェント) 0.35**。
  - GLM の実トラフィックでは、日本語は 9% (課題文)、Engram 対象区間の分類では ja 11%、code 49%、other 40% だった (EH §1.2)。
  - ユーザーが日本語を重視するので、実際の比率の約 3 倍にした。π は 1 行の設定で変えられる。
- 効果の幅 (§2.4 の頑健性の表から)
  - 日本語の routing が agent と「半々」程度ずれている場合: agent だけで計画すると、日本語の損失は約 0.53 になる。日本語を計画に入れると 0.38〜0.40 になり、**日本語の損失は 25〜30% 減る**。
  - ずれがさらに大きい場合は 0.67 → 0.40 で、約 40% 減る。
  - agent 側の代償は +3〜5% (0.383 → 0.40) にとどまる。
  - 一律 K2 の日本語の KL を 0.4〜0.5 と置くと【推測。neutral 0.36 に多言語の倍率を掛けた値】、日本語の KL は約 0.25 → 約 0.17 になる。
- **cold な experts を K1 に落とす危険 (`k1_out.txt`)**
  - K1 を許すと、agent の分布では損失が 0.38 → 0.24〜0.28 と大きく下がる。仮定した K1 の ε は 0.10〜0.25。
  - しかし「agent で cold な experts」は、まさに日本語専用の experts である可能性がある。
  - さらに ds41rt と sparkinfer には K1 が無い (K2〜K6)。**K1 は採らない。**
  - vllm-exl3 の経路 (SAGE 1.59bpw の K1) を選ぶときは、必ず日本語の回数を入れてから判断する。

---

## 5. 課題 4: 推奨する動的量子化の仕様

### 5.1 全体像

```
[harness 教師実行]  domain ごとの counts / wsum / energy / x²  →  routing_stats.v1
[harness 層感度]    層ごとに K2 へ差し替えたときの ΔKL (ja / code / agent)  →  alpha.v1
         │
         ▼
[r3_alloc (配分器)]  予算 (bytes/rank)、段 [2,3]、π、m、下限、super expert の保護、decode-bpw の上限
         │  → mixk_plan.v1.json (層 × expert の K 文字列)
         ▼
[K2 パック (作成中)] + [K3 パック (候補の experts だけ、または全体)] + [DSpark K3]
         │
         ▼
[assemble_staged (CPU)]  expert ごとに K2 か K3 のテンソルを選んで shard を書き直す
         │  → ds41rt staged snapshot (config.json / quantize_config.json / index / shards)
         ▼
[ds41rt 0-spark]  exl3-k23/rtx-tp2 (release image のまま)、KV を明示して起動
```

### 5.2 データ構造

**(1) `routing_stats.v1` (npz または json。harness が出力する)**

```json
{"schema":"dsv41.routing-stats.v1","model":"DeepSeek-V4.1-Flash-1017","source_sha256":"…",
 "domains":{"ja":{"tokens":61402,"files":["ja16.jsonl"]},"agent":{"tokens":95126},"code":{…}},
 "per_domain":{"ja":{"count":[[384 個 int]×40],"wsum":[[float]×40],"energy":[[float]×40],"xsq":[[float]×40]}, …},
 "mtp":{"count":[[128]×3]}}
```

- harness の改修は `moe_forward` (`stream_ref.py:295-311`) に約 30 行で済む【推測】。
  - `weights` の和と ‖w·E(x)‖² と ‖x‖² を expert ごとに足し込む。
  - ドメインのラベルは実行の単位で付けるか、256 トークンの窓ごとに付ける。

**(2) `alpha.v1`**

```json
{"schema":"dsv41.alpha.v1","method":"swap-layer-to-K2","teacher":"…pt","alpha":{"ja":[40 個],"agent":[40 個],"mix":[40 個]}}
```

**(3) `mixk_plan.v1.json` (配分器の出力)**

```json
{"schema":"dsv41.mixk-plan.v1","family":[2,3],"budget_bytes_per_rank":…,"kv_gib_per_rank":4.0,
 "pi":{"ja":0.30,"code":0.35,"other":0.35},"uniform_mix":0.3,
 "protect":{"super_expert_per_layer":1,"floor_layers":{}},"decode_bpw_cap":2.45,
 "layers":{"0":"2332…(384 文字)", …, "39":"…"},          // expert ごとの K (w1 / w3 / w2 で共通)
 "mtp":{"0":"333…(128)","1":"…","2":"…"},
 "predicted":{"proxy_cost":0.38,"avg_bpw":2.19,"decode_weighted_K":2.47},
 "inputs":{"routing_stats_sha256":"…","alpha_sha256":"…"}}
```

- 形は sudoingX の `exl3_moe_plan` (層ごとの K の文字列) とほぼ同じにした (`data/plans/plan_v3_270.json`)。
- projection ごとに K を変える拡張 (例えば w2 だけ K3) は、ds41rt では表せる。ただし感度のデータが無いので、v1 では expert 単位で揃える。

**(4) ds41rt staged snapshot (組み立て器の出力。ds41rt の契約、`v41_exl3.rs:344-438`)**

- `config.json`: `quantization_config` は compact 版で、`quantize_config.json` のキーの部分集合でなければならない (`:376-385`)。
- `quantize_config.json`
  - `quant_method / method / format / checkpoint_format = "exl3"`
  - `codebook: "mcg"`、`out_scales: "never"`、`group_size: -1`、`desc_act: false`、`pack_dtype: "int32"`、`bits: 2`
  - `meta.ds41rt = {schema: "ds41rt.v41-routed-exl3.v1", tensor_naming: "checkpoint-native", native_quantization_config: {公式の FP8 ブロック}}`
  - `tensor_storage[<proj>] = {quant_format: "exl3", bits_per_weight: K, stored_tensors: {<proj>.trellis: {torch_dtype: "int16", shape: [in/16, out/16, 16K]}, .suh, .svh, .mcg}}`
  - 対象は 47,232 projection (DSpark の 3×128×3 を含む)。
- `model.safetensors.index.json` と `model-0000i-of-0000N.safetensors`
  - **shard の名前は 1..N の連番でなければならない** (`v41_catalog.rs:542-559`)。
  - そのため、2 つのパックから hardlink で組み合わせることはできない。expert のテンソルを含む shard は書き直す必要がある。
  - experts を含まない shard (Engram の 47/48 を含む) は hardlink のままでよい。

### 5.3 実装箇所

| # | 作業 | 場所 | 規模【推測】 |
|---|---|---|---|
| 1 | harness で routing の統計を集める (wsum / energy / xsq とドメイン) | `harness/stream_ref.py:295-311, 472` | 約 30〜50 行、半日 |
| 2 | harness で層感度を測る。量子化する層を絞る `--exl3-layers` と、層ごとの ΔKL を記録する処理 | `stream_ref.py:179-192` (`_is_exl3` に層の条件を足す)、`teacher()` | 約 30 行、半日 |
| 3 | 配分器。ドメインの混合、安全項、super expert の保護、下限、decode-bpw の制約。出力は plan.v1 | `tools/r3_alloc.py` を拡張して `tools/mixk_plan.py` にする | 1 日 |
| 4 | K3 パックを作る。全体なら `--bits 3` で 40 層。候補の experts だけにするなら、`--expert-filter plan.json` を足す | `build/qrecipe/tools/quantize_experts_exl3_sm120.py` (`convert_shards`、`:376-470`) | フィルタは約 20 行。実行は §5.5 |
| 5 | DSpark の experts を K3 にする | diffbot の `quantize_mtp_exl3.py` (QT §1.2-4) | 実行は 1 GPU で約 9 分 |
| 6 | **staged 組み立て器** (K2 / K3 パックと plan から snapshot を作る) | 新規 `tools/assemble_ds41rt_staged.py` | 1〜1.5 日。CPU で、読み 150GB と書き 150GB → NVMe で 2〜5 分 |
| 7 | 契約の検査を ds41rt のローダと同じ規則で行う (projection 数、shape、byte 長、compact のキーの一致) | 組み立て器に含める。ds41rt の `--dry-run` でも確かめる | 半日 |
| 8 | ds41rt を起動する。KV を明示し、`exl3-k23/rtx-tp2` を使う | `run-local.sh` (既存)、`local.config` | 設定のみ |
| 9 | 評価: KL と top-1 をドメイン別に、日本語 QA、tool eval、τ | harness の学生モード (`--exl3-pack <snapshot>`)、ds41rt の qualify スクリプト | 既存の手順 (DECISION §4) |

- 任意 (効果は小さい): k234 の AOT を自作する。`python/tools/package_v41_exl3_aot.py --bits 2 3 4` に coordinator の GPU の build 環境が要る。3 段では packed routing になり、速度は未検証。
- 採らないもの: §3.1 の動的プール (2〜3 週)、§3.2 の残差カーネル (3〜4 週)。静的配分を評価した後で、セッション内の routing のずれを実測してから判断する。

### 5.4 GPU 時間とディスクの見積もり【推測】

| 作業 | GPU | 時間 | ディスク |
|---|---|---|---|
| ja16 の教師と counts (#1 の後) | 1 枚 | 15〜25 分 | 数 MB |
| 層感度: 8 群 (5 層ずつ) × 3 ドメイン × 各 4 本 (約 1.6 万トークン)。1 群あたり教師 12 本 ≈ 12〜20 分 | 2 枚 | 約 1.5〜3 時間 (2 枚) | 数 MB |
| K3 パック (全 40 層)。K2 は 1 層あたり約 4 分/GPU (課題文)。K3 は greedy の候補が 2 倍なので約 8 分 | 2 枚 | 約 2.7 時間 | 約 205GB |
| K3 パック (候補の experts だけ。各層の上位 40%) | 2 枚 | 約 1.1 時間 | 約 82GB |
| DSpark の K3 | 1 枚 | 約 9 分 | 5.1GB |
| 組み立て (CPU) | 0 | 2〜5 分 | 約 150GB |

- ディスク: 現在の空きは 475GB (`df -h /`、2026-10-18 10:2x)。K2 の作成が終わると約 400GB になる見込み【推測】。
  - K3 の全体版 (205GB) と snapshot (150GB) の両方を置くと、残りは約 45GB で、ぎりぎりになる。
  - **K3 は候補の experts だけにする (82GB)** のが安全である。候補は「どのドメインでも上位 40% に入る experts の和集合」とする。
  - 組み立て後は、K2 パックと K3 の候補パックを残しておけば、π の変更や再配分を GPU なしで 5 分以内にやり直せる。これを「オフラインの動的再配分」として、ユーザーの日本語重視の調整に使う。

### 5.5 工程

1. (CPU、今すぐ) #1〜#3 と #6〜#7 の実装。
2. (GPU、教師実行の空き時間に) ja16 の教師を回し、counts を取る。agent と ja の routing の違いを §4.2 の方法で数値化する。
3. (GPU、K2 の作成後) K3 の候補パックと DSpark の K3 を作る (2 枚で約 1.2 時間)。
4. (GPU) 層感度を測る (約 2 時間)。
5. (CPU) plan を作って組み立てる。第 1 版は π = {0.30, 0.35, 0.35}、m=0.3、KV 4GiB、2.19bpw。比較用に「agent だけの計画」も作る。
6. (GPU) 評価する。基準は E3-0 (一律 K2)。DECISION の G1 にドメイン別の判定を足す (§6)。
7. 合格したら ds41rt で速度を測る。c1 −5%、c16 −10% 程度の見込みが当たるかを確かめる。

**合計の工数: 実装 3〜4 日 + GPU 約 4〜5 時間 (2 枚)、全体で 4〜6 日【推測】。**

### 5.6 他の文書との関係 (1 行ずつ)

- QAD: 静的配分で決めた K2 / K3 の構成を固定したまま、教師 logits (harness) に対して router の bias や experts の scale を蒸留する形にすると、§3.4 d3 の言語別 bias と同じ枠組みに入る。
- TensorFold: ragged な expert 別形式 (QT §1.6) である。本文書の `mixk_plan.v1` は、TensorFold の配分の入力にも流用できる形にしてある。

---

## 6. 判断が要る点と、次に測るもの

1. **KV と品質の交換 (§2.3)**: KV を 4GiB/枚 (2.19bpw) とするか、3GiB (2.22bpw) とするか、2GiB (2.26bpw) とするか。1M 文脈を何本同時に保持するかで決まる。
2. **日本語の重み π_ja**: 案は 0.30。日本語の合格線の案は次のとおり。
   - 日本語の top-1 一致の低下が、コード・エージェントの低下の 1.5 倍以内。
   - 日本語 QA の低下が 3 点以内 (DECISION G1 と同じ)。
   - Marchisio の結果があるので、人手での確認 (日本語の自由記述 20 問程度の比較) も入れる。
3. **速度と品質の交換**: c16 が約 10% 落ちる見込みを許すか。許さない場合は decode-bpw の上限を 2.35 前後にする。
4. 未測定の項目 (推測が多い箇所)
   - 1017 での日本語の routing (ja16 の教師実行)。
   - 1017 での α_l (現在は 0909 の値を流用)。
   - super expert が 1017 のどの層・どの番号か。
   - K2/K3 混合での τ (DSpark の受理率)。特に、層 37〜39 を K2 のままにしたとき。
   - 代理損失から実際の KL への換算 (第 1 版の評価で係数を当てはめる)。
   - セッション内の routing のずれ (§3 の動的方式に進むかを決める材料)。

---

## 参考文献 (Web)

- Unsloth Dynamic 2.0: https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- Bandarkar et al., Multilingual Routing in Mixture-of-Experts: https://arxiv.org/abs/2510.04694
- Chen et al., Understanding Multilingualism in MoE LLMs: https://arxiv.org/abs/2601.14050
- OpenMoE: https://arxiv.org/abs/2402.01739
- Koishekenov et al., Language-specific Expert Pruning of NLLB-200: https://arxiv.org/abs/2212.09811
- Marchisio et al., How Does Quantization Affect Multilingual LLMs?: https://arxiv.org/abs/2407.03211
- HOBBIT, Mixed Precision Expert Offloading: https://arxiv.org/abs/2411.01433
