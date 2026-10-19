# R2: 1017 の routed experts を 2 GPU の VRAM (約 150GB) に収める量子化・刈り込みツールの調査

調査日時: 2026-10-18 03:50〜 JST (HF の再スキャンは 03:52 JST = 2026-10-17 18:52Z 時点)
対象: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (rev `1bae0e10`)
前提: `research/PLAN.md` §0 / §2.3 案 A / §4 X6・X7、`research/web_engines.md` §3・§4
方針: GPU は使っていない。glm53v には触れていない。重みはダウンロードしていない (HF API、README、git clone のみ)。
凡例: 【事実】は出典あり、【計算】は式から、【推測】は根拠が間接的なもの。

取得物の置き場 (すべて `` 以下)

- clone: `ext/exllamav3` (本家 dev `448f5ea7` v1.6.3)、`ext/vcruz305_exllamav3`、`ext/vcruz305_vllm-exl3` (`2a740a3`、壊れていた clone を取り直した)、`ext/sfxnz_recipe` (`3de9146`)、`ext/sudoingx_dsv41` (`cc59c36`)、`ext/libertai_reap` (`210cceb`)、`ext/exllamav3_anoane_dsv41f` (exllamav3 PR #449 のブランチ `c341a8f`)
- HF の再スキャン結果: `research/tmp/r2q/scan0352/` (検索結果 `s_*.json`、個別 API `info/*.json`、カード `cards/*.md`、GitHub README `gh/*.md`)
- 第 1 ラウンドの取得物: `research/tmp/r2q/{cards,info,diffbot}/`
- routing 偏りの集計スクリプト: `tools/r2_routing_skew.py`。出力は `research/tmp/r2q/skew_count.txt` と `skew_tshare.txt`

---

## 0. 結論

1. **2x96GB の VRAM に experts を全部載せられる (experts ≤150GB) 1017 の既製量子化物は、まだ無い。**
   - 2026-10-17 18:52Z の再スキャンでは、1017 の EXL3 は次の 4 本だけだった。
     - LibertAIDAI 2.9bpw (210.7GB)
     - LibertAIDAI 3.5bpw (257.6GB。新着)
     - PixelML 3bpw (223.0GB。新着、校正なし)
     - LakyFx 3.5bpw (アップロード途中)
   - 1017 で 150GB を下回るのは次の 2 本だけで、どちらもこの機械では使えない。
     - Baekpica VQ (約 1.6bpw)。GB10 専用エンジン向けで、KL も 0.42〜0.69 と悪い。
     - Lucebox ROCMFP2S (2bit、experts 170GB)。AMD 専用。
   - **したがって自分で量子化する必要がある** (§3)。
2. **量子化の道具は揃っている。本家 exllamav3 だけでは V4.1 を変換できないが、V4.1 に対応した外部ツールが 3 系統ある。**
   - Q1: **sfxnz `quantize_experts_exl3.py`** (MIT)
     - shard を 1 本ずつ流して routed experts だけを EXL3 にする。校正はしない。
     - RAM は数十 GB で足り、Engram は hardlink で済む。
     - 出力は diffbot / nktlabs の vLLM+vllm-exl3 構成でそのまま読める。PixelML が 1017 で実際に使っている。
   - Q2: **sudoingX のパイプライン** (Apache-2.0)
     - DeepSeek の参照 `inference/model.py` を層単位で流して校正する。expert ごとに K∈{2,3,4,5} を割り当てる。
     - 公開物の中で最も品質が良い。2.70bpw で KLD 0.0243、Mia 2.9bpw より良い。
   - Q3: **exllamav3 の V4.1 アーキ** (本家 PR #449 = anoane ブランチ。LibertAIDAI も同等品を自作した)
     - `convert.py` と `-rcp` によるテンソル別配分の経路。ただし本家にはまだマージされていない。
3. **128GB RAM でも 510GB 全体を流せる。** 3 系統とも shard 単位か層単位で読む。常駐するのは 1 層 (約 7.4GB) と校正用の活性化だけで、Engram は disk から gather する。
4. **2.2bpw 前後の品質は、expert 別の混合配分と校正をするかどうかで大きく変わる。**
   - 一律 2.0bpw (校正なし) は KL 0.36 (diffbot) と悪い。
   - 混合配分した例は次のとおり (V4.1-Flash)。
     - SAGE 1.59bpw (混合 K1〜K6): KL 0.19
     - Krishando 2.41bpw (expert 別): KLD 0.198 / 0.129
   - 1017 の 2bit 一律 (Lucebox) は KL 0.265 だった。
   - 使用頻度の偏りが大きいので、混合配分が効く (§4)。上位 25% の experts に routing の約 56% が集まり、下位 30% は約 7% しかない。
5. **刈り込み (REAP) は 1017 版がまだ公開されていない。**
   - V4.1 の公開値は、text PPL が 272E で +4.1%、256E で +5.5%、192E で +14.7% (Libertai)。MLX の 192E は ×1.168 だった。
   - Libertai の REAP ツールは 1017 にもそのまま使える【推測。config が同一のため】。
   - 256E + 3bpw なら experts は 136GB で収まる【計算】。比較候補の一つになる。
6. **GGUF (IQ2/IQ3) の経路は、このマシンでは不適と判断する。** 理由は 3 つ。
   - llama.cpp 本家に V4.1 の runtime が無い。PR #28696 は open のまま。
   - ik_llama.cpp は SM120 で GPU に experts を載せると NaN になる (#2519 open)。並列処理も弱い。
   - 1017 から変換するには約 500GB の中間 GGUF が要り、ディスクが足りない。

---

## 1. EXL3 (exllamav3 と各 fork)

### 1.1 所在と V4.1 対応の実態【事実】

| 実装 | 所在 / rev | V4.1 の扱い | 根拠 |
|---|---|---|---|
| 本家 exllamav3 | `turboderp-org/exllamav3` dev `448f5ea7` (v1.6.3, 2026-10-17) | **`DeepseekV41ForCausalLM` が無い**。`architecture/` にあるのは `deepseek_v4*.py` だけで、`grep -ril engram` も 0 件。V4 (`DeepseekV4ForCausalLM`) は変換に対応しており、`calibration_all_experts = True` | `ext/exllamav3/exllamav3/architecture/deepseek_v4.py:23,268`。`git log` |
| 本家 PR #449 (anoane) | `anoane/exllamav3-anemone` ブランチ `dsv41f` `c341a8f` (open, 2026-10-05〜15) | `deepseek_v41.py` / `_mtp.py` / `_vision.py` を追加。FP8/MXFP4 (E8M0) の source を読む `Linear` を持つ。推論は vLLM と logprob が一致したと報告されている (1〜3 枚)。**変換 (convert.py) が通るかは PR に書かれていない** | https://github.com/turboderp-org/exllamav3/pull/449 、`ext/exllamav3_anoane_dsv41f/exllamav3/architecture/deepseek_v41.py:21` |
| LibertAIDAI | 公開コードは無い (GitHub の Libertai org に量子化ツールは無い) | 「exllamav3 dev `ba2feb2` (1.6.2) の `convert.py` に、**このリリース用に自作した DeepseekV41 定義**」で 1017 の 2.9 / 3.5bpw を作った。Upstream exllamav3 has no DeepseekV41 model class yet | `research/tmp/r2q/scan0352/cards/LibertAIDAI_DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw.md` (How it was made / Serving) |
| vcruz305/exllamav3 (fork) | `ext/vcruz305_exllamav3` `475461e` | V4.1 のモデルクラスは無い (grep で 0 件)。sm_120 を含む fat wheel を配布している。`exl3_moe_mixedk` (mixed-K MoE kernel) を提供 | `ext/vcruz305_exllamav3/README.md` (wheel の節)、`ext/vcruz305_vllm-exl3/README.md:41` |
| **vcruz305/vllm-exl3** (vLLM プラグイン、AGPL) | `ext/vcruz305_vllm-exl3` `2a740a3` (0.5.0 dev) | **serving 専用で、量子化はしない**。「a V4.1-capable vLLM runtime owns DeepseekV41ForCausalLM … this plugin does not implement a standalone V4.1 model class」。`layer_bits` (層別 K) と **expert 別の mixed-K** (#10、`p2b_fused_moe_mk` #31。既定では off) に対応 | `ext/vcruz305_vllm-exl3/README.md:7,55-60,168`、`CHANGELOG.md:57,82` |
| diffbot | HF `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000` の `recipe/` | vllm-exl3 `d3cfd394` への AGPL パッチ (host experts、側 stream、`EXL3_EXPERT_STATS`) と、**sm_120 向けに書き直した MoE kernel `xmoe`** (K=2/3、MUL1/MCG) | `research/tmp/r2q/diffbot/README.md` (The MoE kernel 節) |
| nktlabs | HF `nktlabs/dsv41-flash-engram-disk-prestage` | 量子化はしない。diffbot 2.0bpw 用に **Engram を NVMe から prestage するパッチ** (4 ファイル) | `research/tmp/r2q/cards/nktlabs_dsv41-flash-engram-disk-prestage.md` |
| **sfxnz** | `ext/sfxnz_recipe/tools/quantize_experts_exl3.py` (MIT) | **V4.1 の routed experts を量子化するツール**。PixelML の 1017 3bpw はこれで作られた。Krishando もこれを出発点にしている | 同ファイルの冒頭 docstring。PixelML / Krishando のカード |
| **sudoingX** | `ext/sudoingx_dsv41/scripts/quantize/` (Apache-2.0) | **校正つき・expert 別 K の完全なパイプライン**と serving 用 plugin (`patches/exl3_moe` → `vllm_exl3_moe`) | `ext/sudoingx_dsv41/RECIPE.md:89-104` |

結論として、この課題で使える「量子化器」は sfxnz (Q1)、sudoingX (Q2)、exllamav3 の V4.1 アーキ (Q3) の 3 つである。課題文の候補 (diffbot / LibertAIDAI / vcruz305 / nktlabs) は、どれも量子化器としては公開されていない。各候補が実際に担っているものは次のとおり。

- diffbot: MTP の 3bit 量子化スクリプト `quantize_mtp_exl3.py` と assemble だけ。
- LibertAIDAI: 量子化器は非公開。
- vcruz305: serving 側。
- nktlabs: Engram の扱い。

### 1.2 Q1: sfxnz `quantize_experts_exl3.py` (校正なし、shard ストリーム)

**中身【事実】**

- shard を 1 本ずつ開く。routed experts の `w1/w2/w3` だけを MXFP4 → fp32 に CPU で展開し (`_dequant_t`、`:148`)、GPU で EXL3 trellis 符号化する。
- 他のテンソルはそのままコピーする。experts を含まない shard (Engram の 47/48 を含む) は **hardlink** する (`:6`、`link_or_copy`)。
- Hessian は単位行列で、校正データは使わない。符号化方式は 3 つから選べる。
  - 既定: tail-biting Viterbi の `quantize_tiles`
  - `--greedy --beam 16`
  - `--ldlq`
- `--allow-partial` で未ダウンロードの shard を飛ばせる (`:602`)。shard ごとに atomic rename するので、途中から再開できる。`--only-files` で 2 GPU に shard を分担させられる (`:607`)。
- **既定では `--bits 2` 以外を拒否する** (`:617-620`)。PixelML はこの 1 行を書き換えて K=3 にした (カードに明記)。
- config に `layer_bits` を書けば、層別の K を宣言できる (`pack_meta.py:73,98-99`)。
- DSpark の drafter experts は `"mtp_experts": "source"`、つまり MXFP4 のまま残す (`pack_meta.py:82`)。

**1017 での手順 (案。GPU を使う段階で実行する)**

1. exllamav3 を入れる。vcruz305 の sm_120 fat wheel か、本家を sm_120 で build する。diffbot と PixelML は本家の `5be88657` (v1.4.9) を pin している。
2. `quantize_experts_exl3.py` の bits ガード (`:617`) を外す。`write_pack_config` に `layer_bits` を渡す改修を入れる。
3. 層を K2 群と K3 群に分け、それぞれ `--only-files` で GPU0 / GPU1 に流す。例: K3 を 6 層、K2 を 34 層にすると平均 2.15bpw で 146.8GB になる【計算】。
   ```
   python3 -I tools/quantize_experts_exl3.py --src $MODELS/DeepSeek-V4.1-Flash-1017 \
     --dst $MODELS/dsv41-1017-exl3-2.15 --bits 2 --codebook mul1 --greedy --beam 16 \
     --batch 8 --device cuda:0 --only-files <K2 層の shard 群>
   # K3 層の shard 群は --bits 3 で cuda:1 に流す
   ```
   - shard と層の対応は確認済み【実測】。1017 の `model.safetensors.index.json` を Python で集計すると、expert を含む shard は 43 本 (層 0〜39 と mtp 0〜2) だった。2 層にまたがる shard は 0 本で、例えば `model-00003` は層 0、`model-00004` は層 1 である。したがって **`--only-files` で shard を選べば、そのまま層別の K になる**。
4. 必要なら drafter の experts を diffbot の `quantize_mtp_exl3.py` で 3bit にする。7.2→5.1GB になり、所要は 1 GPU で約 9 分【事実: diffbot quantize_README】。
5. `assemble` (sfxnz の `tools/assemble_pack.sh`、または diffbot の `assemble_pack.py`) で index を作り直す。

**資源の見積もり**

- ホスト RAM【計算】: 一度に持つのは、1 shard 分の非 expert テンソル (≤ 約 11GB) と、fp32 に展開した 8 experts 分 (8×35.4M×4B ≒ 1.1GB) と、出力中の shard 分だけ。**30GB 以下**。128GB で問題ない。
- GPU メモリ【計算】: 1 batch で数 GB。1 枚 96GB で余裕がある。
- 所要時間【計算】
  - 根拠 1: diffbot の drafter experts (1,152 テンソル、K3、greedy beam16) が 1 GPU で約 9 分 = 0.47s/テンソル。
  - 根拠 2: PixelML の本体 46,080 テンソル (K3) が H100 で 6.6 GPU 時間。
  - ここから、K3 は約 6 GPU 時間。K2 は候補数が半分なので約 3 GPU 時間。平均 2.15bpw なら **2 GPU で約 1.5〜2.5 時間**。
  - ただし MXFP4 の展開を CPU で逐次に行う部分 (`_dequant_t`) が律速になる可能性がある。その場合は GPU で展開するよう改修する (UMA でない機械なので安全)。
- ディスク【計算】
  - 出力は experts 約 147GB と、experts を含む shard に同居する非 expert テンソルのコピー (数 GB〜十数 GB)。
  - Engram (203GB) は hardlink なので増えない。
  - 現在の空きは 528GB (`df -h`)。

**品質の目安**: §5 を参照。校正なしの一律 2bit は KL 0.36 級で、案 A の品質リスクそのものである。Viterbi と scale refit を併用すると、NLL が 0.243→0.139 nats/token に下がる (sfxnz の `requant_full.py`、HF `sfxnz/DeepSeek-V4.1-Flash-EXL3` のカード)。

### 1.3 Q2: sudoingX のパイプライン (校正つき、expert 別 K)【事実】

`ext/sudoingx_dsv41/RECIPE.md:89-104` と `scripts/quantize/*.py` の内容。

| 段 | スクリプト | 中身 | 2x GB10 での所要 (実測) |
|---|---|---|---|
| 参照の確認 | `ref_truncated_logits.py` | DeepSeek の `inference/model.py` を層単位で流し、vLLM と一致することを確かめる | 数分 |
| pass 0 | `p4_pass0.py` | expert ごとの routed 数、routing 重み、出力エネルギー、per-token share | 約 35 分 (1 台) |
| 感度 | `p4_sensitivity.py` | 層ごとに、ノイズ単位あたりの最終 KL を測る | 約 27 分 (1 台) |
| 配分 | `p4_allocate.py` | expert ごとの K。K∈{1.5,2,2.5,3,3.5,4,5,6} (`:29`)。既定は整数 {2,3,4,5}。メモリ予算 `--bpw` と decode 読み出しの上限 `--decode-bpw`。super expert の保護 (`--protect-share`)。DSpark が参照する層 37〜39 の下限 (`--floor-layers`) | 数秒 |
| pass 1 | `p4_pass1.py` | 層を順に、量子化済みの前段を通した stream で校正する。routing 重みつき Hessian に全トークンの事前分布を足す。mul1 で符号化。`--part i/n` で 2 台に experts を分担 | 約 20 分/層、**40 層で約 13 時間** |
| dense / head / drafter | `p4_dense.py` / `p4_drafter*.py` | 任意。dense を K4、head を K4、drafter を dense K5 / experts K3 に | TBD |
| 組み立て | `p4_assemble.py` / `p4_link_variant.py` | 公式の非 expert shard と EXL3 を組み合わせる。Engram の shard は hardlink | — |
| 忠実度 | `p4_kld.py` | 参照実装との KLD と top-1 (他 pack も `--exl3-ckpt` で採点できる) | — |

- Streamer は 1 層 (約 7GiB) だけを常駐させ、全シーケンスの residual stream `h [L, hc=4, 5120] bf16` を device に置く (`dsv41q/stream.py:1-3,33-56`)。
  - 853K トークンで stream だけで約 35GB【計算】。pass1 は「1 層あたり約 49GiB (853K tokens)」(`p4_pass1.py:16`)。
  - **96GB の GPU 1 枚に収まる。** 校正トークンを 300〜500K に減らせば、さらに余裕が出る。
  - GB10 (UMA) 向けの `set_per_process_memory_fraction(0.70)` (`stream.py:39`) は、この機械では外すか値を上げる。
- Engram は disk から gather するので RAM に載せない。ただし **reader は vLLM 側の拡張 `vllm.models.deepseek_v41.common.engram._load_engram_disk_ext` に依存する** (`dsv41q/refmodel.py:198`)。
  - これは彼らのパッチ `patches/0002-engram-disk-offload.patch` を当てた vLLM image が前提になる。
  - 代わりに、Libertai の `scripts/stream_model.py` (Engram の disk gather と行キャッシュ) に差し替えることもできる。
- このマシン (2x RTX PRO 6000 Max-Q) での所要【推測】
  - GB10 1 台に対して計算性能は約 3 倍、帯域は約 6 倍。
  - **pass1 は 2 GPU で約 4〜6 時間。pass0 と感度測定で約 0.5 時間。合わせて GLM の停止時間は 5〜7 時間。**
- 校正データ
  - 公開分: `data/caldata/conversations.jsonl` と `instruct_v3.jsonl` (V4.1-Flash の出力、853K トークン)。
  - 自前: GLM の要求ログ (日本語とエージェント型) を V4.1 の encoder で再エンコードして足すべきである。sudoingX は「EXL3 は Hessian の入力統計に合わせて符号化する。校正で薄い領域ほど KLD の差が広がる」と記録している (`RECIPE.md` 校正 v3 の節)。
- 出力形式
  - `quant_method: deepseek_v41_exl3` と、expert ごとの K を持つ `exl3_moe_plan.json`。
  - 読めるのは彼らの `vllm_exl3_moe` plugin だけ。vLLM main `6e517b15` に 20 本のパッチ (0001-0008、0010-0021) を当てたもの。
  - vcruz305 / diffbot 形式に変換するには、tensor 名の対応付けと mixed-K の kernel が要る (§1.6)。

### 1.4 Q3: exllamav3 `convert.py` + V4.1 アーキ (LibertAIDAI 方式)

- 方法【事実: LibertAIDAI 1017 のカード】
  - exllamav3 の本来の変換経路 (モジュールごとの Hessian 取得、LDLQ、trellis 符号化) を使う。
  - 校正は既定の mix で 250 行 × 2048 トークン。全 expert に全校正トークンを通す (`calibration_all_experts`)。
  - MTP は校正なし。
- K の決め方: 1017 の 2 版は「テンソル別のレシピで、整数 K のみ、routed experts は 1 層に 1 つの K」とした。理由は「the kit's MoE loader needs that」。
  - 2.9bpw: experts K3、層 18〜22 だけ K2。
  - 3.5bpw: 層 0〜9 と 30〜39 が K4、層 10〜29 が K3。
- 本家 exllamav3 の **`sc_measure.py` → `sc_optimize.py` → `convert.py -rcp`** (`doc/optimize.md`) は、テンソル単位で最適に配分する仕組みである。
  - 各 expert の w1/w2/w3 はそれぞれ別テンソルなので、原理的には expert 別の K にできる。
  - ただし「**It is currently untested on sparse models**」(`doc/optimize.md:16`) とある。
- 使える状態か
  - V4.1 のアーキは本家に未マージ (PR #449)。PR #449 で変換まで通るかは未検証【推測】。
  - 出力を読むのは Mia kit (vLLM 0909 image の上に EXL3 overlay。GB10 / arm64 向け) か、anoane の exllamav3 単体 (TabbyAPI)。後者は並列処理に向かず、3 枚で decode 46 tok/s 程度 (PR #449 本文)。
  - **この課題の本命の経路にはしない。** 比較用の 2.9bpw 品質の基準として残す。

### 1.5 層別 / expert 別の混合 bpw (公開事例)【事実】

| pack | 対象 | expert の bpw | 配分方法 | 結果 |
|---|---|---|---|---|
| LibertAIDAI 1017-2.9 | 1017 | 2.875 | 層別 (K3、層 18〜22 は K2) | PPL code +1.8%、prose +0.4% |
| LibertAIDAI 1017-3.5 | 1017 | 3.53 | 層別 (端の 20 層 K4、中央 20 層 K3) | PPL +0.8% / +0.1% |
| Krishando 2.86 | V4.1 | 2.86 | expert 別 K2〜5 (2bit 7,575、3bit 15,697、4bit 6,877、5bit 571 テンソル)。量子化済み自身の活性で Hessian を取り直す | KLD agent 0.168 / 一般 0.088、MMLU 88.0% |
| Krishando の途中版 | V4.1 | 2.63 → 2.41 → 2.66 | 層別 2.63: 0.209/0.131 → **expert 別 2.41: 0.198/0.129** → 2.66: 0.176/0.104 | **expert 別は、ビットが少なくても層別より良い** |
| sudoingX 2.70 | V4.1 | 2.70 (decode 読み出しは 2.90) | importance = 層の感度 × per-token share。super expert は K5 | KLD 0.0243 (Mia 2.9 は 0.0295) |
| jayleaton 2.8 | V4.1 | 2.78 | expert 別 K2〜5。decode 読み出しバイト数の上限つき | KL 0.0469 (Mia 0.0594) |
| vcruz305 SAGE | V4.1 | 1.59 (K1〜K6) / 3.30 (K2〜K8) | 非公開の手法 | KL 0.1885 / 0.0910 |

- 出典: `research/tmp/r2q/scan0352/cards/` の各カード。
- KL の測り方は pack ごとに違う (top-k か全語彙か、served か teacher-forced か)。比べてよいのは同じ表の中の数値だけである。
- V4.1 固有の注意 (sudoingX `FINDINGS.md:50-53`): **各層に「super expert」がいる**。
  - 少数のトークンで巨大な出力を書く expert で、層 1 ではエネルギーの 92.3% を持つが、per-token share は 1.4% しかない。
  - エネルギーで配分すると他の experts が飢えるので、per-token share で配分し、super expert は K5 で保護する。

### 1.6 出力を読むエンジン (SM120 TP2 で、DSpark、Engram disk、OpenAI 互換を満たすもの)

| エンジン | 読める形式 | mixed-K | 2x RTX PRO 6000 での実績 | 注意 |
|---|---|---|---|---|
| **vLLM nightly + diffbot の patched vllm-exl3 + xmoe** | sfxnz / PixelML / diffbot 形式 (`quant_method: exl3`、`layer_bits`) | 層別は可。expert 別は未確認 (xmoe には K2/K3 の instance がある) | **あり**。3bpw + host experts で agentic 1 本 159 tok/s。2.0bpw を全 VRAM に置いて 46K decode 116.6 tok/s | AGPL のパッチ。Engram は pinned int4 前提なので、nktlabs の prestage を `nightly-5555a2d8` 系に移植する必要がある |
| vLLM + vcruz305 vllm-exl3 0.5.x | 同上 | **expert 別 mixed-K に対応** (`p2b_fused_moe_mk`、既定 off、GB10 で検証中) | 直接は無い (inferencekey が 2x96GB で host offload を研究) | 主な検証対象は GB10 TP1 GLM (`README.md:49`) |
| vLLM main `6e517b15` + sudoingX の 20 パッチ + `vllm_exl3_moe` | sudoingX 形式 (`exl3_moe_plan.json`) | **expert 別に対応** (`exl3_smoe`。全 K を 1 launch set で処理) | 無い (GB10 2 台。code 103 tok/s) | kernel は sm_121 向けの JIT。sm_120 で build し直し、RDMA ring を無効にして PCIe TP2 で動かす必要がある |
| TensorFold 系 (bertholomus / Kristianaaron / jayleaton) | 独自。ragged per-expert | 対応 | GB10 だけ | SM120 での V4.1 の実績は見つからない |
| exllamav3 単体 (PR #449) + TabbyAPI | LibertAIDAI / Mia 形式 | 層別 | 1〜3 枚で 35〜46 tok/s | 並列処理に向かない |

---

## 2. GGUF (IQ2/IQ3、llama.cpp / ik_llama.cpp など)

### 2.1 状況【事実】

| 項目 | 状態 | 根拠 |
|---|---|---|
| llama.cpp 本家 | **PR #28696 は open** (最終更新 2026-10-10、コメント 26)。apetersson は「converter-only and V4.1 ships no `output_hc_*` tensors」と書いている。mrfakename は vcruz305 fork の `2076412` で 1017 の Q2_K〜Q8_0 を作った | GitHub API `pulls/28696`、`scan0352/cards/apetersson_*.md`、`research/tmp/r2q/cards/mrfakename_*.md` |
| ik_llama.cpp | `deepseek41` は #2455 で 2026-09-21 にマージ済み。converter (#2506)、vision (#2545)、DSpark (#2522/#2546) もある。**trellis 系の IQ2_KT / IQ1_KT を CUDA で encode する機能が #2682 (10-16) に入った**。IQ3_KS などは #2671 | GitHub API `search/issues` (`gh/ik_search.json`) |
| ik_llama.cpp の SM120 不具合 | **#2519 open**: SM120 で experts を GPU に置くと NaN か illegal memory access。CPU に置いても長い入力で出力が崩れる。10-15 に、KV / 圧縮キャッシュの書き込みと読み出しの間に依存の辺が欠けている (CUDA 固有の read-before-write) という原因分析と修正案が投稿された。#2683 (200K 超で文脈を失う) も open | `gh/ik2519.json`、`ik2519c.json` |
| DwarfStar (antirez/ds4) | 独自の GGUF。Q2 = routed gate/up IQ2_XXS、down Q2_K で、main weights は 151.77GiB (V4.1-Flash 版。1017 版は未公開)。**CUDA は V4.1 では text のみ**。`--cuda-tensor-parallel` で 2 枚を組にできる。ただし session batching 中は DSpark が無効で、Q2 の grouped shape は遅い fallback になる | `scan0352/cards/kyuz0_*.md`、`gh/ds4_main.md:10`、`gh/ds4_CUDA_MULTI_GPU.md`、`gh/ds4_SERVER.md:88-91` |
| 1017 の GGUF | mrfakename / mondk: Q2_K 246.3 / Q3_K_M 323.4 / Q4_K_M 414.2 / Q8_0 473.1 GiB (imatrix なし)。Lucebox: ROCMFP (AMD 専用)。AMAImedia FP8 660GiB。ほかに改変版の Q6_K | `scan0352/info/*.json` の集計 |

### 2.2 1017 を IQ2/IQ3 にする手順と、このマシンでの可否

- 手順 (ik_llama.cpp)
  - `convert_hf_to_gguf.py` で 1017 を GGUF (Q8_0 か BF16、experts は MXFP4 のまま) に変換する。
  - imatrix を取る。
  - `llama-quantize --custom-q` で experts を IQ2_KT / IQ2_XXS / IQ3_KS に、Engram を Q8_0 か FP8 相当にする。
- **ブロッカー 1: ディスク。** 中間の GGUF は vcruz305 Q8_0 の内訳 (routed MXFP4 288.8GB + engram Q8_0 209.2GB + 他 9.9GB) で約 508GB になる (`gh/ik2519.json` 本文)。1017 本体の 476GB と並べると、空き 528GB を超える。
- **ブロッカー 2: 2bit 級でも 150GB に収まらない、または収まっても余裕が無い。**
  - Q2_K 一律は experts だけで 2.625bpw×543.6B ≒ 178GB【計算】。
  - IQ2_XXS/Q2_K の混合 (ds4 Q2 と同じ) は約 2.25bpw で 153GB。
  - さらに IQ2_KT (2.125bpw) などに寄せる必要がある。
- **ブロッカー 3: 速く読めるエンジンが無い。**
  - ik_llama.cpp は SM120 の GPU experts が #2519 で壊れている。
  - 2 枚は層分割で、vLLM のような連続バッチや DSpark の並列は弱い。c16 で 384 tok/s を出す見込みは無い【推測】。
  - ds4 は CUDA の V4.1 が text のみで、batching 時に DSpark が無い。L40S×8 でも V4 Flash の合計は約 126 tok/s。
- **評価: GGUF の経路は、ユーザーの目的 (GLM 並みの並列、画像、ツール、長文脈) には合わない。** 参照品質の比較や CPU 側の検証にとどめる。

---

## 3. 1017 の既存量子化物の再スキャン (2026-10-17 18:52Z 時点)

コマンド: `curl "https://huggingface.co/api/models?search=<q>&sort=lastModified&direction=-1&limit=100&full=true"`。q は V4.1-Flash-1017 / DeepSeek-V4.1-Flash / 1017-EXL3 / V4.1-Flash-EXL3 / V4.1-Flash-REAP / V4.1-Flash-GGUF / dsv41 / V4.1-Flash-1017-REAP / DSV4.1。結果は `research/tmp/r2q/scan0352/s_*.json`。サイズは `?blobs=true` の siblings の合計。

### 3.1 1017 ベース (第 1 ラウンド 01:00 以降の新着に ★)

| repo | 形式 | 全体 | experts 推定 | 2x96GB に収まるか | エンジン | 品質 |
|---|---|---|---|---|---|---|
| LibertAIDAI/…-1017-EXL3-2.9bpw | EXL3 mul1 2.875 (層別) | 210.7GB | 約 196GB | ×、約 46GB 超過 | Mia kit (vLLM+EXL3)。diffbot 系でも layout はほぼ同じ【推測】 | PPL +1.8% / +0.4% |
| ★ LibertAIDAI/…-1017-EXL3-3.5bpw | EXL3 3.53 | 257.6GB | 約 240GB | × | 同上 | PPL +0.8% / +0.1%。Engram off で 4.15 / 8.97 |
| ★ PixelML/…-1017-EXL3-3bpw | EXL3 K3 (校正なし、greedy beam16) | 223.0GB | 約 204GB | × | vLLM + vllm-exl3 (diffbot / sfxnz 形式) | 未測定 |
| LakyFx/…-1017-EXL3-3.5bpw-TensorFold | EXL3 3.51 | 145.3GB (途中) | — | × | TensorFold (GB10×3) | 未完成 |
| Baekpica/…-1017-VQ-GGUF | 8 次元 VQ、約 1.6bpw | 本体 113.6GB と Engram | 約 100GB | 容量は ○ | **ds4-dfm-rs (GB10 専用)** | V4.1 で top-1 69〜79%、KL 0.42〜0.69 |
| Lucebox/…-1017-ROCMFP-GGUF | fp2s / fp2s+fp3 | 383〜405GB | 158〜179GiB | fp2s だけ容量は ○ | **AMD 専用 (luce_server)** | **1017 で KL 0.265 (2bit) / 0.244 (MIX10) / 0.195 (fp2/3)** |
| mrfakename / mondk 1017-GGUF | Q2_K〜Q8_0 | 246〜473GiB | Q2_K で約 178GB | × | llama.cpp の vcruz305 branch | 未測定 |
| LibertAIDAI/…-1017-NVFP4 | NVFP4 | 399.5GiB | 約 285GiB | × | vLLM (#61530) | 未測定 |
| ★ pipenetwork/…-1017-MLX-mixed-4_8bit、mlx-community oQ*、gnahzhang LSQ3、inferencerlabs Q9 | MLX | — | — | 対象外 | Apple | pipenetwork: wikitext PPL 2.9238 (旧版比 ×1.0095) |
| ★ 重みの改変版 (範囲外)、AMAImedia FP8-GGUF、Tylogi (空) ほか | 改変・その他 | — | — | 対象外 | — | — |

**結論: experts が 150GB 以下で、かつ SM120 の vLLM 系で読める 1017 の量子化物は存在しない。** V4.1-Flash 版には近いものがある (diffbot の旧 2.0bpw、satgeze 2.0bpw、vcruz305 SAGE 1.59bpw = experts 108.9GB)。しかし 1017 は重みも Engram もすべて変わったので、流用はできない (`web_engines.md` §1.1)。

### 3.2 参考: V4.1-Flash (0909) で experts ≤150GB のもの

| repo | experts | KL | 備考 |
|---|---|---|---|
| vcruz305/DSV4.1-Flash-SAGE-EXL3-1.59bpw | 108.9GB (K1〜K6 混合) | 0.1885 (全語彙、vs FP4)、top-1 91.2% | 総 330GB。読むには「DeepSeek V4.1 / CED 対応の exllamav3 と K1 kernel」が要る。手法は非公開 |
| diffbot 旧 2.0bpw (rev `28b7ab71`、現在は 3bpw repo に統合) / sfxnz 2.0bpw MCG | 約 136GB | 0.361 (AtomicChat neutral) | nktlabs: c1 80.2、c8 449.7 tok/s |
| satgeze 2.0bpw-Ablit | 同上 | stock 0.4389 (on-policy) | Engram は MXINT-4 |

---

## 4. 専門家の刈り込み (REAP など) と routing の偏り

### 4.1 公開済みの刈り込み版【事実】

| repo / 実験 | ベース | 残す数 | 品質 | 備考 |
|---|---|---|---|---|
| LibertAIDAI/…-REAP-272E | V4.1 | 272 | text PPL +4.13%、caption +1.9% | **vLLM の expert 数テーブル {…,192,256,320,384,…} に無いのでロードできない** |
| LibertAIDAI/…-REAP-256E | V4.1 | 256 | mean ranking: text +5.49%、caption +27.4%。balanced (公開版): text +14.1%、caption +8.9% | MXFP4 のまま 385.5GiB |
| Libertai の測定 | V4.1 | 240 / 192 | +7.12% / **+14.66%** | 同じ README |
| pipenetwork REAP50 (MLX 4bit) | V4.1 | 192 | wikitext PPL ×1.168 (140 窓中 138 で悪化) | |
| rapid-mlx REAP-2bit | V4.1 | 336 | routing の質量を 99.995% 保持 | Apple |
| buddalaw K128 / K160 (ds4 Q2) | V4.1 | 平均 128 / 160 (router でマスク) | **領域内** (イタリアの法律) で KL 0.119 / 0.098。領域外は大きく悪化 | 領域特化型。ファイル自体は削らない |
| **1017 の REAP 版** | — | — | **無い** (search=REAP、V4.1-Flash-1017-REAP の結果は 0 件) | |

**道具**: `ext/libertai_reap` (`Libertai/deepseek-v41-flash-reap`)

- 120GB UMA の機械 1 台で動く。meta 上でモデルを組み、層ごとに experts をストリームし、Engram は disk から gather する。
- `reap_calibrate.py` (text と image の saliency)、`prune_experts.py` / `prune_convert_mp.py` (router の `gate.weight/bias/bias_vl` の行も削る)。
- 1017 は config が同一なので、そのまま使える【推測】。
- 注意点: **画像も入れて校正し、text と image の saliency を正規化してから平均する**こと (text だけで選ぶと、vision はランダムに刈るより悪くなる)。日本語も校正に入れること【推測】。

### 4.2 routing の偏り (公開データから集計)【事実 + 計算】

- データ: sudoingX `data/p0_stats_v3/merged_bal/layer_XX.json`。V4.1-Flash に 853,106 トークン (コード、エージェント、長文、会話、数学、多言語など) を流し、expert ごとに routed slot 数 `count` と per-token share `tshare` を記録したもの。
- 集計: `python3 -I tools/r2_routing_skew.py <dir> --key count`。出力は `research/tmp/r2q/skew_count.txt`。

| 指標 (40 層の代表値) | count (routing 回数) | tshare (出力への寄与) |
|---|---|---|
| 上位 10% の experts が持つ routing の割合 | 0.25〜0.37 | — |
| 上位 25% | 0.49〜0.60 (多くの層で約 0.56) | — |
| 上位 50% | 0.76〜0.84 | — |
| 下位 30% (層内) | 0.057〜0.101、平均 **0.072** | 平均 **0.040** |
| 全 15,360 experts のうち下位 30% (層を同じ重みで正規化) | **0.072** | 0.038 |
| 全体の下位 50% | 0.189 | 0.126 |
| routing の 90% を覆うのに要る experts 数 | 234〜270 (平均 247) | 平均 211 |
| Gini | 0.37〜0.50 (平均 0.46) | 0.55 |
| 最大 / 平均 | 4.0〜21.3 | — |

- diffbot の実トラフィックでの値【事実: `research/tmp/r2q/diffbot/plans_README.md`】: 全体の下位 30% (4,608 experts) に当たる decode の routing の割合は、長いエージェント型コードセッションで **2.4%**、合成プロンプトでは 11〜24%。
  - **用途を絞るほど偏りは大きくなり、偏り方は用途によって変わる。**
- この機械への意味【推測】
  - 混合配分 (案 A) は効く。expert の 25% を K1.5 にしても、routing の約 5〜7% にしか触れない。
  - 192E への刈り込みは、少なくとも routing の約 19% (出力寄与で約 13%) を捨てる。PPL +15% と整合する。
  - 校正は自分のトラフィック (GLM の要求ログ) でやるべきである。

---

## 5. 品質の目安: bpw 別の KL / PPL / ベンチ

KL の測り方が出典ごとに違うので、**同じ行の中か、同じ出典の中だけで比べること。**

| 出典 (測り方) | 構成 | expert の bpw | KL など | ベンチ |
|---|---|---|---|---|
| diffbot (AtomicChat: top-512、neutral / code / agentic) | 原モデルの 2 回目の実行 (ノイズの床) | — | 0.0158 / 0.0100 / 0.0055 | |
| 同上 | NVIDIA NVFP4 | 4.5 | 0.0352 / 0.0197 / 0.0089 | |
| 同上 | coolbho3k 3.0 (校正あり) | 3.0 | 0.0545 / 0.0321 / 0.0101、top-1 92.9 / 96.0 / 98.2 | GSM8K-200 98.0%、HumanEval 95.7% |
| 同上 | 旧 2.0 (一律) | 2.0 | **0.3613 / 0.2147 / 0.0766**、top-1 80.4 / 88.5 / 94.7、PPL +13.4% / +11.2% / +2.7% | GSM8K 97.5%、**HumanEval 91.5%** |
| sfxnz (40 文の NLL) | 2.0 MCG → Viterbi + refit | 2.0 | 0.243 → **0.139** nats/token | GSM8K-100 97、MMLU 4×57 206/204 |
| vcruz305 SAGE (全語彙、32×4096、vs FP4) | 混合 K1〜K6 / K2〜K8 | 1.59 / 3.30 | **0.1885** / 0.0910 (top-1 91.2 / 94.0) | |
| Krishando (served、top-20、agent / 一般) | expert 別 2.41 / 2.66 / 2.86 | 2.41〜2.86 | 0.198/0.129 → 0.176/0.104 → 0.168/0.088 (Mia 2.9: 0.248 agent) | 2.86 で MMLU 88.0%、GSM8K 97% |
| sudoingX (自前パネル 254K 位置) | expert 別 2.70 / 0xSero 2.43 / Mia 2.9 | 2.43〜2.9 | 0.0243 / 0.0327 / 0.0295 | HumanEval+ 97.0/90.9 (native 97.6/92.1)、IFEval で有意差なし |
| jayleaton (served) | expert 別 2.78 / Mia 2.9 | 2.8 / 2.9 | 0.0469 / 0.0594 | MMLU-200 88.5 / 87.5 |
| coolbho3k (teacher-forced 81,880) | 3.0 校正 / refit | 3.0 | 0.0414 (top-1 93.6%) | ARC-C 123/128、GSM8K 63/64 |
| satgeze (on-policy、公式 API の軌跡) | 2.0 stock / Mia 2.9 | 2.0 / 2.9 | 0.4389 / 0.1238 | GSM8K-50 92 / 96 |
| **LibertAIDAI (1017、PPL 65K トークン)** | 2.9 / 3.5 | 2.875 / 3.53 | code 1.8749→1.9094 (+1.8%) / 1.8895 (+0.8%)、prose +0.4% / +0.1% | |
| **Lucebox (1017、top-256、vs fp4)** | fp2s / MIX10 / fp2/3 | 約 2.2 / 2.3 / 2.5 | **0.265 / 0.244 / 0.195**、top-1 78.6〜81.5%。September 版は 0.266 / 0.246 / 0.204 | |
| Baekpica / YoungAi (V4.1) | VQ | 約 1.6 | 0.42〜0.69、top-1 69〜79% | |
| REAP (Libertai、PPL) | 272 / 256 / 240 / 192E | MXFP4 | +4.1% / +5.5% / +7.1% / +14.7% | |

- 出典はすべて `research/tmp/r2q/{cards,diffbot}/`、`research/tmp/r2q/scan0352/cards/` の該当カード。
- Lucebox の比較から、**1017 と 0909 の量子化耐性はほぼ同じ**と見てよい【推測】。

**読み取り【推測】**

- 2.0〜2.2bpw では、**一律・校正なしと、expert 別・校正ありとで KL が 2〜3 倍違う。**
  - 一律・校正なし: 0.25〜0.36 (diffbot、Lucebox)。
  - expert 別・校正あり: SAGE 1.59 で 0.19、Krishando 2.41 で 0.13〜0.20。
- 案 A の現実的な見込み (expert 別、自分のトラフィックで校正、平均 2.15〜2.2bpw)
  - KL は 3bpw の約 2〜4 倍 (AtomicChat 換算で 0.10〜0.20)。
  - PPL は +3〜6%。
  - GSM8K はほぼ同等。HumanEval は −1〜4 点。
  - PLAN X7 の合格線 (KL ≤ 0.10) は、**同じ尺度 (AtomicChat 系) で測ると届かない可能性が高い**。合格線を「top-1 一致と実タスク」に置き直すか、§6 の比較案と並べて判断すること。

---

## 6. 2 GPU に収める構成の候補 (experts の予算 ≤150GB)【計算】

前提: experts 1 個 = 3×5120×2304 = 35.4M param、全 15,360 個で 543.6B。suh/svh の付加は全体で約 0.68GB。

| 案 | 中身 | experts の容量 | 品質の見込み | 道具 | エンジン |
|---|---|---|---|---|---|
| A1 | 層別 K2/K3 (34 層 K2、6 層 K3)、校正なし (sfxnz) | 146.8GB | KL 0.25〜0.33 級 | Q1、2 GPU で約 2 時間 | diffbot 系 vLLM + xmoe (そのまま) |
| **A2 (本命)** | expert 別 K∈{1.5,2,3,4}、平均 2.15〜2.2、校正あり (sudoingX 方式) | 146〜150GB | KL 0.10〜0.20 級、PPL +3〜6% | Q2、2 GPU で 5〜7 時間 | sudoingX plugin を sm_120 化する。または vcruz305 の mixed-K |
| B | REAP 256E (vLLM で合法な数) + EXL3 3.0bpw (層別) | 136GB | PPL: REAP +5.5% (text mean ranking) に 3bpw の約 +1〜2% | Libertai REAP + Q1 か Q2 | diffbot 系。E=256 のロードは要検証 |
| B' | REAP 320E + 2.6bpw | 約 147GB | REAP +2〜3% (推定) と 2.6bpw | 同上 | 同上 |
| C | 1017 の 2.9bpw (既製品) + host experts 約 46GB (diffbot 方式) | 196GB (VRAM 150 + host 46) | 最良 (PPL +1.8%) | DL 約 211GB (ユーザーの許可が要る) | diffbot 系。RAM が厳しい (PLAN E2) |

- DSpark の drafter experts (384 個、13.6B param) は、MXFP4 のまま 7.2GB、K3 で 5.1GB、K2 で 3.4GB【計算】。
- 予算 (PLAN §2.3 の 2×(97.5−6.9−3.6−7.0−3.6)=152.7GB) は、drafter を 3.6GB/GPU とした値。drafter を K3 にすれば約 2GB 浮き、その分を本体に回せる。

---

## 7. Engram と DSpark の扱い (全経路で共通)

- **Engram**: どの量子化器も触らない。
  - sfxnz: shard 47/48 を hardlink する (`quantize_experts_exl3.py:6`)。
  - sudoingX: 公式の Engram shard を hardlink し、serving 時に `patches/0002-engram-disk-offload.patch` で disk から pread する (CUDA graph でも安全)。
  - LibertAIDAI / PixelML: repo に含めず、1017 本体の 47/48 を使う。sha256 は `76a05106…` / `9dc51433…`。
  - **旧 V4.1-Flash の Engram を流用してはいけない。** 1017 のものと sha が違う (Baekpica のカードに両方の sha が載っている)。
  - 圧縮を選ぶ場合の参考:
    - MXFP4 / NVFP4: cos 0.9934
    - int4: 111GB
    - MXINT-4: 97.6GiB
    - dongnhdev lloyd4 (136B/行): NLL −0.05%
    - pipenetwork: 6bit は FP8 と区別できないが、4bit は free-running で +7.3〜9.2%
- **DSpark (mtp.0-2)**: 1017 には専用の drafter が同梱されている。
  - sfxnz の既定は source (MXFP4)。
  - diffbot は `quantize_mtp_exl3.py` で 3bit (校正なし、beam16、約 9 分)。
  - LibertAIDAI は experts を 4bpw (2.9 版) か 6bpw (3.5 版) にした。
  - Lucebox は「September の drafter は 1017 の target での受理率が下がる」と書いている。必ず 1017 同梱のものを使うこと。

---

## 8. このマシンで動かすまでの手順 (GPU 段階。GLM の停止はユーザーの判断)

1. (CPU、今すぐできる) ダウンロードの完了を確認し、shard 47/48 を含む 48 本の sha256 を照合する (PLAN X1)。
2. (CPU) 校正データを作る。GLM の要求ログを V4.1 の `encoding/encoding.py` でエンコードし、sudoingX の `data/caldata` と混ぜる。日本語、ツール、thinking を含める。
3. (CPU) `ext/sfxnz_recipe/tools/quantize_experts_exl3.py` に 3 つの改修を入れる。
   - bits ガードを外す。
   - `layer_bits` を出力する。
   - MXFP4 の展開を GPU で行う。
   - あわせて、shard と層の対応表を index から作る。
4. (CPU) sudoingX の `dsv41q` を x86 と sm_120 の単体機向けに直す。
   - `DSV41_CUDA_MEM_FRACTION` を外す。
   - Engram の reader を、LIL か sudoingX 0002 の vLLM 拡張から切り離す (Libertai の `stream_model.py` を流用)。
   - `--part 0/2`、`1/2` で GPU0 / GPU1 を 2 プロセスにし、part ファイルの交換をローカル FS で行う。
5. (GPU 約 2 時間) A1 を作る。サイズと KL の下限を確かめる「最短の動作版」である。
6. (GPU 約 5〜7 時間) A2 の pass0 → 感度 → 配分 (`--bpw 2.18 --decode-bpw 2.4 --protect-share 0.2 --floor-layers 37,38,39:3`) → pass1 → 組み立て。
7. (GPU) serving: diffbot `recipe/docker/build.sh` で、`nightly-5555a2d8` ベースの image、xmoe、nktlabs の prestage 移植を用意する。
   - mixed-K を使う場合は、sudoingX plugin を sm_120 化するか、vcruz305 0.5 の `p2b_fused_moe_mk` を使う。
8. (GPU) 評価: `p4_kld.py` で参照実装と比べる。AtomicChat 方式の KL、GSM8K、HumanEval、日本語、ツール、needle も測る。B (REAP 256E + 3bpw) と同じ基盤で比べる。

---

## 9. 未解決事項

- diffbot の xmoe が expert 別 mixed-K に対応しているか (README には K2/K3 の instance しか書かれていない)。vcruz305 の `p2b_fused_moe_mk` と sudoingX の `exl3_smoe` は、SM120 TP2 でまだ検証されていない。
- sfxnz で expert 別の K を使う場合は、plan を読んで expert ごとに `bits` を切り替える改修が要る (現状は 1 回の実行で 1 つの K)。層別の K は shard の選び方で実現できる (§1.2)。
- E=256 / 192 の刈り込み版を、diffbot 系の vLLM + EXL3 MoE が読めるか (Libertai は MXFP4 のまま GB10 TP2 で動作を確認している)。
- PR #449 (exllamav3 の V4.1) で `convert.py -rcp` が通るか。本家にマージされる時期。
- 1017 の 2.2bpw での実際の KL。公開値の多くは 0909 版で、1017 で直接測ったのは Lucebox (AMD 形式) と LibertAIDAI (PPL) だけ。
