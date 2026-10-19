# r2: tpurtell/ds41rt の精読 (DeepSeek-V4.1-Flash-1017 を 2x RTX PRO 6000 Max-Q で動かせるか)

作成日: 2026-10-18 / 対象: <https://github.com/tpurtell/ds41rt> (MIT)
clone 先: `ext/ds41rt`。HEAD は `3067d06` (2026-09-25 19:32 +0800、v15 公開直後)。commit は 1,077 本、初回は 2026-08-21。
初期化した submodule: `third_party/sparkinfer` (`7fcc094`、v15 image のラベルと一致)、`third_party/xgrammar` (`557becf`、v0.2.3)。量子化専用の `gptqmodel` と `transformers` は取得していない。
補助取得: diffbot 2.0bpw 版の量子化レシピ (rev `28b7ab7`) の小さいファイルを `ext/diffbot_exl3_2bpw_recipe/` に置いた。
検証スクリプト: `research/tmp/r2_ds41rt/` (`cmp_tensors.py`、`enc_cmp2.py`、Flash 0909 版の `inference/`)。
凡例: 【実測】はこの調査で手元で確かめた値、【資料】はリポジトリや HF の記載、【計算】は式から出した値、【推測】は根拠が間接的な値。

---

## 0. 結論

1. **ds41rt は「2 GPU だけ、Spark なし」の構成 (`SPARK_COUNT=0`) で動く。ただし EXL3 の約 2.0bpw の重みが必要で、1017 版の対応重みはまだ無い。**
   - 本来の構成は、RTX 1〜2 枚 (coordinator) と DGX Spark 4 台 (routed experts の TP4) の組み合わせである。
   - EXL3 2.0bpw (K2、MCG) のときだけ、40 層の experts をすべて 2 枚の RTX に TP2 で載せる 0-spark 構成が公式に計測されている。
   - ユーザーが挙げた「C1 225 tok/s」はこの構成の値で、正確には **C1 code decode 225.20**。条件は 400W、dSpark、greedy である。
   - 同じ構成の c16 合計は code 1,426、topic 767、counting 1,799 tok/s。code/fable/topic を混ぜた c16 は 390 tok/s。
2. **1017 は ds41rt のローダ契約をそのまま満たす。**【実測】
   - 1017 の `config.json` は ds41rt に埋め込まれた公式 config と全項目一致した。
   - 全 96,085 テンソルの名前、dtype、shape も、ds41rt の契約 (`official-v41-tensors.json`) と一致した。
   - したがって、1017 から ds41rt 形式の EXL3 K2 を作れば読み込めるはずである。
   - **ただし、既に公開されている 1017 の EXL3 はどれも ds41rt では読めない。**
     - PixelML 3bpw は codebook が `mul1`。
     - LibertAIDAI 2.9bpw は `out_scales: always` などの余分なキーを持つ。
     - ds41rt は MCG codebook、`mtp_experts=source`、未知キー無しを厳密に要求する (`v41_exl3.rs:219-286`)。
     - そもそも 2.9〜3.5bpw では 2 枚の VRAM に入らない。
3. **Engram は外さずに使える。表はホストの mmap に置く (FP8、NVMe 上の safetensors をそのまま MAP_PRIVATE で開き、MADV_RANDOM と WILLNEED で先読みする)。**
   - pin しないので、**RAM 128GB でも起動はできる** (diffbot/vLLM の方式は約 300GB の RAM が要る)。
   - ただし、表 203GB の大半はページキャッシュに乗らない。そのため NVMe のランダム読み出しが速度を落とす【推測】。
   - ds41rt が計測に使ったホスト (Threadripper 9970X、空き RAM 174GiB 以上) では、表の多くがキャッシュに乗っていた可能性が高い。
4. **experts をホスト RAM に退避する機能は無い。** 層の残りは Spark に送る設計だけである。
   - したがって、1017 の native (MXFP4 288.8GB) と 2.9bpw 以上の EXL3 は、2 GPU 単独では動かない。
5. **API と機能**
   - 対応しているもの: OpenAI 互換の `/v1/chat/completions` (SSE、tools、並列 tool calls、JSON Schema 制約)、画像 16 枚まで、1M 文脈、prefix cache、dSpark、ブラウザのコンソール。
   - 足りないもの: `/v1/completions`、`/v1/responses`、API キー認証、logprobs、n、penalty 系。並列は 16 が上限である。
6. **GLM-5.3 の置き換えとしての判断: 「条件付きで可」。**
   - 条件は次の 4 つ。
     - (a) 1017 を diffbot のレシピ (uncalibrated、K2、MCG、beam 16) で自分で量子化する。
     - (b) `run.sh` の `--device=/dev/infiniband` を外す。
     - (c) GLM を止めて VRAM と RAM を空ける。
     - (d) 2.0bpw の品質を自分の評価で受け入れる。
   - 速度の見込み【推測】
     - 当環境 (Max-Q 300W、PCIe x8、RAM 128GB) では公式値から 1〜2 割落ちる。
     - c1 code は 180〜210、加重平均は 115〜135 tok/s。
     - c16 合計は code 型で 1,100〜1,350、話題型で 600〜700、混在で 330〜380。
   - GLM の c16 合計 384 と比べると、生成内容次第で同等〜3 倍強になる。
   - prefill は 3.5〜5k tok/s。
   - PLAN の推奨案 A (自前 EXL3 約 2.2bpw + 本家 vLLM + Engram パッチ) よりも、**既製のエンジンをそのまま使える分、早く速い可能性が高い**。PLAN の最有力候補に繰り上げるべきである。

---

## 1. ds41rt の正体

### 1.1 構成 (architecture.md、docs/ENGINEERING.md)

- **独自ランタイム**である。Rust の daemon と API (約 38 万行)、CUDA/CuTe の AOT カーネル (`native/`、`.cu` 約 3.2 万行)、SparkInfer (b12x 系の fork、Apache-2.0) でできている。vLLM や SGLang には依存しない。
- **coordinator** (x86_64、RTX 1〜2 枚) が担当するもの:
  - API、tokenizer、スケジューラ
  - 注意 (CED、圧縮 KV、indexer)
  - mHC、Engram の gather と射影
  - router、shared expert
  - 画像、dSpark 全段、語彙 head、sampling
  - 根拠: `architecture.md:9-26`
- **Spark worker** (ARM64、SM121、4 台) が担当するのは、coordinator に入りきらない層の routed experts だけである。intermediate 次元の TP4 で分担し、RoCE でつなぐ。
- **dual RTX モード**の動き:
  - 2 枚で依存関係ごとに仕事を分ける。
  - shared experts と下位層の routed experts は TP2、語彙 head は行方向の TP2 で分割する。
  - dSpark の transformer は RTX1 に置く。
  - VRAM に入るだけの層を下から順に RTX に載せ、残りを Spark に回す (`architecture.md:55-63`)。
- **0-spark モード**: `SPARK_COUNT=0` で起動すると、2 枚の RTX と `RTX_EXPERT_LAYERS=40` が必須になる。peers には存在しない 127.0.0.1:1〜4 を渡し、実際には接続しない (`run.sh:138-141, 359-364`、`scripts/release-common.sh:397-400`)。
- 開発機 `raptor` の構成は次のとおり。当環境とは CPU、PCIe、RAM、電力が違う。
  - Threadripper 9970X (`docs/tp-ep-host-overhead.md:28`)
  - RTX PRO 6000 Blackwell **Workstation** を 400W に制限して使用 (`docs/release-v7-exl3-k2-homogeneous.md:72-73`)
  - 量子化の開始時点で空き RAM 174GiB (`quantization/PLAN.md:1494`)

### 1.2 リリースと image

- 最新は **v15**。image は `ghcr.io/tpurtell/ds41rt-coordinator:v15` と `ds41rt-spark-expert:v15`。
- coordinator image の中身【実測: ghcr の manifest と config を匿名で取得】
  - amd64、圧縮サイズ 10.46GB (104 層)
  - ラベル: `io.ds41rt.cuda_arch=120`、`io.ds41rt.sparkinfer.revision=7fcc094…`、`org.opencontainers.image.revision=bd06bec…`
  - ベースは NGC `nvcr.io/nvidia/pytorch:26.05-py3`。`CUDA_VERSION=13.2.1`、PyTorch `2.12.0a0`、`CUDA_DRIVER_VERSION=595.58.03`
  - 当機のドライバは 615.78.08 なので足りている【実測: nvidia-smi】。
- release image には EXL3 の 2 系統 (`k23` と `k34`) の AOT パッケージが入る (`scripts/build-release-artifacts.sh:84-87`、`docker/Dockerfile.release:101,124`)。
- 注意点: **EXL3 の性能を測った最後の版は v7** (2026-09-19) である。
  - v14 と v15 は official checkpoint だけを測っている。
  - v15 での EXL3 0-spark の動作は未確認なので、v7 image を退避先に残しておく。

---

## 2. 対応する重み形式と 1017

### 2.1 形式の一覧 (`rust/crates/ds41rt-loader/src/v41_catalog.rs:482-503`、`run.sh:184-209`)

| 形式 | 検出方法 | experts の置き場 | 2 GPU のみ (0-spark) |
|---|---|---|---|
| native (公式 MXFP4 experts、FP8 の非 expert) | 既定 | RTX は下位層、残りは Spark TP4 | **不可** (288.8GB) |
| NVFP4 (ModelOpt W4A4、`nvidia/DeepSeek-V4.1-Flash-NVFP4`) | `moe_quant_algo=NVFP4` | 同上 | 不可 |
| EXL3 raw publication (exllamav3 系、例 diffbot 2.0bpw) | `quant_method=exl3` で `quantize_config.json` が無い | RTX TP2 (0-spark) か Spark | **可 (K2 のとき)** |
| EXL3 staged (ds41rt 自前の manifest、例 `wrldsuksgo2mars/DeepSeek-V4.1-EXL3-K3.25-v1`) | `quantize_config.json` に `meta.ds41rt.schema=ds41rt.v41-routed-exl3.v1` | 同上。混合ビット (隣り合う 2 段) と NVFP4 PLE も表せる | K3.25 は入らない |

**EXL3 raw publication の受け入れ条件** (`v41_exl3.rs:219-339`)
- `bits` は 2..5 の整数で、全 projection で同じでなければならない。
- `codebook=mcg` と `mtp_experts=source` (DSpark の experts は MXFP4 のまま) が必須。
- 許されるキーは列挙された 11 個だけで、それ以外があると拒否する。
- `non_routed_quantization` は公式の FP8 と同じでなければならない。
- **Engram は FP8 の公式表**でなければならない。raw publication は `ple_quantization=None` 固定である。
- **層ごとにビットを変える `layer_bits` は raw では拒否される**。混合にするには staged manifest を書く必要がある。

**VRAM の容量**
- K2 は 1 rank あたり 69,122,129,920B/40 層で、64.4GiB になる (`scripts/select-release-gpus.py:36-48` のコメント)。
- 0-spark の起動直後の占有は、RTX0 が 90.94GiB、RTX1 が 91.53GiB だった (`docs/release-v7-configurations.svg` の注記)。
  - RTX0 の内訳: experts 64.38、KV 7.34、実行 2.44、その他 16.79
  - RTX1 の内訳: experts 64.38、KV 4.90、実行 2.68、その他 19.59
- したがって、**2.0bpw がほぼ上限**である。KV を約 2GiB まで削っても、混合配分で約 2.2bpw までしか上げられない【計算】。

### 2.2 1017 の互換性【実測】

| 項目 | 結果 | 根拠 |
|---|---|---|
| `config.json` | ds41rt が埋め込む `official-v41-config.json` と**全項目一致** | 再帰比較スクリプト。差分 0 件 |
| テンソル契約 | 48 shard の全ヘッダを読み、**96,085 テンソルすべてで名前・dtype・shape が一致** (不一致 0) | `research/tmp/r2_ds41rt/cmp_tensors.py`。ds41rt は `expected.len()==96085` と 48 shard を要求する (`v41_catalog.rs:437, 543`) |
| tokenizer.json / tokenizer_config.json | V4.1-Flash (0909) と**バイト一致** | `cmp` |
| `inference/` の engram.py、vision.py、kernel.py、image_processor.py、config.json | 0909 と**バイト一致** | `cmp` |
| `inference/model.py` | 差分は 14 行 (下記) | `diff` |
| `encoding/encoding.py` | tool 結果の結合、content_blocks、画像の順序の処理が変わった。**OpenAI 形式の入力 (公式 fixture 5 件と、自作の多ターン agent 会話、ツール結果で終わる途中ターン) では、0909 版と出力が完全一致した** | `research/tmp/r2_ds41rt/enc_cmp2.py` |
| 重み (Engram 表、DSpark を含む) | 全部新しい。Engram を 0909 から流用してはいけない (arch_spec §0) | — |

**model.py の差分 (0909 → 1017)**
- (1) indexer の `shared_attn.index_k = self.k_cache` が、`latent is not None` の条件の外に出た。グループが未完のときの参照実装の不具合修正と見られる。ds41rt は独自の cache 管理なので影響は無いと見る【推測】。
- (2) **注意出力 `o` を wo_a に入れる前に FP8 で量子化して戻す処理 (`act_quant`) が追加された** (本体の Attention と DSpark の Attention の両方。`ref/model.py:787, 1073`)。
  - ds41rt の wo_a は、もともと FP8 の W8A8 GEMM である。入力の BF16 を scratch 内で FP8 値と行スケールに量子化している (`v41_attention_output.rs:1`「grouped FP8 wo_a」、`native/include/ds41rt_v41_fp8.h:7-15`)。
  - したがって、1017 の新しい挙動にはむしろ近い。
  - ただし量子化の粒度 (block 長、UE8M0) が一致しているかは未確認なので、要検証【推測】。

**model_id の制約**
- native の経路では、`MODEL_ID` が `deepseek-ai/DeepSeek-V4.1-Flash` でなければ拒否される (`v41_config.rs:18-21`)。
- EXL3 の経路では、内部の定数で検証するので名前は自由である (`v41_exl3.rs:296`)。
- 起動スクリプトが要求するのは `$HF_HOME/hub/models--<org>--<name>/snapshots/<40〜64 桁の hex>/` という配置だけで、重みの hash は照合しない (`scripts/release-common.sh:522-534`)。

### 2.3 公開済みの量子化物で使えるもの【実測: HF API】

| repo | 対象 | 形式 | ds41rt | 0-spark |
|---|---|---|---|---|
| `diffbot/…-EXL3-3bpw-2x-RTX-PRO-6000` の rev `28b7ab71` (旧名 `…-EXL3-2.0bpw-…`。改名後も rev は取得できる) | **V4.1-Flash 0909** | K2、MCG、mtp source、Engram は FP8 公式表 (shard 47/48、各 101.5GB)。合計 358.1GB | ○ (ds41rt の v7 で計測済み) | ○ |
| `PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw` | 1017 | `codebook: mul1` | ×。MCG 必須の検査で拒否される | × (容量も超える) |
| `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` | 1017 | exllamav3 1.6.2 の形式 (`avg_bits`、`out_scales: always`、`head_bits`) | ×。未知キーで拒否される | × (196GiB) |
| LibertAIDAI 3.5bpw、LakyFx 3.5bpw TensorFold | 1017 | 3.5bpw | × | × |

**1017 用の K2 MCG 版は存在しない** (2026-10-18 時点。HF で `V4.1-Flash-1017` を検索して確認)。

---

## 3. Engram の置き方と必要なホスト RAM

**実装** (`rust/crates/ds41rt-loader/src/mapped_rows.rs:62-91, 173-176`、`engram_prefetch.rs`、`docs/ds41-engram-pipeline.md`、`docs/ENGINEERING.md`「Engram and mapped storage」)
- checkpoint の shard を `mmap(MAP_PRIVATE)` で開き、`MADV_RANDOM` を付ける。
- 入力トークンが決まった時点でハッシュから行番号を出し、背景のワーカーがページごとに `MADV_WILLNEED` を出す。
- 次に gather ワーカーが、行を固定サイズの pinned staging (上限 256MiB、2 lane × 2 層 = 4 枠) に集める。
- その後 GPU に上げ、FP8 から戻す処理、wkv の射影、gate を GPU で行う。
- Engram の射影重み (wkv FP8 [25600,6144] など) は層あたり 162MB で、VRAM に常駐する。
- **表そのものは VRAM にも pinned RAM にも置かない** (`DEVELOPER.md`「Do not preload the entire tables onto a GPU」)。
- 画像のトークン区間は n-gram の履歴を切り、Engram の寄与を 0 にする。
- 投機トークンが棄却されたときは、履歴を進めない。
- **NVFP4 の PLE** (Engram を E2M1、16 要素ごと E4M3 scale、表ごとの FP32 global scale にしたもの。103GiB) にも対応する。
  - ただし staged manifest の経路でしか使えない。
  - tool eval は 153〜157/176 で、FP8 と同程度だった (`docs/release-v5-performance.md:189-216`)。

**RAM 128GB で動くか**
- **起動はできる**【資料 + 推測】。
  - pin が要るのは KV の退避キャッシュ (`--host-cache-bytes auto`) と staging だけである。
  - auto の値は、dual native で 6.75GiB、NVFP4 2x で 15.25GiB だった (`docs/release-v7-published-memory-evidence.log:38`)。
  - 0-spark の EXL3 は GPU 側の KV が小さいので、auto だと十数 GiB になると見込む。`--host-cache-bytes 4GiB` か `--prefix-cache-entries` で絞れる。
- **速度の不安**
  - 表は約 203GB、ページキャッシュに使えるのは 85〜100GB 程度【推測】である。
  - しかも 1 行 264B に対して 4KiB ページを丸ごと抱えるので、容量効率は約 1/15 になる (PLAN §2.4 の D2)。
  - ヒット率は 35〜45% 程度にとどまり、残りは NVMe のランダム読み出しになる【推測】。
  - 当機の NVMe は 4K ランダム読みで約 23 万 IOPS が上限だった (PLAN §1.1)。
  - 見積もり【計算】
    - prefill 2,048 トークンの 1 波: 約 48 行 × 0.67 (重複除去) × 0.6 (ミス) × 2,048 ≈ 4 万 IO → 約 0.17 秒。GPU の計算 (約 0.37 秒) と半分程度は重なる。
    - c16 の 1 round: 約 80 トークン × 48 × 0.6 ≈ 2.3k IO → 約 10ms。round 時間は約 56ms。
  - ds41rt 自身の計測は、ページを部分的に追い出した状態で、18 トークン (432 行) の WILLNEED gather が 2.17ms だった (`docs/ds41-engram-io-comparison.md`)。
  - ただし開発機の NVMe は「約 14GB/s」級 (`quantization/PLAN.md:43`) で、当機の DRAM-less の S880 より速い。
- **NVMe に表を置く運用は前提どおり**で、特別な設定は要らない。io_uring の経路は実験だけで採用されていない (seccomp の追加も不要)。
- 対策の候補 (どれも ds41rt の改修が要る)【推測】
  - (i) PLAN D3 のアプリ側の行キャッシュと、4KiB に揃えて並べ直した表
  - (ii) staged manifest で NVFP4 PLE (103GiB) にして、キャッシュ効率を約 2 倍にする
  - (iii) RAM の増設
  - Engram そのものは外さない。

---

## 4. experts のホスト退避

- **無い。** `ds41rt-hostcache` crate は、終わったターンや prompt の KV スナップショットを pinned RAM に退避するためのものである (`rust/crates/ds41rt-hostcache/src/lib.rs:1-7`)。
- coordinator に載らない層は Spark に送る設計だけである。0-spark では 40 層すべてが VRAM に入ることが必須条件になる (`run.sh:138-141`)。
- 帰結: LibertAIDAI 2.9bpw のように、experts の約 23% を host に出す構成 (PLAN の E2) は ds41rt では作れない。

---

## 5. 並列 (バッチ) と c16 の性能

- 同時実行は 1〜16 (`--concurrency`、既定 16、`README.md:585`)。
- 実行 lane は 2 本で、遠隔 expert の待ちと手元の計算を交互に重ねる。
- prefill は既定で 2,048 トークンずつ (80〜4,096 で変更可)。
- prefix の再利用はトークン radix で行い、ページを COW で共有する。部分一致のときは、encoder の末尾 128 トークンだけを作り直す。
- 計測条件は 3 サンプルの中央値で、`temperature=0`。reasoning code は high effort で、それ以外は thinking off である。

### 5.1 README の見出しの表 (全 8 列、`README.md:34-39`)

単位は tok/s。prefill は完走した表の中で最良のセル。decode は C1 dSpark。

| Measurement | Official 1x | Official 2x | NVFP4 1x | NVFP4 2x | EXL3 5090+2-spark | **EXL3 2x6000 0-spark** | Official TP6 1x | Official TP6 2x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefill | 7,824 | 8,355 | 5,237 | 7,371 | 2,013 | **5,597** | 7,131 | 8,577 |
| Counting decode | 182.39 | 219.86 | 144.66 | 192.42 | 161.98 | **337.00** | 174.54 | 231.51 |
| Weighted decode | 104.69 | 129.67 | 81.74 | 100.78 | 88.20 | **146.24** | 96.83 | 110.19 |
| C1 code decode | 146.12 | 179.86 | 112.55 | 145.96 | 122.50 | **225.20** | 139.56 | 159.34 |

- 「Official」と「NVFP4」は RTX と Spark 4 台の構成、TP6 は Spark 6 台。
- EXL3 の 2 列は v7 の値、Official の decode は v14、prefill は v6。
- **当環境で再現できる構成は「EXL3 2x6000 0-spark」の列だけ**である。

### 5.2 EXL3 2x6000 0-spark の詳細 (`docs/release-v7-exl3-k2-performance.md`)

**内容別の C1 dSpark**

| Case | 1 RTX 32GiB + 2 Spark | **2 RTX (0-spark)** |
|---|---:|---:|
| Code | 122.50 | **225.20** |
| Code with reasoning | 93.49 | 161.63 |
| Math | 118.02 | 200.54 |
| Fable | 55.42 | 79.72 |
| Hello | 67.87 | 119.43 |
| Topic | 68.55 | 99.31 |
| Natural JSON | 94.91 | 161.48 |
| Schema JSON | 98.86 | 160.70 |
| Multilingual | 70.89 | 104.32 |
| Counting 1–200 | 161.98 | 337.00 |

**並列 (合計 tok/s)。最初の出力から最後の完了までで測り、受付の待ちも含む**

| 並列 | 1x counting | 1x code | 1x topic | **2x counting** | **2x code** | **2x topic** |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 163 | 131 | 76 | 330 | 247 | 130 |
| 2 | 261 | 197 | 124 | 442 | 328 | 180 |
| 4 | 475 | 365 | 236 | 771 | 545 | 299 |
| 8 | 716 | 535 | 383 | 1,297 | 934 | 471 |
| **16** | 1,101 | 878 | 577 | **1,799** | **1,426** | **767** |

**その他の指標**

| 指標 | 1x | 2x (0-spark) |
|---|---:|---:|
| 混在トラフィック (code/fable/topic) c4 | 135 | 234 |
| 混在トラフィック c16 | 184 | **390** |
| dSpark なし C1 code | 49.06 | 61.96 |
| dSpark なし C1 counting | 50.09 | 62.96 |
| 起動 (API が使えるまで) | 7.0 s (完全起動は 36.6 s) | 17.0 s |
| 起動直後の GPU 使用量 | 29,056 MiB | 94,550 / 95,304 MiB |

**文脈を保持したままの decode (加重、2x)**: 0K で 138、2K で 150、32K で 131、64K で 128、128K で 145、256K で 121。

**prefill (2x、0-spark、tok/s)**

| 保持済み文脈 | +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|
| 0K | 3,561 | 4,206 | 5,493 | 5,597 | 5,551 | 5,486 |
| 32K | 3,005 | 3,701 | 4,261 | 4,616 | 4,832 | 4,910 |
| 64K | 2,681 | 3,365 | 3,907 | 4,309 | 4,501 | 4,567 |
| 128K | 2,191 | 2,823 | 3,371 | 3,714 | 3,973 | 4,051 |
| 256K | 1,536 | 2,100 | 2,558 | 2,967 | 3,192 | 3,304 |

**品質**
- 完了チェックは 28/30。2 件は reasoning code が 4,096 トークンの上限に達し、最終回答が空だった。この 2 件も速度の値に含まれている。
- tool eval (high effort) は 159、154、156/176。公式の full checkpoint は v4 で 155、160、156/176 だった。

### 5.3 README の Official 構成の表 (参考。Spark 4 台が前提)

**内容別** (`README.md:107-118`。Official Flash API は DeepSeek のホスト API を 2026-09-24 に測った値)

| Case | 1 RTX target | 1 RTX dSpark | 2 RTX target | 2 RTX dSpark | Official Flash API |
|---|---:|---:|---:|---:|---:|
| Code | 60.63 | 146.12 | 61.76 | 179.86 | 328.92 |
| Code with reasoning | 59.87 | 115.78 | 60.60 | 138.99 | 265.65 |
| Math | 59.79 | 129.15 | 60.91 | 155.34 | 230.37 |
| Fable | 59.36 | 67.59 | 60.56 | 84.98 | 128.86 |
| Hello | 58.37 | 77.20 | 60.43 | 113.79 | 151.88 |
| Topic | 60.00 | 84.05 | 60.50 | 102.32 | 189.53 |
| Natural JSON | 59.72 | 133.54 | 60.48 | 157.00 | 266.83 |
| Schema JSON | 59.12 | 138.74 † | 60.17 | 148.66 † | HTTP 400 |
| Multilingual | 59.18 | 87.46 | 60.24 | 106.52 | 176.78 |
| Counting 1–200 | 60.81 | 182.39 | 61.64 | 219.86 | 417.01 |

**並列** (`README.md:157-163`)

| 並列 | 1 RTX counting | 2 RTX counting | 1 RTX code | 2 RTX code | 1 RTX topic | 2 RTX topic |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 183.05 | 218.43 | 143.36 | 187.76 | 86.28 | 108.02 |
| 2 | 268.07 | 325.51 | 221.85 | 284.08 | 131.75 | 168.36 |
| 4 | 480.50 | 595.51 | 381.69 | 516.72 | 251.27 | 311.28 |
| 8 | 768.33 | 992.72 | 610.86 | 861.79 | 382.46 | 487.66 |
| 16 | 1,326.87 | 1,622.69 | 1,057.35 | 1,394.33 | 605.29 | 834.23 |

**混在トラフィック** (`README.md:167-173`)

| 並列 | 1 RTX | 2 RTX |
|---|---:|---:|
| 1 | 128.34 (127.87–146.76) | 171.68 (171.23–180.94) |
| 2 | 119.97 (116.85–128.93) | 167.00 (162.91–172.04) |
| 4 | 160.51 (142.95–179.96) | 214.11 (197.18–218.97) |
| 8 | 171.70 (167.87–172.92) | 247.43 (240.65–252.11) |
| 16 | 212.53 (203.73–213.00) | 314.35 (308.25–319.88) |

**文脈を保持したままの decode** (`README.md:146-153`)

| 保持済み文脈 | 1 RTX | 2 RTX |
|---|---:|---:|
| 0K | 102.79 | 124.91 |
| 2K | 108.50 | 129.83 |
| 32K | 100.37 | 118.29 |
| 64K | 99.76 | 123.70 |
| 128K | 99.13 | 119.95 |
| 256K | 94.70 | 108.22 |

**prefill (Official、v6、tok/s)**

| 保持済み文脈 | 1 RTX +1K | +2K | +4K | +8K | +16K | +32K | 2 RTX +1K | +2K | +4K | +8K | +16K | +32K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0K | 3,005 | 4,031 | 6,950 | 7,493 | 7,801 | 7,824 | 4,725 | 6,369 | 7,825 | 8,185 | 8,355 | 8,290 |
| 32K | 2,463 | 3,326 | 5,379 | 6,140 | 6,598 | 6,863 | 3,683 | 5,153 | 6,373 | 6,930 | 7,207 | 7,251 |
| 64K | 2,242 | 3,103 | 4,961 | 5,665 | 6,151 | 6,361 | 3,221 | 4,626 | 5,673 | 6,226 | 6,507 | 6,567 |
| 128K | 1,928 | 2,718 | 4,205 | 4,914 | 5,343 | 5,530 | 2,588 | 3,678 | 4,655 | 5,202 | 5,486 | 5,556 |
| 256K | 1,463 | 2,122 | 3,195 | 3,797 | 4,156 | 4,306 | 1,770 | 2,697 | 3,336 | 3,847 | 4,126 | 4,229 |

**配置と cache 容量** (`README.md:177-180`)

| Configuration | RTX expert layers | Spark resident / active layers / budget | Global FP4 source pool | RAM cache | Combined usable tokens | Prompt / turn retention |
|---|---:|---:|---:|---:|---:|---:|
| 1 RTX | 0–4, TP1 | 40 / 35 / 100 GiB each | 16.701 GB / 18,736,128 usable + 28,672 private-tail tokens | 3.25 GiB pinned / 2,235,904 logical tokens | 20,972,032 | 20 / 20 |
| 2 RTX | 0–19, TP2 | 20 / 20 / 100 GiB each | 13.091 GB / 14,680,064 usable + 28,672 private-tail tokens | 6.75 GiB pinned / 6,291,968 logical tokens | 20,972,032 | 20 / 20 |

**起動と起動直後のメモリ**
- 起動時間: 1 RTX が 58.63 s、2 RTX が 31.71 s。
- 1 RTX: Loaded 94,590 MiB、Free 2,661 MiB、reserve 2,048 MiB。
- 2 RTX: RTX0 が 95,132 / 2,119 / 800 MiB、RTX1 が 95,738 / 1,510 / 800 MiB。

**dSpark の受理率** (v14、C1。括弧内は 1 round で出るトークン数の平均)

| Content | 1 RTX | 2 RTX |
|---|---:|---:|
| Code | 91.10% (5.37) | 88.05% (5.19) |
| Code with reasoning | 75.41% (3.72) | 70.88% (3.61) |
| Math | 78.54% (4.50) | 73.51% (4.49) |
| Fable | 54.94% (1.86) | 56.63% (2.14) |
| Hello | 53.97% (2.15) | 44.23% (2.31) |
| Topic | 61.29% (2.54) | 57.82% (2.62) |
| Natural JSON | 71.21% (2.96) | 74.07% (3.61) |
| Schema JSON | 89.62% (4.17) | 82.64% (4.03) |
| Multilingual | 57.55% (2.45) | 56.30% (2.73) |
| Counting | 99.27% (5.90) | 99.47% (5.94) |

**native のツール呼び出し** (v4)
- 155/176 (124/138 + 31/38)、160/176 (126 + 34)、156/176 (123 + 33)。
- v15 では 155/176 (120 + 35)。

**sampling 別の加重 decode** (1 RTX + Spark 4 台)
- v12 の GPU sampler: greedy 94.17、T0.2/top-p 0.95 で 89.65、T0.7/top-p 0.9 で 89.73、T0.7/min-p 0.05 で 90.91、T0.7/top-k 40 で 89.64。
- v11 の CPU sampler: 95.46、80.58、83.29、88.99、87.74。
- 内容別の値は `README.md:240-252`、`314-327` にある。
- **sampling を有効にすると、greedy より約 5% 落ちる。**

---

## 6. DSpark と投機

- 1017 の MTP (DSpark: 3 段、128 experts、top-3、5 トークンの block) をそのまま使う。
  - 目標層 37/38/39 の残差平均を条件にする。
  - 埋め込みと語彙 head は本体と共有する。
  - 根拠: `architecture.md:42-51`
- EXL3 raw でも、DSpark の experts は MXFP4 のまま (`mtp_experts=source`) で RTX1 に常駐する。
- 検証長は request ごとに毎 round 決める。決め方は「帯域バランス」方式で、層ごとの expert バイトを、lane の実測 timing から常時当てはめた実効帯域で割って予測 round 時間を出す。オフラインの較正は要らない (`README.md` の Engineering 節、`docs/dspark-bandwidth-policy.md`)。
- `--dspark-draft-limit 7` (実験的)、`--dspark-fixed`、`--no-dspark` が使える。
- sampling 時は sample-and-match で検証する。rejection sampling ではない (`README.md:440-450`)。
- v15 で、tools 付きの request でも投機が効くようになった。2 RTX で C1 は 54〜63 から 98〜111 tok/s に上がった。
  - **v7 の EXL3 の tool eval は、投機が効かない状態で測った値**である。
- SM120 の adaptive verification が hang する問題 (FlashInfer #5015) は、独自カーネルなので関係しない。

---

## 7. API、ツール、画像、長文脈

- **エンドポイント**: `/v1/chat/completions`、`/v1/models`、`/health`、`/v1/stats`、`/` (コンソール)、`/v1/console` (WebSocket) (`rust/crates/ds41rt-api/src/native_v41.rs:95-103`)。
  - **`/v1/completions`、`/v1/responses`、embeddings は無い。**
  - **API キー認証も無い。** 必要なら前段に proxy を置く。
- **パラメータ** (`openai.rs:27-80`)
  - 受け付けるもの: stream、stream_options.include_usage、max_tokens / max_completion_tokens、min_tokens、ignore_eos、temperature、top_p、top_k、min_p、seed、stop、response_format (text / json_object / json_schema)、tools、tool_choice、parallel_tool_calls、thinking / enable_thinking、reasoning_effort (none / low / high / max)
  - logprobs、n、presence / frequency / repetition penalty は持たない。
  - 未知のフィールドは無視する (`deny_unknown_fields` は無い)。
- **既定値が GLM と違う点**
  - thinking が既定で **high**。
  - temperature を省くと **greedy** になる (`README.md:426-428`)。
  - 返る model 名は `deepseek-ai/DeepSeek-V4.1-Flash` に固定 (`scripts/release-common.sh:5`)。
- **ツール**
  - DSML 形式で、並列呼び出しに対応する。
  - strict schema は XGrammar で logits をマスクする。
  - prefix に完全一致したときは、保持した logits を今の文法で選び直す。
- **画像**
  - 1 prompt に 16 枚まで。data URL と HTTP(S) URL を受け付け、JPEG / PNG / WebP / GIF を読める。
  - ViT と aligner は checkpoint のものを RTX で動かす (`README.md:641`)。
  - diffbot/vLLM の経路はテキストだけなので、ここは ds41rt の明確な利点である。
- **長文脈**
  - `--max-context-tokens` は最大 1,048,576、出力は最大 393,216。
  - KV の形式: 圧縮 KV は FP4 E2M1 + E4M3 の 16 要素ごとの scale、SWA は FP8、index は別形式の FP4。
  - 保持できるのは、完了したターンと prompt スナップショットを各 20 件 (LRU)。
  - 0-spark の decode は 256K 保持でも計測されている (§5.2)。
- **chat テンプレート**
  - crates.io の `deepseek-recipe-encoding = "=0.1.0"` (DeepSeek 公式、2026-09-10) を使う。
  - 1017 の encoding.py の変更は、OpenAI 形式の入力では出力が変わらなかった【実測、§2.2】。
  - 生の content_blocks を使う入力では違いが出る可能性がある。

---

## 8. 依存、ビルド、SM120

**最短の経路は公開 image を使うこと**
- `ghcr.io/tpurtell/ds41rt-coordinator:v15` (10.5GB) に CUDA 13.2.1、PyTorch 2.12、SM120 の AOT カーネル、EXL3 の k23/k34 が全部入っている。
- ホスト側に要るものは次のとおり。
  - Docker と NVIDIA Container Toolkit
  - `jq`、`ss`、`ssh`、`curl`、`sha256sum`、`python3`。当機には全部ある【実測: `command -v`】。
  - `third_party/sparkinfer` の取得。`run.sh` が image のラベルと照合する。取得済みで照合も通った【実測: `verify-sparkinfer-source.py` → `revision=7fcc094…`】。
- GPU の選び方: `nvidia-smi topo -p2p r` で双方向の peer read を確かめ、空き VRAM を見積もって選ぶ (`scripts/select-release-gpus.py:123-138, 184-197`)。
  - 当機の peer read は GPU0⇄GPU1 で `OK`、接続は PHB【実測】。
  - **GLM が VRAM を使っている間は、選択の段階で落ちる。**

**当機でそのままでは動かない点**
- `run.sh:468` が coordinator を `docker run … --device=/dev/infiniband` で起動する。当機には `/dev/infiniband` が無い【実測】。
- そのため、0-spark 用にこの引数を外した起動スクリプト (自前の `run-local.sh`) が要る。0-spark では RDMA を使わない。

**ソースからビルドする場合**
- `./build.sh` は、coordinator を手元で、Spark の ARM64 image を Spark 上でビルドし、配布までする。当機では使えない。
- coordinator だけなら次の手順になる。いずれも GPU を使う段階で行う。
  - `scripts/build-release-artifacts.sh` と `docker/Dockerfile.release` で作る (NGC pytorch 26.05 ベース、CuTe の AOT を coordinator GPU で書き出す)。
  - Rust の daemon は `cargo build -p ds41rt-daemon`。当機には cargo が無い。
- **SM120 対応**
  - image のラベルは `cuda_arch=120`。開発機も RTX PRO 6000 Blackwell (SM120、188 SM)。
  - Max-Q も同じ GB202 で 188 SM なので、カーネルの互換性に問題は無いと見る【推測】。
  - **400W の Workstation 版との差** (当機は 300W 上限)
    - prefill のような計算律速の処理は 15〜25% 落ちる見込み【推測】。
    - diffbot は Max-Q の 300W で、同じ 2.0bpw を vLLM で動かして c1 113 tok/s だった。

---

## 9. 量子化ツール (1017 を自分で量子化する方法)

### 9.1 ds41rt 自前のツール (`quantization/`、GPTQModel の fork と transformers の fork を使う)

- 作れるのは **K3.25 (K3 を基本に、誤差の大きい 1/4 を K4 へ)、calibrated、DSpark の experts も EXL3** という固定のレシピである (`quantization/mixed_recipe.py:10-14, 40`)。
- **2 枚の RTX と Spark 4 台 (ostrich/dodo/emu/kiwi) が必須**と、コードに明記されている。
  - `distributed_search.py:20-21`: 「V4.1 search requires the two RTX slots and four Spark endpoints」
  - `run_quantization.py:126-129`
- 実績【資料】
  - 本体 40 block は 1 block 36〜38 分で、計約 25 時間かかった。
  - host のピーク RSS は約 31GiB。
  - 作業領域として NVMe を最大 380GiB 程度使った (空きが 1,002GiB から 620GiB に減った)。
  - 根拠: `quantization/PLAN.md:1494, 1713-1750`
- 0909 を前提にしている (`Source model: deepseek-ai/DeepSeek-V4.1-Flash`)。
- K2/K3 で 2.2bpw 以下にし、RTX だけで動かすには、レシピと分散探索を改修する必要がある。工数は大きい。

### 9.2 diffbot 2.0bpw のレシピ (ds41rt が 0-spark の計測に使った重みの作り方)

- 場所: `ext/diffbot_exl3_2bpw_recipe/recipe/quantize/quantize.sh` と `quantize_experts_exl3_sm120.py` (HF rev `28b7ab71` から取得)。
- **uncalibrated** で、Hessian は単位行列。MCG の trellis、K2、greedy beam 16。
- 2 枚の GPU に 20 層ずつ分けて、shard 単位で処理する。中断しても shard 単位で再開できる。
- 依存:
  - sfxnz のレシピ (`github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark`。HEAD は `3de9146`)
  - exllamav3 `5be88657`
  - vLLM day-0 image (`deepseekv41-flash-0909`) をベースにした `dsv41-flash-exl3-sm120` image
- 出力:
  - 新しく書くのは experts の EXL3 で約 137GB。
  - 残りの shard (Engram の 47/48 を含む) は**元の 1017 からの hardlink** になる。同じファイルシステムが必要。
  - 当機の空きは 526GB【実測: `df -h`】。
- **1017 にもそのまま使えるはずである**【推測】。
  - 理由は、テンソルの配置と shard の割り当てが 0909 と同一であること (§2.2)。
  - スクリプトは「MoE layer N は shard N+3」と仮定している。1017 の index でもこの割り当ては成り立っていた。routed expert のテンソルで外れるものは 0 件で、Engram は shard 47/48 だけにある【実測】。
- 所要時間はレシピに書かれていない。数時間〜1 日程度と見る【推測】。
- 層ごとにビットを変える `--layer-bits` もあるが、ds41rt の raw 経路は `layer_bits` キーを拒否する (§2.1)。
  - 混合にするには、ds41rt の staged manifest (`quantize_config.json` の `meta.ds41rt`、`tensor_storage` に全 projection) を自分で書く必要がある。
  - staged の形式は DSpark の experts も EXL3 で持つことが前提なので、DSpark の experts も変換する必要がある (`v41_exl3.rs:142-151, 390-420`)。

### 9.3 品質の材料

**V4.1 0909 での資料**
- diffbot 2.0bpw: GSM8K-200 97.5%、HumanEval 91.5% (thinking off)
- ds41rt 上の tool eval: 154〜159/176
- 完了チェック: 28/30
- 2.0bpw の KL は 0.36 (PLAN §0-3、WE §3.2)

**ds41rt の K3.25**
- fixed-history の top-1 一致は 95/132 (72%) (`docs/release-v5-performance.md:220-226`)。
- 3.25bpw でもこの程度なので、**2.0bpw の劣化は評価で必ず確かめる**必要がある。

---

## 10. 当環境での見込み (EXL3 1017 K2 を自作した場合)【推測】

| 項目 | ds41rt の公表値 (400W WS、x16、RAM 多) | 当環境の見込み | 主な差の理由 |
|---|---:|---:|---|
| C1 code | 225 | 180〜210 | Max-Q 300W、Engram のキャッシュミス、x8 |
| C1 加重 | 146 | 115〜135 | 同上 |
| c16 code 合計 | 1,426 | 1,100〜1,350 | TP2 の行和を PCIe x8 (P2P 約 20GB/s) で送るのに 1 round 約 3ms【計算】、Engram の IO |
| c16 topic 合計 | 767 | 600〜720 | 同上 |
| c16 混在合計 | 390 | 330〜380 | 同上 |
| prefill 最良 | 5,597 | 3,500〜5,000 | 300W、Engram の IOPS (§3) |
| 1M 文脈 | 可 | 可 | KV は VRAM 上で約 12GiB |
| VRAM | 94.5 / 95.3 GiB | 同程度 | 2.0bpw がほぼ上限 |
| host RAM | (未公表) | pinned は 4〜16GiB、Engram はページキャッシュ任せ | GLM の停止が前提 (/dev/shm の 49GB を解放) |

- GLM-5.3 (c1 224、c16 384) と比べた見込み
  - c1 は code 系でほぼ同等、雑談系では 0.5〜0.6 倍。
  - c16 は code 系で約 3 倍、雑談系で約 1.7 倍、混在で同等。
  - ただし計測方法が違う。GLM の 384 がどの負荷で測った値かを合わせて比べ直すこと。

---

## 11. 動かすまでの手順 (GPU を使う段階で、ユーザーの判断で GLM を止めてから)

1. **1017 のダウンロード完了と sha256 の照合を待つ** (PLAN X1)。
2. **量子化用の image を作る** (CPU とネットワーク)
   - `ext/diffbot_exl3_2bpw_recipe` に、HF rev `28b7ab71` の `recipe/` 一式を揃える。今あるのは一部だけである。
   - `docker/build.sh` で、sfxnz のレシピと exllamav3 `5be88657` を含む `dsv41-flash-exl3-sm120` を作る。
   - **vLLM の day-0 image を取得する必要がある**。ユーザーに確認すること。
3. **GLM を止めて量子化する**
   - `MODELS_DIR=$MODELS SRC=DeepSeek-V4.1-Flash-1017 DST=DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw recipe/quantize/quantize.sh`
   - shard N+3 の割り当ては確認済み。`--link-only` による hardlink が効くこと (同じファイルシステムであること) は事前に確かめる。
4. **HF キャッシュの形に並べる**
   - 例: `HF_HOME=hf` とし、`hub/models--local--DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw/snapshots/<40 桁 hex>/` に hardlink で置く。
   - コンテナには `$HF_HOME` だけを読み取り専用でマウントするので、外を指す symlink は使えない。
   - `config.json` の `quantization_config` が `{quant_method: exl3, bits: 2, codebook: mcg, mtp_experts: source, mtp_experts_start_layer: 40, weight_block_size, non_routed_quantization}` だけであることを確かめる (`v41_exl3.rs:265-286`)。
5. **coordinator image を取得する** (`docker pull ghcr.io/tpurtell/ds41rt-coordinator:v15`、10.5GB)。v15 で動かない場合に備えて `:v7` も候補にする。
6. **ローカル設定を書く**
   - 書く値: `MODEL_ID=local/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw`、`MODEL_REVISION=<hex>`、`EXPERT_FORMAT=exl3`、`SPARK_COUNT=0`、`RTX_GPUS=2`、`RTX_EXPERT_LAYERS=40`、`DSPARK=on`、`HOST_CACHE_BYTES=4GiB`、`PREFIX_CACHE_ENTRIES=8`、`COORDINATOR_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-coordinator:v15`、`ADDR=0.0.0.0:8000`
   - Spark のホスト欄は空にする。
7. **`run.sh` を複製して `run-local.sh` を作る**
   - coordinator の `docker run` から `--device=/dev/infiniband` を外す (`run.sh:468`)。
   - `HF_HOME=… ./run-local.sh --config local.config --dry-run` で確認してから本番を起動する。
8. **検証する**
   - `scripts/api-smoke.sh`、`scripts/bench-ds41-release-decode.py`、`scripts/bench-ds41-concurrent-api.py`、`scripts/qualify-ds41-tool-eval.py` を実行する。
   - 画像と 1M needle (`scripts/qualify-ds41-native-needle.py`) を確かめる。
   - Engram を見る: `RUST_LOG=info,ds41rt::timing=debug` で gather の待ち時間を見る。`/proc/vmstat` の major fault と、NVMe の IOPS も見る。
   - PLAN の X7 の評価 (KL、GSM8K、HumanEval、日本語、agent の再生) で、2.0bpw の品質を判定する。

---

## 12. リスクとブロッカー

| 種別 | 内容 | 対処 |
|---|---|---|
| **ブロッカー** | 1017 の K2 MCG EXL3 が存在しない | §9.2 で自作する (GPU 時間と、vLLM day-0 image の取得が要る) |
| **ブロッカー** | `run.sh` が `/dev/infiniband` を要求する | 1 行直す |
| ブロッカー (運用) | GLM と同時には動かせない (VRAM が全部埋まる。RAM も GLM の 49GB を解放する必要がある) | 切り替えて運用する |
| 高 | 2.0bpw の uncalibrated 品質 (KL 0.36 級)。reasoning code が上限まで止まらない事例が 2/30 | X7 で評価する。ds41rt の staged manifest で混合 (≤2.2bpw) にする余地はある |
| 高 | EXL3 経路は v7 以降に測り直されておらず、v15 での動作は未確認 | v7 image を退避先として用意する |
| 中 | RAM 128GB のため Engram がページキャッシュに乗りきらず、NVMe の IOPS で律速される | 計測したうえで、行キャッシュの追加か NVFP4 PLE (staged) を検討する |
| 中 | 1017 の wo_a FP8 量子化の粒度が ds41rt の実装と同じかどうか未確認 | 層 0 の参照比較 (`scripts/qualify-ds41-attention-output.py` 系) |
| 中 | 開発者 1 人のプロジェクトで、Spark 4 台の構成が主対象。0-spark は副次的 | 必要な部分は fork して保守する (MIT) |
| 低 | API の機能不足 (completions、responses、認証、logprobs、penalty)。並列は 16 が上限 | 前段に proxy を置く |
| 低 | Max-Q 300W と PCIe x8 による性能低下 | §10 の幅に含めた |

---

## 13. PLAN.md への影響

- PLAN の案 E (既存のエンジンをそのまま使う) は「どれも 2 GPU では成り立たない」としていた。これを修正する。
  - **ds41rt の 0-spark + 自作の 1017 K2 (案 E3)** を加え、最優先にする。
  - これは案 A (EXL3 約 2.2bpw を全部 VRAM、Engram を FP8 で外さない、DSpark あり) のうち、Engram 以外を既に実装済みのものに当たる。
  - 公表値でも、案 A の見込み (c1 90〜140、c16 500〜700) を上回る。
- Engram の扱いは D2 (mmap とページキャッシュ) に当たる。D3 (行キャッシュと並べ直した表) は、ds41rt の `EngramPipeline` に足す改修として位置づけ直す。
- 次にやること
  - X2 (Engram のヒット率) の結果で、128GB での実効 IO を決める。
  - X7 で 2.0bpw と 2.2bpw 混合の品質を確かめる。
  - そのうえで、ds41rt E3 と案 A を同じ評価で比べる。
