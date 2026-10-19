# DeepSeek-V4.1-Flash-1017 調査: Engram と技術報告 (推論の観点)

作成日: 2026-10-18 / 対象: `deepseek-ai/DeepSeek-V4.1-Flash-1017`
GPU は使っていない。稼働中の `glm53v` には触れていない。ダウンロードしたのは小さいファイルだけ (PDF 1.8MB / 0.77MB / 1.06MB、tokenizer.json 6.4MB、Gutenberg テキスト 2.3MB)。

## 一次資料 (ローカルに保存したもの)

| 略号 | 資料 | ローカルパス |
|---|---|---|
| **TR** | DeepSeek-V4.1-Flash 技術報告 "Pushing the Limits of KV Cache Compression" (51 ページ、2026-09-10 作成) <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/DeepSeek_V41_Tech_Report.pdf> | `research/tmp/techreport/DeepSeek_V41_Tech_Report.pdf`、ページ番号付きテキスト `trp.txt` |
| **EP** | Engram 論文 "Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs" arXiv:2601.07372v2 (2026-07-12 版) <https://arxiv.org/abs/2601.07372> | `research/tmp/engram_paper/engram_2601.07372.pdf`、`ep.txt` |
| **DP** | DSpark 論文 "Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation" arXiv:2607.05147v1 <https://arxiv.org/abs/2607.05147> | `research/tmp/dspark_paper/dspark_2607.05147.pdf`、`dp.txt` |
| ref | 公式 inference/ (`ref/*.py`)。ダウンロード中のモデルに含まれる `inference/` と diff を取り、一致を確認した | — |
| enc | `$MODELS/DeepSeek-V4.1-Flash-1017/encoding/{README.md,encoding.py}` | — |
| recipe | deepseek-recipe <https://github.com/deepseek-ai/deepseek-recipe> (main、最終 push 2026-09-10) | `research/tmp/recipe/` |

二次資料 (WebFetch で取得した要約なので、数値は原典に当たって再確認すること):
- vLLM recipe <https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash>
- SGLang cookbook <https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1>
- LMSYS blog <https://www.lmsys.org/blog/2026-09-10-deepseek-v41>
- SemiAnalysis "Engram's embedding entendre" <https://newsletter.semianalysis.com/p/engrams-embedding-entendre-codesign> (後半は有料のため未読)

ページ番号は PDF の物理ページで示す。TR では印字ページと一致する。

---

## 0. 要点

1. **Engram の表は 2 個 (層 1 と層 14)、合計 202.76 GB (189 GiB)。** 1 行は FP8 256B に UE8M0 scale 8B を足した 264 B。1 トークン 1 層あたり 24 行を読むので、全体で **12,672 B/トークン (12.4 KiB)**。行番号は入力トークン列だけから決まる。このため、該当層を計算する前にいつでも先読みできる (TR p13 §2.4.2、p18 §3.1.3、EP p6 §2.5、ref `engram.py:159-184`)。
2. **公式の推論配置はホスト DRAM 常駐とバックグラウンド RDMA 先読み。** 第 1 モジュール (層 1) の先読みは第 1 ブロックの計算と重ねる (TR p13)。Engram 論文の実測値は次のとおり。H800 で 100B パラメータ表を全量ホスト DRAM に置いた場合、性能劣化は **4B-Dense で −1.9%、8B-Dense で −2.8%** (EP p18 Table 4)。Zipf 分布を使う多段キャッシュ (HBM / DRAM / NVMe) は提案として書かれているだけで、**ホット率やヒット率の数値は論文にも技術報告にも無い** (EP p6)。
3. **ホット率は自分で測った (CPU のみ、参考値)。** 公式のハッシュを再現し、圧縮語彙 99,092 とテーブル行数 384,006,168 / 384,016,682 が config と完全に一致することを確認した。英文と Python コード約 46.5 万トークンから作ったキャッシュでは、別の 15.7 万トークンに対するヒット率が次のとおりだった。
   - 上位 10 万行 (26 MB) で 18%
   - 上位 100 万行 (264 MB) で 29%
   - 学習側に現れた全 733 万行 (1.9 GB) で 37%
   - n-gram 次数別のヒット率は 2-gram 68%、3-gram 32%、4-gram 12%

   3/4-gram はロングテールが厚く、小さなキャッシュではほとんど当たらない。
4. **このマシンには全体が収まらない。** VRAM 192 GB と DRAM 128 GB を足しても 320 GB で、重み 510 GB に足りない。Engram 以外の重みだけでも 307 GB あり (routed experts 288.8 GB)、VRAM を超える。vLLM の `--engram-config '{"cpu_offload":true}'` は表 189 GiB 全体を pinned host に置く前提なので、DRAM 128 GB のこの機械ではそのまま使えない。推測だが、Engram は NVMe (mmap / io_uring) に置き、DRAM と VRAM をホット行キャッシュにするのが現実的な線になる。より大きな問題は experts が VRAM に入らないことにある (§5)。
5. **CSA2 と FP4 KV により、グローバル KV は 890 B/トークン** (TR p1、p4)。config から再計算しても 890 B になった (§2.4)。**1M (1,048,576) 文脈 1 本で 0.93 GB、16 本でも 14.9 GB。** これとは別に、SWA の KV が 1 系列あたり固定で約 2.9 MB 要る。KV キャッシュはこのモデルでは制約にならない。
6. **DSpark は MTP の後継で、block 5 の半自己回帰ドラフタ。** 3 層 MoE (128 experts、top-3)、Markov head (rank 256)、confidence head と、負荷に応じて検証長を決めるスケジューラで構成される (TR p13-14 §2.4.3、DP §3、§5)。V4.1 の受理長の公式値は無い。vLLM と SGLang はベンチマーク用に合成受理長 3.51 を使っている。
7. **サンプリングは temperature 1.0 / top_p 1.0** (README)。エージェント評価では top_p 0.95 を使っている (TR p32、p35)。reasoning effort は 1〜100 の数値で、公式の対応は **low=50 / high=75 / max=100、既定 high=75** (TR p30 Table 2、enc `encoding.py:443-453`)。**vLLM recipe はこれと異なる対応 (low=25 / high=50 / xhigh=75 / max=100、既定 50) を記載している。** 不一致なので要確認。

---

## 1. Engram

### 1.1 構造 (技術報告 + 参照実装)

- **配置と規模。** Engram の総パラメータは 196B で、2 モジュールに均等に分けられている。N-gram の次数は {2,3,4}、各次数に 8 hash heads があり、次数ごとの埋め込み次元は計 2048 (= 8 heads × 256)。1 head は約 16M エントリの表を引き、表サイズは互いに異なる素数にしてある。埋め込み表と K/V 射影はともに FP8。配置は層 1 と層 14 (0 始まり) で、理由は「学習パイプライン段のメモリ均衡のため」(TR p13 §2.4.2)。
- **元論文からの変更は 2 点** (TR p13)。
  1. short causal conv を省いた。理由は「推論スタックの複雑さに見合わない」。
  2. 埋め込みの最適化に momentum + Sinkhorn balancing を使う (TR p15-16 §2.5、Algorithm 1)。
  - Engram の学習率は 5 倍 (TR p22 §4.2.2)。
- **ハッシュの手順** (`engram.py`)。
  - トークン ID を正規化 (NFKC→NFD→アクセント除去→小文字化→空白の畳み込み) し、圧縮語彙 99,092 に写す (`engram.py:17-61`)。
  - 各位置で、末尾が自分になる 2〜4-gram を multiply-XOR の rolling hash で計算する。(次数, head) ごとに素数の剰余を取り、オフセットを足す (`engram.py:177-184`)。
  - 層ごとの乗数は `np.random.default_rng(10007*layer_id)` から作る (`engram.py:64-83`)。
  - 画像トークンは DEAD (−1) として扱い、n-gram が画像をまたがないようにする (`engram.py:138`、`model.py:1253-1255`)。
- **ゲートと合流** (`model.py:328-366`)。
  - 24 行 × 256 = 6144 次元を 1 本の `wkv` (FP8、6144→25600 = 5120×(hc4+1)) で射影し、mHC の 4 系統それぞれの key と、共有の value 1 本を得る。
  - 系統ごとに `sigmoid(signed_sqrt(RMS 正規化した内積))` でゲートを作り、`h + gate·value` を残差に加える。
  - Engram は対象層の **Attention より前**で残差に加わる (`model.py:1265-1266`)。
- **重みの実体** (`ref/headers.json` から集計)。

| テンソル | dtype | shape | バイト |
|---|---|---|---|
| `layers.1.engram.embed.weight` | F8_E4M3 | [384006168, 256] | 98.31 GB |
| `layers.1.engram.embed.scale` | F8_E8M0 | [384006168, 8] | 3.07 GB |
| `layers.14.engram.embed.weight` | F8_E4M3 | [384016682, 256] | 98.31 GB |
| `layers.14.engram.embed.scale` | F8_E8M0 | [384016682, 8] | 3.07 GB |
| `layers.{1,14}.engram.wkv.weight` | F8_E4M3 | [25600, 6144] | 0.157 GB × 2 |

  - 1 行 = 256 B (FP8) + 8 B (32 要素ごとに UE8M0 scale) = **264 B**。
  - 1 層の行数は 24 本の素数区間 (各約 16.0M) の和で、約 3.84 億行。2/3/4-gram がそれぞれ約 1.28 億行 (約 33.8 GB) を占める。
  - 1 トークンあたりの読み出しは 24 行 × 264 B × 2 層 = **12,672 B (12.4 KiB)**。SemiAnalysis の値「24 rows × 2 layers, ≈12.4 KiB/token」と一致する。

### 1.2 推論時の配置: 公式の記述

- **TR p13 §2.4.2:** "During inference, deterministic addressing enables embeddings to be prefetched from host memory via background RDMA transfers, with prefetching for the first module overlapping computation in the first Transformer block."
- **TR p6 (Introduction):** 推論システムの最適化の一つとして "sharded Engram embedding tables" を挙げている。
- **TR p18 §3.1.3 (学習時):**
  - 表は engram parallel group 内で行方向に分割する。
  - インデックスは入力トークン列だけで決まるので、各パイプライン段がマイクロバッチを処理する前に、ローカルバッチ全体の prefetch を発行する。
  - FP8 のまま格納・取得し、取得した値と scale をそのまま次の GEMM に渡す。
  - **RL rollout 中は表を GPU メモリに常駐させる。** 理由は、ホストメモリの断片化による OOM を避けるため。
- **参照実装** (`model.py:296-325`)。表を TP rank 数で行分割して各 GPU に置き、lookup 結果を `all_reduce` する。オフロードは実装していない。TP2 なら 1 GPU あたり 101 GB になり、96 GB の GPU には入らない。

### 1.3 Engram 論文に書かれたシステム設計と実測値

- **EP p6 §2.5:** 学習時は表を GPU 間で分割し、All-to-All で集める。推論時は "prefetch-and-overlap" を使う。インデックスは forward の前に分かるので、PCIe 経由でホストメモリから非同期に取得し、前段の層の計算で遅延を隠す。この遅延を隠すために、Engram を置く層を選ぶ必要がある。
  - 深い層に置けば隠せる時間は長くなる。一方、モデル性能は早期の層に置くほうが良い (§6.2)。
- **Zipf と多段キャッシュ (EP p6)。** 原文は "natural language N-grams inherently follow a Zipfian distribution … a small fraction of patterns accounts for the vast majority of memory accesses. This statistical property motivates a Multi-Level Cache Hierarchy: frequently accessed embeddings can be cached in faster storage tiers (e.g., GPU HBM or Host DRAM), while the long tail of rare patterns resides in slower, high-capacity media (e.g., NVMe SSD)."
  - **ここは設計方針の記述にとどまる。ヒット率、ホット集合のサイズ、NVMe の実測は書かれていない。**
- **実測 (EP p17-18 §6.4 Table 4)。** nano-vLLM ベースのハーネスで測定。条件と結果は次のとおり。

| 項目 | 内容 |
|---|---|
| 条件 | 100B の Engram 1 層を第 2 ブロックに挿入し、表は全量ホスト DRAM。H800、512 系列、長さ Uniform(100,1024) |
| 4B-Dense | 9,031.62 tok/s → 8,858.28 tok/s (−1.9%) |
| 8B-Dense | 6,315.52 tok/s → 6,140.02 tok/s (−2.8%) |
| 論文の注記 | 「通信量は表サイズではなく活性スロット数に比例」。HBM キャッシュを使わず全取得を PCIe に通した保守的な条件 |

- **Engram を抜いたときの影響 (EP p17 §6.3)。** 推論時に Engram の出力をゼロにすると、事実知識系のベンチマークは元の性能の 29〜44% しか残らない (TriviaQA は 29%)。読解系は 81〜93% を保つ (C3 は 93%)。
  - 注意: これは Engram-27B 実験での値で、V4.1 で測ったものではない。
- **SemiAnalysis の追試** (二次資料、無料部分のみ)。
  - V4.1-Flash で Engram を除去すると、CRUXEval の answer loss が 0.2848 から 0.3093 bits/token に悪化した。GSM8K は誤差範囲だった。
  - B200 上で、表を SSD (mmap、GDS 無効、未最適化の vLLM fork) に置いた構成は、DRAM 構成に全点で負けた。125 tok/s/user 付近の tokens/$ は DRAM 121M に対して SSD 52M。
  - B300 では DRAM オフロードで TP4 を TP2 に減らせ、Pareto が最大 1.6 倍改善した。HBM に戻しても改善しなかった。

### 1.4 既存エンジンの実装状況 (二次資料)

- **vLLM (0.30.0 以降で対応、推奨 0.31.0 以降)。**
  - `--engram-config '{"cpu_offload":true}'` で表を pinned host DRAM に置き、**UVA で直接読む**。出力は変わらない。
  - B200 TP2 の c1 では `"use_thp":true` を足す。DEP 構成では `"embedding_across_dp":true`。
  - オフロード時のスループット差は記載が無い (vLLM recipe)。
- **SGLang (preview イメージのみ)。**
  - `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`。`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared|per_rank` を選べる。
  - shared は各 rank が完全なホストコピー 1 部から gather する方式で、all-reduce が要らない。
  - 4×GB300 では KV 容量が +36%、decode スループットと TTFT は同等、greedy で 28/28 一致した (LMSYS blog、SGLang cookbook)。
- **どちらの実装も「表全体が DRAM に載る」ことが前提になっている。** NVMe 階層やホット行キャッシュの既存実装は見つからなかった。SemiAnalysis の mmap fork は非公開のようだ。

### 1.5 行アクセス分布の自前測定 (Zipf とホット率の目安)

公式の論文にも報告にも数値が無いため、CPU だけで測った。

- **方法。**
  - スクリプトは `research/tmp/engram_sim.py`、結果は `research/tmp/sim_out.json`。
  - `engram.py` のハッシュを NumPy で再実装した。正規化、素数レイアウト、乗数、rolling XOR はすべて公式と同じにしている。
  - 再現の検証として、圧縮語彙 99,092 (config の `engram_compressed_vocab_size` と一致) と、層ごとの行数 384,006,168 / 384,016,682 (config の `engram_num_embeddings` と一致) を確認した。
- **コーパス。**
  - 学習側 (キャッシュ作成用): Gutenberg 英語小説 4 冊、Engram 論文本文、Python 標準ライブラリ a-m 約 150 ファイル。計 465,065 トークン。
  - 評価側: 別の小説 2 冊、技術報告本文、標準ライブラリ n-z 60 ファイル。計 157,498 トークン。
  - 日本語と中国語は含んでいない。

| 層 1 (層 14 もほぼ同じ) | 値 |
|---|---|
| 学習側の lookup 数 / ユニーク行 | 11.16M / 7.33M (66% がユニーク)。1 回しか出ない行が 6.36M |
| 学習側で、上位 1万 / 10万 / 100万 / 400万 行が占めるアクセス割合 | 10.6% / 21.8% / 43.3% / 70.2% |
| 評価側の lookup 数 / ユニーク行 | 3.78M / 2.55M (同一テキスト内でも 67% がユニーク) |
| 学習側上位 1万行 (2.6MB) をキャッシュした場合の評価側ヒット率 | 9.3% (2-gram 25%、3-gram 2.6%、4-gram 0.14%) |
| 上位 10万行 (26MB) | 18.2% (2g 43%、3g 10%、4g 1.1%) |
| 上位 100万行 (264MB) | 28.9% (2g 60%、3g 22%、4g 4.9%) |
| 上位 400万行 (1.06GB) | 32.0% (2g 63%、3g 26%、4g 7.3%) |
| 学習側の全 733万行 (1.94GB) | 37.0% (2g 68%、3g 32%、4g 12%) |

- **解釈 (推測を含む)。**
  - 2-gram は Zipf 的でよく当たる。3-gram と 4-gram はロングテールが厚い。
  - 「少数のパターンがアクセスの大半を占める」(EP p6) は、2-gram については成り立つ。行数で 2/3 を占める 3/4-gram には当てはまらない。
  - コーパスが 46 万トークンと小さいので、実トラフィックで数十 GB 級のキャッシュを作ればヒット率はもっと上がるはずだ (未検証)。それでも、全 lookup の 3〜5 割は DRAM か NVMe の深い階層まで行くと見ておくのが安全だろう。
  - n-gram の頻度は 8 heads で共通なので、ヒット率は head 間で同じになる。表の中で 2-gram が占める区間だけでも 1 層あたり約 33.8 GB ある。

### 1.6 このマシン (DRAM 128GB / NVMe 1 本 / PCIe Gen5 x8) での Engram 配置案 (推測)

- **帯域は問題にならない。**

| 場面 | 必要帯域 (概算) |
|---|---|
| c16、検証長 5 の decode | 1 step で 80 位置 × 12.4 KiB ≈ 1 MB |
| 1 万 tok/s の prefill | 約 127 MB/s |
| 参考: PCIe Gen5 x8 実効 | 約 20 GB/s |

- **NVMe で問題になるのは IOPS とレイテンシ。**
  - 264 B の行 1 本に 4 KiB のページ read が 1 回要るので、約 15.5 倍の読み増しになる。
  - キャッシュが無い場合、1 トークンで 48 IO。
  - 1 万 tok/s の prefill でミス率 60% なら約 29 万 IOPS が要る。decode c16 なら約 1.5k IO/step で、余裕がある。
  - **S880 の 4K ランダム read 性能は未実測。fio で QD 32〜256 の値を測る必要がある。**
- **レイテンシの隠し方。**
  - 層 1 の行は、層 0 (SWA のみの軽い層) が終わるまでに必要になる。NVMe の 1 回あたり約 80〜150 µs を隠すには、トークン ID が確定した時点で即座に IO を発行する必要がある。
  - decode なら DSpark が draft を出した直後、prefill ならチャンクの受付時点で発行する。
  - 層 14 側は 13 層分の余裕がある。
- **配置の候補。**
  1. DRAM に 2-gram 区間を中心にホット行を置き、残りは NVMe に置く。DRAM 予算は GLM 停止時に約 100 GB (`/dev/shm` の 49 GB も解放される)。
  2. VRAM に余裕があれば、最上位のホット行 (数 GB) を GPU 側に置く。
  3. 頻度表は実トラフィックで更新する。
- **DRAM を experts に回す必要が出たら (§5)、Engram は大部分 NVMe に置くことになる。** SemiAnalysis の結果 (SSD は DRAM に全敗) を考えると、品質を保ったまま速度を出す難易度は高い。
- **Engram を外す、または層 14 だけ残すといった削減は、品質劣化が大きいと予想される** (EP p17、SemiAnalysis)。勧めない。

---

## 2. CSA2 (Compressed Sparse Attention 2) と KV キャッシュ

### 2.1 V4 からの変更

- V4 は CSA と HCA の混成だった。V4.1-Flash は **CSA2 だけ**を使う (TR p4)。HCA は廃止された。
- 圧縮器を簡素化した (TR p10 §2.3)。
  - CSA では比率 m の圧縮に 2m エントリを重ねて使い、絶対位置埋め込みも入れていた。CSA2 は重なりと絶対位置埋め込みを除去した。
  - indexer K は main KV から射影して作る。別の圧縮経路は持たない。
  - 比率 1 (圧縮なし) も特殊ケースとして含む。
- 全層が SWA (window 128) を併用する。最初の 2 層は SWA だけを使う (TR p7 §2.1)。

### 2.2 3 つのモード (TR p10-11 §2.3.1、Figure 4)

全モードで、その層の main Q と SWA KV は自前で計算する。

| モード | main KV / indexer K | Top-K インデックス |
|---|---|---|
| **Full** | 自前で計算する | 自前の indexer で全域を走査して選ぶ |
| **Reindex** | 直近の Full 層のものを再利用する | 自前の indexer Q で共有 K を再採点する |
| **Reuse** | 再利用する | 直近の Full または Reindex 層のものを再利用する (indexer の計算は無い) |

### 2.3 層の割り当て (TR p21-22 §4.2.1、Figure 3) と config・参照実装の対応

| 層 | 区分 | 圧縮比 | モード | config での対応 |
|---|---|---|---|---|
| 0-1 | encoder | — | SWA のみ | `compress_ratios[0:2]=0` |
| 2-7 / 8-13 / 14-19 | encoder (6 層 × 3 群) | 2 | 先頭 Full、残り 5 層 Reuse | `kv_source=[2,8,14]`、`index_source` に 2,8,14 |
| 20-23 | decoder 第 1 群 | 1 | 20 が Full (候補プールの生成元)、21-23 Reuse | `kv_source` に 20、`candidate_source_layer_id=20` |
| 24-27 / 28-31 / 32-35 / 36-39 | decoder 第 2〜5 群 | 1 | 先頭 Reindex、残り 3 層 Reuse | `index_source` に 24,28,32,36 |
| 40-42 | DSpark | — | SWA のみ | `compress_ratios[40:43]=0` |

- **Causal Encoder-Decoder (CED)** (TR p9 §2.2)。
  - decoder (l > L/2) のグローバル KV は、各層の hidden からではなく **encoder の最終 hidden H_{L/2} から射影する**。
  - その結果、prefill は encoder の 20 層だけで済む。活性パラメータは prefill 8B、decode 16B。
  - 参照実装でも、層 20 の圧縮器が層 20 の入力 (= 層 19 の出力) から KV を作る構造になっている (`model.py:740-764`)。
- **Hierarchical Sparse Indexer** (TR p11-12 §2.3.2、Figure 5)。
  - decoder の最初の Full 層 (20) は全域を採点し、自分の top-512 を選ぶ。同時に、8 位置のブロックごとに最大スコアを取り、上位 2,048 ブロック (= 16,384 位置) を候補プールにする。
  - 以降の Reindex 層 (24/28/32/36) は、このプール内だけを採点する。
  - これにより、深い層の indexer のコストは文脈長に依存しなくなる。
  - この仕組みは post-training で導入され、学習時と推論時で同じ制限をかけている。
  - 参照実装: `select_candidate_blocks` (`model.py:584-611`)、`uses_candidates` (`model.py:503`)。最新位置のブロックは常にプールに入れる。
- encoder の Full 層 (2/8/14) の indexer は毎回全域を走査する。1M 文脈では 50 万の圧縮位置を走査する。

### 2.4 KV キャッシュのサイズ (1 トークンあたり)

形式は TR p14 §2.4.4 と ref で確認した。
- **main KV:** FP4 E2M1 に、16 チャネルごとの E4M3 scale を付ける (NVFP4 から 2 段目の global scale を除いた形)。RoPE を掛けた後に量子化する (`model.py:761`)。
- **indexer K:** MXFP4 (32 要素ごとの E8M0 scale、`model.py:546`)。
- **SWA KV:** FP8。32 要素ごとの UE8M0 scale (`model.py:708`)。量子化に敏感なため FP8 を維持している (TR p14)。

**グローバル KV (トークン数に比例する部分)** は、ソース層 4 つ分だけを持つ。

| ソース層 | 比率 | エントリ/トークン | main KV (512ch: 256B + scale 32B = 288B) | indexer K (128ch: 64B + 4B = 68B) |
|---|---|---|---|---|
| 2, 8, 14 | 2 | 各 0.5 (計 1.5) | 432 B | 102 B |
| 20 | 1 | 1.0 | 288 B | 68 B |
| **計** | | 2.5 | 720 B | 170 B |

- **合計 890 B/トークンで、TR の公称値 890 B (p1、p37) と一致する。** V4-Flash の約 1/4、V1 の 1/437 (TR p1 Figure 1)。

| 文脈長 | 1 系列 | 16 系列 |
|---|---|---|
| 128K | 0.117 GB | 1.87 GB |
| 1M (1,048,576) | **0.933 GB** (0.869 GiB) | **14.9 GB** |

- **SWA KV (系列あたり固定):** 43 層 × 128 × (512 B + 16 B) ≈ **2.9 MB**。これに Compressor の未完グループ状態 (比率 2 の層で 2×512 fp32 程度) が加わる。
- 参照実装はキャッシュを BF16 の tensor として確保し、量子化と逆量子化を擬似的に再現している。メモリ量は上の値の約 3.6 倍になる。実運用のエンジン (vLLM/SGLang) は FP4 で詰めて格納する。vLLM は `--kv-cache-dtype fp8` と `indexer_kv_dtype=mxfp4` を設定している。
- **永続 KV (prefix cache)。** SWA KV は永続化せず、グローバル KV だけを SSD に残す。SWA は再生で近似復元する ("Encoder SWA Bounded Replay"、TR p19-20 §3.2.1-3.2.2)。
  - SWA の再構成は、直近 n_win=128 トークンだけを再生して近似する。厳密に復元するには L×n_win トークンの再生が要る。
  - TR は品質劣化を「ほぼ無し」としている (p20)。
  - SGLang ではオプトインで、`--enable-encoder-swa-bounded-replay` と `--enable-decoder-swa-bounded-replay` を指定する。8K プロンプト × 8 の prefill で、8×H200 で 1.56 倍、4×GB300 で 1.37 倍になった。AIME 2026 は 453/480 で、on と off で同じだった (LMSYS blog)。
  - TR は限界として、CSA2 の選択誤りと SWA 近似が「未検証の境界条件」で性能を落とす可能性を挙げている (p37 §6)。
- **decode の計算量。** 4K から 1M に 256 倍伸ばしても、Decode FLOPs は 1/4 しか増えない (TR p5 Figure 2)。
- 1M 文脈での indexer K 読み出しを概算すると、1 トークン 1 系列あたり 170 MB 程度 (推測)。内訳は Full 3 層 × 34 MB と、層 20 の 68 MB。

---

## 3. DSpark (投機デコード)

- **構造** (TR p13-14 §2.4.3、DP §3.1、ref `model.py:1023-1160`)。
  - 3 層の Transformer ブロック (SWA 128、MoE は 128 experts の top-3)。
  - **1 回の forward で 5 位置分の base logits を並列に出す。** 入力は anchor (直前に確定したトークン) と noise token (id 128799) × 4 (`model.py:1131-1138`)。
  - 本体の層 37/38/39 の attention 入力 (hc 平均) を連結し、`main_proj` に通す (`model.py:1268-1269`)。
  - **Markov head** は低ランク (r=256) の bigram 遷移バイアス `W1[x_{k-1}]·W2` を逐次に加え、ドラフト内の依存を補う (DP §3.1 p6)。
  - **confidence head** は各位置の条件付き受理確率を出す (`model.py:1092-1101`)。
- **MTP との関係。**
  - V3/V4 の MTP は本体と一緒に事前学習していた。V4.1 は **事前学習で MTP を省き**、事前学習の後に本体を凍結して DSpark だけを学習する。post-training でも本体と並行して学習を続けるが、DSpark の勾配は本体に流さない (TR p8 §2.1、p14 §2.4.3)。
  - **重みは `mtp.*` 名前空間に格納されている** (`model.py:1103-1104`、config の `num_nextn_predict_layers: 3`)。MTP experts 7.2 GB というのはこの DSpark の experts のことだ。
  - vLLM は「MTP は非対応 (V4.1 で廃止)、`method: dspark` を使う」としている。drafter の experts は約 14B パラメータ。
- **検証のスケジューリング** (DP §3.2.2 Algorithm 1、§5.2)。
  - 各位置の confidence を累積積にして生存確率 a_{r,j} を出す。
  - 起動時に、バッチトークン数ごとの step/s を測った表 SPS(B) を作っておく。
  - 全リクエストの候補を a_{r,j} の降順で貪欲に採用し、Θ = τ·SPS(B) が最大になる長さで打ち切る。
  - 本番環境では、zero-overhead scheduling や CUDA graph と両立させるため、**2 step 前の confidence から容量 K を決める非同期方式**を使う。
- **報告されている効果。**
  - オフライン、block 7、ドラフト 5 層の条件で、受理長 τ (bonus を含む) は次のとおり (DP Table 1 p11)。
    - Qwen3-8B: math 5.0〜6.2、code 5.2〜5.5、chat 3.2〜3.7
    - DFlash 比 +16〜18%、Eagle3 比 +27〜31%
  - V4 の本番環境 (block 5、Markov head) では、MTP-1 に比べて matched throughput で **per-user 速度が V4-Flash で +60〜85%、V4-Pro で +57〜78%** (DP §5.4 p18-19)。
  - 中程度の並列 (V4-Flash で 200 未満) では、1 リクエストあたりの検証長は MTP-1 の固定 2 から 4〜6 に伸びる。負荷が上がると自動的に短くなる (DP Figure 8 p19)。
- **V4.1 での受理長の公式値は無い。** vLLM と SGLang はベンチマークで合成受理長 3.51 を使っている (golden AL)。実際の値は自分のワークロードで測る必要がある (vLLM recipe)。
- **推奨設定** (二次資料)。
  - vLLM: `--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic", "enable_adaptive_verification":true}'`。実運用では `rejection_sample_method: block` を使う。synthetic はベンチマーク専用。
  - SGLang: `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`。
- **参照実装は投機ループを持たない。** `forward_spec` を呼ぶ箇所が無い (`model.py:129-131` のコメント)。generate.py は通常の自己回帰だけを行う。
- DSpark の層は SWA だけを使うので、長文脈でもドラフトのコストは一定。DP の limitation は「受理率が低い要求でも、ドラフト 1 回分の固定コストはかかる」(p20)。
- Engram との関係 (推測): 検証する draft の各位置にも Engram の 48 行 lookup が要る。draft のトークン ID は検証 forward の前に確定するので、先読みは可能。

---

## 4. サンプリング、思考モード、reasoning effort、エンコーディング

- **推奨サンプリング** (README): temperature 1.0、top_p 1.0、context 1M、max_tokens 256K 以上。
  - TR の評価条件: 推論ベンチは T=1.0 / top_p=1.0。コードエージェントと視覚エージェントは T=1.0 / top_p=0.95 (TR p32 §5.3.1、p35 Table 4 の注記)。視覚エージェントは文脈 512K で評価している。
  - inference README の対話例は `--temperature 0.6`。これは単なる例示と思われる。
- **思考モード** (enc README、`encoding.py`)。
  - `thinking_mode="thinking"` のとき、プロンプトは `<｜Assistant｜><think>` で終わる。chat モードでは `<｜Assistant｜></think>` で終わる。
  - `drop_thinking` (既定 True) は直近の user より前の reasoning を落とす。ただし tools が定義されていると自動で無効になり、全ターンの reasoning を保持する (`encoding.py:785-788`)。
- **reasoning effort。**
  - 1〜100 のスカラーで、thinking モードのときだけ、会話の先頭に `<｜System｜>` ブロックとして 1 回入る。
  - 文字列は `"Reasoning Effort: {b} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"` (`encoding.py:443-446`)。TR p29 の本文とは文言が少し違う。実装側を正とする。
  - 公式 API の対応は max=100 / high=75 / low=50 (TR p30 Table 2)。`encoding.py:448-453` も low 50 / high 75 / max 100、既定 high。
  - deepseek-recipe は Low→50、High・Xhigh・未指定→75、Max→100 (`dsv41.rs:6-13`)。thinking の既定は on (`options.rs`)。
  - **vLLM recipe は low=25 / high=50 / xhigh=75 / max=100、未指定で 50 と書いていて、公式と食い違う。** vLLM で文字列を指定すると公式より浅い effort になる恐れがある。数値で指定すれば回避できる。
  - 効果 (TR p33-34 §5.3.2-5.3.3、Figure 9)。effort を 25 から 100 に上げると、推論 8 種の平均は 67.1% から 76.3%、DeepSWE は 66.0% から 74.2%、出力トークンは約 2.5 倍になる。**60〜80 で最大値の精度の大半が得られ、トークン数は半分以下で済む。** 100 にするとエージェントの軌跡が 1.6〜1.8 倍に伸びるが、精度の伸びはわずかだ。
- **チャットエンコーディング。**
  - Jinja テンプレートは配布されていない。`encoding/encoding.py` が参照実装で、本番向けには deepseek-recipe (Rust 実装と Python binding、`pip install deepseek-recipe`) がある。
  - deepseek-recipe は Chat Completions、Responses、Anthropic Messages の 3 形式を Conversation 形式に変換し、prompt やトークン ID へのエンコード、出力のストリームパースまで担う。推論と HTTP は呼び出し側が実装する (recipe README)。
  - 未対応: logprobs、n>1、JSON Schema 制約 (recipe README)。
  - V4 からの変更 (enc README):
    1. DSML タグの先頭に空白が入る (`<｜DSML｜ calls>`、`<｜DSML｜ invoke>`、`<｜DSML｜ parameter>`)
    2. effort が数値になった
    3. 会話の途中で `<｜System｜>` を使える
  - tool の結果は user メッセージ内の `<tool_result>` として渡す。
  - 画像は `<｜deepseek_image｜>` で表す。
  - 補助タスク用の特殊トークンとして `<｜action｜>`、`<｜title｜>`、`<｜query｜>` などがある。
  - vLLM は `--tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice` を指定する。SGLang は `--reasoning-parser auto --tool-call-parser auto` と `SGLANG_DEFAULT_THINKING=1`、`SGLANG_DSV41_REASONING_EFFORT=high` を使う。

---

## 5. 推論システムの設計と、少数 GPU 構成への示唆

### 5.1 技術報告に書かれていること (TR p18-20 §3.2)

- **カーネル。** FlashMLA の fused RoPE-attention-RoPE-cast、DeepGEMM の Mega-Gate / Mega-mHC / Mega-MoE、TileKernels、DeepSelect の TopK を使う。Reuse モードの層 (大半を占める) は **prefill 15 カーネル、decode 11 カーネル**で動く。
- **Mega-mHC は Single-Pass mHC 用。** 残差の読み書きを 1 回ずつに減らし、activation のメモリトラフィックを半分にする (TR p12-13 §2.4.1)。
- **デプロイ構成。** Encoder–Prefill–Decode (EPD) 分離。永続 KV は SSD に置き、72 時間以上保持する。SWA KV は各機の DRAM の 10% を充てたプールに置き、TTL は数分。
- **TR に書かれていないこと。**
  - FP4 experts の推論形式。
  - 想定する推論ハードの具体名。
  - Engram を NVMe に置いた場合の実測値。
- **重みの形式** (config と convert.py、TR 本文には無い)。
  - experts は FP4 E2M1。K 方向 32 要素ごとに UE8M0 scale を持つ (MXFP4 相当)。
  - 密層は FP8 E4M3 で、32×32 ブロックの UE8M0 scale。
  - `convert.py --expert-dtype fp4` は FP4 のまま保持する。fp8 を指定すると E4M3 に変換する (`convert.py:18-53`)。
  - 参照実装の `fp4_gemm` は FP8 activation × FP4 weight を tilelang で実装している (`kernel.py:477-591`)。

### 5.2 重みの内訳 (`headers.json` を集計した実測)

| 区分 | GB |
|---|---|
| routed experts (40 層 × 384、1 expert あたり約 18.8 MB) | 288.78 |
| Engram 表 (2 層) | 202.76 |
| DSpark (mtp) の routed experts | 7.22 |
| attention (compressor と indexer を含む) | 5.42 |
| shared experts | 1.42 |
| embed / head | 1.32 / 1.32 |
| mtp のその他 | 0.71 |
| ViT と aligner | 0.70 |
| Engram 射影 (wkv) | 0.32 |
| gate、norm、mHC など | 0.32 |
| **合計** | **510.29** |

**課題文の「その他 ≈ 15〜20 GB」は過大な見積もりで、実際は約 11.5 GB (DSpark の experts を除く)。**

### 5.3 2×RTX PRO 6000 (192 GB) と DRAM 128 GB への示唆 (推測)

1. **このままでは収まらない。** Engram を除いた重みが 307.3 GB あり、VRAM 192 GB を超える。ホストを足した 320 GB も、全体の 510 GB を下回る。
   - vLLM の検証済み最小構成は `vram_minimum_gb: 614` (TP4 GB200 など)。
   - 特殊例として、DGX Station GB300 の TP1 構成がある。Engram と routed experts 60 GiB を pinned host に置き、UVA で読む。ただしこれは NVLink-C2C (約 900 GB/s) が前提で、PCIe Gen5 x8 (約 20 GB/s) では成り立たない。
2. **experts をホストに置いた場合の概算** (推測)。

| 方式 | 前提 | 概算 |
|---|---|---|
| ホスト上の experts を PCIe で GPU に転送 | 1 expert 18.8 MB。c16 で検証 4 トークンなら 1 層あたり延べ 384 活性、うち異なる expert は約 6 割。その半分がホスト側 | 1 step あたり数十 GB の転送になり、成立しない |
| CPU 側で計算 | DRAM 帯域 50 GB/s | c1 でも、1 トークンあたりホスト側 expert の読み出し約 2 GB で約 25 tok/s が上限。並列を増やすと悪化する |

3. **現実的な選択肢** (いずれも推測で、品質の検証が必要)。
   - **(a) experts を約 2.1 bit に再量子化して VRAM に収める。**
     - 計算: 192 GB から、非 expert 11.5 GB、DSpark 7.2 GB、KV と作業領域約 25 GB を引くと、experts に使えるのは約 148 GB。これは 288.8 GB × 2.2/4.25 に相当する。
     - Engram は DRAM と NVMe に置く (§1.6)。
   - **(b) expert の枝刈り (REAP 等) で約 50% 削る。**
   - **(c) GPU を足す。** 4 枚 (384 GB) なら FP4 のまま Engram 以外が VRAM に入り、Engram は DRAM 128 GB と NVMe で賄える。
4. **KV、長文脈、並列 16 の条件は楽に満たせる。** 1M × 16 本でも KV は約 15 GB。ネックは重みの容量と、prefill 時の Engram IOPS。
5. **SM120 への対応。** vLLM、SGLang、TR のいずれにも RTX PRO 6000 / SM120 の記述は無い。検証済みの構成は Hopper、B200/B300、GB200/GB300、MI3xx だけ。
   - FlashMLA の sparse 版や DeepGEMM の Mega-MoE は、SM90/SM100 専用である可能性が高い (未確認)。
   - 参照実装の tilelang カーネルは汎用の `T.gemm` を使っており、SM120 でも動く可能性がある (未検証)。
   - vLLM は `torch.compile` に非対応で、初回ロード時の autotune に 75〜90 分かかる (vLLM recipe)。

---

## 6. 矛盾点と未解決事項

| # | 内容 | 根拠 |
|---|---|---|
| 1 | effort の文字列の対応が、公式 (50/75/100、既定 75) と vLLM recipe (25/50/75/100、既定 50) で異なる | `encoding.py:448-453`、`dsv41.rs:6-13`、TR p30 / vLLM recipe |
| 2 | Engram の Zipf ホット率の公式値が無い。自前測定 (§1.5) は英文とコードの約 62 万トークンだけで行ったもので、日本語、中国語、エージェント的トラフィックでは値が変わる | EP p6、TR p13 |
| 3 | vLLM と SGLang の Engram オフロードは、表全体を DRAM に置く前提。NVMe 階層の既存実装は無い (SemiAnalysis の mmap fork は非公開とみられる) | vLLM recipe、LMSYS、SemiAnalysis |
| 4 | V4.1 の DSpark の受理長の実測値が無い (公開されているのは合成値 3.51 だけ) | vLLM recipe、SGLang cookbook |
| 5 | vLLM recipe の実測は 0909 チェックポイントで行われ、1017 では再測定していない | vLLM recipe |
| 6 | TR は Engram のパラメータを 196B としているが、実際の表は 2 × 3.84 億 × 256 = 196.6B 要素。一致している (scale は別勘定) | TR p7、p13 / headers.json |
