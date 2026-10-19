# DeepSeek-V4.1-Flash(-1017) 推論エンジン・量子化・少数 GPU 実績 Web 調査

調査日: 2026-10-18 / 対象: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (2026-10-17 公開)
方針: 読み取りのみ。GPU は使っていない (docker は `--network none` でイメージ内ファイルを読んだだけ)。glm53v には触れていない。
凡例: 【事実】は出典つき。【推測】は筆者の推測。

---

## 0. 結論 (要点)

1. **1017 は V4.1-Flash と同じアーキテクチャで、重みはすべて新しい。Engram 表も変わっている。**
   - HF の `config.json`、`model.safetensors.index.json`、`tokenizer*.json`、`inference/*.py`、`encoding/*` は Flash と 1017 で blob が一致する。
   - safetensors 48 本は全部ハッシュが違う。Engram の shard 47/48 も違う (サイズは同一)。
   - 1017 には `chat_template.jinja` が無い。
   - したがって **V4.1-Flash 由来の量子化物 (NVIDIA NVFP4、diffbot EXL3 など) は 1017 ではない**。Engram 表を Flash から流用してもいけない。
2. **本家 vLLM は V4.1 を正式サポートしている (0.30.0 でアーキ追加、0.31.0+ 推奨)。1017 は recipe の既定チェックポイントになっている。**
   - Engram は `--engram-config '{"cpu_offload":true}'` で pinned host RAM に置き、UVA で読む。
   - DSpark (`method: dspark`, 5 トークン) は唯一の投機方式。
   - SM120 は recipe の検証済みハードウェアに入っていない。コミュニティが TP4 / TP8 で動かしており、TP8 では CUDA graph と DSpark で約 165 tok/s (c1) の報告がある。
3. **LIL vLLM (b12x) は `ds41-flash` プロファイルで 4x RTX PRO 6000 Max-Q の TP4 を qualified にしている。C1 249 tok/s、C8 805 tok/s、32K prefill 18K tok/s。**
   - Engram は RAM (pinned 188.83 GiB) か disk (io_uring) から選べる。
   - **このマシンにある `ghcr.io/local-inference-lab/vllm:release-10152026` にもすでに `ds41-flash.yaml` が入っている。** ただし TP2 のプリセットは無い。
4. **SGLang** はプレビューイメージ (`lmsysorg/sglang:dev-dsv41`) で V4.1 / DSpark / Engram host table に対応している。SM120 で起動したのは TP8/EP8 だけという報告がある。ローカルの `lmsysorg/sglang:dev-cu13` (commit cae69be, 2026-09-29) にも V4.1 と Engram host table の環境変数が入っている。
5. その他のエンジン:
   - **TensorRT-LLM**: V4.1-Flash 対応 PR が 2026-10-11 にマージ済み。ただし SM100/103 前提で、SM120 は未検証。
   - **ik_llama.cpp**: `deepseek41` をマージ済み。vision と DSpark にも対応した。ただし SM120 で MXFP4 専門家を GPU に載せると NaN になるバグが未解決。
   - **llama.cpp 本家**: PR 未マージ。
   - **ktransformers**: 未対応 (issue のみ)。
6. **このマシン (2x96GB, RAM 128GB, NVMe 1 本) ではネイティブ重み (510GB) をそのまま TP2 で動かすことはできない。**
   - Engram 以外の部分だけで約 286 GiB あり、2 枚の VRAM (192GB) を超える。
   - Engram 189 GiB も RAM 128GB に入らない。
   - 現実的なのは「**専門家を EXL3 2〜3bpw にし、Engram を NVMe から読む**」構成。実例は次の 2 つ (どちらも V4.1-Flash 版。1017 版ではない)。
     - nktlabs: 2x96GB、host 125GB、EXL3 2.0bpw、Engram を NVMe から prestage。c1 80 tok/s、8 並列で合計 450 tok/s。
     - diffbot: EXL3 3.0bpw、host RAM 約 230GiB が必要。
7. **Engram は外せない。**
   - 1017 EXL3 のカードでは、Engram lookup を切ると PPL が 1.87→4.16 (code)、6.43→9.06 (prose) に悪化した。
   - 圧縮版は 3 系統ある。MXFP4 / NVFP4 で 97 GiB (cos 0.9934)、int4 で 111GB、MXINT-4 で 97.6 GiB。

---

## 1. モデルの系譜と公開日

| モデル | 日付 | 根拠 |
|---|---|---|
| DeepSeek-V4 (V4-Pro / V4-Flash) API | 2026-04-24 | https://api-docs.deepseek.com/updates/ |
| V4 技術報告 | arXiv 2606.19348 | https://arxiv.org/pdf/2606.19348 |
| V4-Flash 0731 更新 (再 post-train のみ。アーキとサイズは同じ) | 2026-07-31 | https://api-docs.deepseek.com/updates/ |
| **V4.1-Flash** (API `deepseek-flash`) | 2026-09-10 (HF createdAt 2026-09-10T02:17Z。vLLM recipe は「0909 launch」と表記) | 同上。HF API `https://huggingface.co/api/models/deepseek-ai/DeepSeek-V4.1-Flash` |
| V4.1-Flash 技術報告 | arXiv 2609.19969 v1 (2026-09-17) | https://arxiv.org/abs/2609.19969 |
| **V4.1-Flash-1017** | 2026-10-17 (HF createdAt 2026-10-17T02:40Z、sha `1bae0e10`) | https://api-docs.deepseek.com/updates/ 、https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 |

### 1.1 1017 と V4.1-Flash の差分 (実測)

HF API (`?blobs=true`) で両リポの全ファイルのハッシュを比べた。コマンドは `curl https://huggingface.co/api/models/deepseek-ai/<repo>?blobs=true` と Python による比較。

- **同一 (33 ファイル)**:
  - `config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`
  - `inference/{model,kernel,engram,convert,generate,vision,image_processor}.py`
  - `encoding/*`
  - ローカルの `ref/config.json` も 1017 HF の config と `diff` で一致した。
- **異なる (50 ファイル)**:
  - `model-00001..00048-of-00048.safetensors` の全 48 本。サイズはすべて同一。
  - Engram の shard 47 (101,535,150,936 B) と 48 (101,537,926,640 B) もハッシュが違う。
  - `README.md` と `.gitattributes`。
- **Flash にだけあるファイル**: `chat_template.jinja`, `DeepSeek_V41_Tech_Report.pdf`, `evaluation/*`, `assets/*`。
- safetensors の dtype 内訳は両方とも同じ: BF16 1.98B / F32 42M / F8_E4M3 204.0B / I8 557.2B、計 763.2B params。usedStorage は 510.3GB。
- 第三者による同じ見解:
  - vLLM recipe: 「1017 は same architecture, config.json, tokenizer and prompt encoding — only the weights change」。recipe の既定チェックポイントは 1017 (https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash 、curl で取得)。
  - LibertAIDAI 1017-NVFP4 のカード: 「upstream config.json is identical, and all the weights are new」。
  - LibertAIDAI 1017-EXL3 のカード: 「**The tables changed in the 1017 release. Do not reuse the Engram shards of the older V4.1-Flash**」。1017 の Engram shard の sha256 も載っている (47: `76a05106…2af5`, 48: `9dc51433…3afd`)。
- 1017 の README にある記述: 「minor update to DeepSeek-V4.1-Flash」「This release does not include a Jinja-format chat template」。プロンプトは `encoding/encoding.py` か deepseek-recipe で作る。
- ベンチ (1017 vs Flash、モデルカード):

  | ベンチ | 1017 | Flash |
  |---|---|---|
  | HLE w/o tools | 42.6 | 36.8 |
  | Terminal-Bench 4.0 | 42.3 | 31.2 |
  | DeepSWE v1.1 | 77.0 | 74.2 |

  ほか全 11 項目で 1017 が上回る。出力長と実行時間も短くなったと記載がある (API changelog)。

### 1.2 アーキテクチャ上の Engram (技術報告とカードから)

- **パラメータと配置**
  - 552B backbone + 196B Engram。prefill は 8B active、decode は 16B active (arXiv 2609.19969 本文 §2)。
  - Engram は 2 モジュールで、層 1 と 14 に置かれる。
- **表の構造**
  - N-gram 次数は {2,3,4}、hash head は 8 本、1 次数あたりの埋め込み次元は 2048 (= 8 heads × 256)。
  - head ごとの表は約 16M エントリで、サイズは互いに異なる素数。表と k/v projection は FP8 (報告 §2.4.2)。
- **推論時の扱い**: 報告 §2.4.2 に「During inference, deterministic addressing enables embeddings to be prefetched from host memory via background RDMA transfers」とある。公式はホストメモリからの prefetch を前提にしている。
- **1 トークンあたりの読み出し**
  - 3 次数 × 8 head × 2 層 = 48 行/トークン。
  - FP8 では 1 行 264 B (256 + scale 8)、1 トークン 12.4 KB、表全体で 188.83 GiB。
  - MXFP4 では 1 行 136 B、表全体で 97.28 GiB。
  - 出典: LibertAIDAI 1017-NVFP4 のカード、llama.cpp PR #28696。
- vLLM recipe のメモリ内訳 (on-disk 476 GiB):

  | 部分 | サイズ |
  |---|---|
  | 専門家 (routed + DSpark, MXFP4) | 259.5 GiB |
  | Engram (FP8) | 183.1 GiB (scale は別に計上) |
  | 注意など FP8 | 6.9 GiB |
  | Embed / Head / norm | 3.9 GiB |
  | UE8M0 scale | 21.9 GiB |

---

## 2. 推論エンジン別の対応状況

### 2.1 vLLM 本家 (vllm-project/vllm)

- **モデル追加**
  - PR #56214 "[Model] Support DeepSeek-V4.1-Flash" (2026-09-11 マージ)。
  - recipe によるとアーキは 0.30.0 で入った (#56228)。0.31.0 で KV-offload の修正 (#57145) と ROCm Engram の修正 (#57491) が入った。
  - 「launch-day の `deepseekv41-flash-0909` image は不要」とも書かれている (recipe)。
  - モデルコードは `vllm/models/deepseek_v4_1/` にある。parser と tokenizer は `deepseek_v41`。
- **トラッキング issue**: #56400 (2026-09-18 close)。Engram 関連 PR の一覧:
  - #56219: SP 埋め込みの all-to-all
  - #56220: lookup と decoder 計算の overlap
  - #56224
  - #56357: offload lookup の prefetch と mmap 共有
  - https://github.com/vllm-project/vllm/issues/56400
- **Engram 設定** (`EngramConfig`。ローカルの LIL イメージ内の `vllm/config/engram.py` で確認。本家とほぼ同じと【推測】)
  - `cpu_offload`: 既定は `VLLM_PLE_CPU_OFFLOAD`=有効。pinned CPU に置き UVA で lookup する。
  - `embedding_across_dp`
  - `dp_shared_memory`: DP replica 間で /dev/shm を共有する。
  - LIL 拡張: `table_memory: device|ram|disk`、`disk_resident_scales`、`projection_tp`。
- **関連 PR とブログ** (https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)
  - #56512: CPU-offload lookup の非同期 prefetch
  - #57651: 統一 TP/DP sharding、co-located DP 間での表共有
  - #56926: THP で prefill lookup が最大 10 倍速くなる
  - #61530 (open, 2026-10-14): MXFP4 圧縮 Engram 表。TP4 で 1 rank あたり 23.60→12.16 GiB、GSM8K は 0.966 vs 0.967。https://github.com/vllm-project/vllm/pull/61530
- **recipe** (https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash 。WebFetch が拒否されたため curl で取得)
  - Blackwell (B200/B300/GB200/GB300) の TP2 構成: `FLASHINFER_MLA_SPARSE_DSV41`、`--engram-config '{"cpu_offload":true}'`、`--kv-cache-dtype fp8`、`indexer_kv_dtype mxfp4`。
  - 「Engram tables cost 47.2 GiB of device memory per rank at TP4 and 94.4 GiB at TP2」(MI355X 節)。
  - MI300X は TP2 で offload 後も 1 rank あたり約 145 GiB になり、「TP2 does not fit」。
  - DGX Station (GB300 1 枚) は TP1 で、Engram 189 GiB と専門家 60 GiB を pinned host に置き UVA で読む。1 stream 90.5 tok/s、16 並列で 416〜423 tok/s。
  - DSpark: 「the only speculative method」。5 トークン block、NVIDIA では adaptive verification も有効。drafter の専門家で約 14B 増える。
  - **SM120 / RTX PRO 6000 は検証済み一覧に無い。**
- **SM120 の状況** (コミュニティ報告)
  - #56700 (open): 8x RTX PRO 6000、TP8。
    - FlashInfer sparse-MLA の SM120 実装に欠けがある。MXFP4 専門家は Marlin にフォールバックする。
    - DeepGEMM の autotune がハングする。
    - FlashInfer CUTLASS MoE は DeepGEMM 比で c1 −11%、c8 +8%、c16 +12% (c16 で 833 tok/s)。
    - https://github.com/vllm-project/vllm/issues/56700
  - #56892 (open): 当初は day-0 image で enforce-eager 必須、2〜4 tok/s だった。
    - 2026-10-11 の更新: `vllm/vllm-openai:dev253-fi071rc5-dsv41` (v0.31.1rc1.dev253) で TP8 の CUDA graph が動くようになった。c1 decode 165 tok/s (2K)、1M 文脈で約 120 tok/s。
    - 2026-10-17: capture 512 のとき 20MiB 足りず OOM。gpu-mem 0.96 と expandable_segments で回避した。
    - https://github.com/vllm-project/vllm/issues/56892
  - #56771 (open): DSpark と SM120 sparse-MLA prefill の組み合わせで、長い prompt で illegal memory access。
  - SM120 向け PR (open): #59846 (FlashInfer DSv4.1 KV record。`nvfp4_ds_mla` で KV +31%)、#60762 (64-token kernel block)、#57292 (SM120 64-token page)。
- **このマシンとの関係【推測】**
  - 本家 vLLM で SM120 の TP2 を試した報告は見つからなかった。
  - ネイティブ重みは TP2 に収まらない (後述 §5)。本家だけで動かすなら `--cpu-offload-gb` で専門家を host に出すことになる。DGX Station と同じ仕組みだが、host RAM 128GB では足りない。

### 2.2 local-inference-lab (LIL) vLLM + b12x

- **runbook**: https://github.com/local-inference-lab/rtx6kpro/blob/master/models/deepseek-v4.1-flash.md
  - image: `ghcr.io/local-inference-lab/vllm:karmic-kraken-beta-20260922-3221ccacf71002ea`
  - 環境変数: `PROFILE=ds41-flash`、`HARDWARE_PROFILE=rtx-pro-6000-pcie`
  - **TP4/DCP1 だけを文書化**しており、「Keep TP4/DCP1 and adaptive DSpark K7」と書いている。TP2 の記述は無い。
  - 性能 (4x Max-Q, RAM Engram, DSpark K7):

    | 指標 | 値 |
    |---|---|
    | C1 | 249.1 tok/s |
    | C8 | 805.5 tok/s |
    | 32K prefill | 17,982 tok/s |
    | KV | 4.73M logical tokens |

  - Engram は `--engram-table-memory ram|disk`。
    - disk: io_uring で必要な行だけ SSD から読む。`seccomp=unconfined` が必要。性能は未測定。
    - ram: 188.83 GiB を pinned にする。足りなくても disk にはフォールバックしない。
- **issue と PR**
  - #745: DS4.1 community image R38 (`localinferencelab/vllm:jovian-judgement-community-20260914-r38`) を TP4/DCP1 RAM-Engram で qualified。
  - #736: NVMe Engram の最適化。4x Max-Q で C1 229.6、C32 1,220 tok/s、32K prefill 13.7K (NVMe Engram、1M ctx)。https://github.com/local-inference-lab/vllm/pull/736
  - #764 (open): DCP4。KV 18.26M tokens、C1 154 tok/s。
  - #949 / #982: lossless X4T / MXFP4-CSF の reader。
    - DS4.1 の geometry は TP1/2/4/8 (+TP3)。**qualified なのは TP4 だけ**。
    - TP3 (GB10×3) では 1 rank あたり 90.0 GiB の重み。https://github.com/local-inference-lab/vllm/pull/982
  - #888: SM103 (GB300 1 枚) 向けの「expert residency」。HBM と Grace の 2 層に専門家を置く。SM120 / x86 には無い。
  - b12x の Engram 関連: #360 (bounded NVMe prefetch)、#371 (disk prefetch を削除)、#374 (SM120 DCP)。
- **ローカルイメージの確認 (コマンド出力)**
  - `sudo docker run --rm --network none --entrypoint /bin/sh ghcr.io/local-inference-lab/vllm:release-10152026 -c 'cat /opt/lil/runtime/profiles/ds41-flash.yaml'` を実行した。
  - `ds41-flash` (status: implemented) の既定値:
    - `checkpoint: csf` (`local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF` rev `9c60bcbf`)、`tensor-parallel-size: 4`、`mode: dspark` (draft_tokens 7)
    - `engram-table-memory: ram`、`attention/moe/linear-backend: B12X/b12x/b12x`、`max-num-seqs 32`
  - `presets.yaml` に ds41 の TP2 プリセットは無い (glm53-tp2 などのみ)。
  - `vllm/models/deepseek_v4_1/` に EXL3 の経路は見当たらない (grep で exl3 / offload がヒットしない)。
- **LIL の CSF チェックポイント**: https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF
  - 495.6GB。専門家 scale を可逆圧縮しており、16.99→2.31 GB。
  - V4.1-Flash (rev dba1be0a) 版で、**1017 版はまだ無い** (2026-10-18 時点の HF 検索)。
  - 独自ライセンス (LIL License 1.0) で、再配布禁止。

### 2.3 SGLang

- **cookbook**: https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx
  - 「not shipped in an SGLang release yet」。preview image は `lmsysorg/sglang:dev-dsv41`。
  - DSpark: `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`。EAGLE / MTP は無い。
  - Engram: `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`、`SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared|per_rank`。
- **公式ブログ** (4x GB300): BS=1 で DSpark ありだと 873 tok/s、なしで 203 tok/s。https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization
- **SM120 の報告**: issue #40877 (8x RTX PRO 6000 Max-Q)。https://github.com/sgl-project/sglang/issues/40877
  - TP8/EP8 だけが起動した。「V4.1 vision currently supports TP/EP without DP, CP, PP or MoE A2A」。
  - prefill 6.85K tok/s (128K)。decode は prose 94 / structured 134 tok/s。DSpark を切ると 36.5 tok/s。
- **進行中の PR**
  - #43838 (open): TP2 の small-batch decode fusion (GB300 で検証)。
  - #43405 (open): Engram 表 backend (peer-visible device shards / host tables / VMM)。
  - #42787: hash-column 境界で shard。
  - ロードマップ: #42170。
- **ローカルイメージ**: `lmsysorg/sglang:dev-cu13` (commit cae69be, 2026-09-29) に `deepseek_v41_vit.py`、`configs/deepseek_v41.py`、`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` がある (grep で確認)。`lmsysorg/sglang:latest` (0.5.20) は V4 だけ。
- **4x RTX PRO 6000 + 128GB DDR5 + NVMe Engram (SGLang 改造)**: https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 (sybil-solutions)
  - **1017 (rev 1bae0e10) を使っている**。
  - Engram は NVMe から読み、DDR5 に 64GiB の行キャッシュを置く。理由は「native tables occupy ~189 GiB, so they cannot fit in this 128 GB host」。
  - C1 は 196〜232 tok/s、C8 合計は 600〜748 tok/s。prefill は約 7.4K tok/s (32K)。
  - 500K 入力を 8 本、4.06M トークン同時保持できた。DSpark の受理率は 74〜88%。
  - B12x io_uring reader はまだ有効化していない。
  - **このマシンと RAM 構成が同じなので参考になる (GPU は 4 枚)**。

### 2.4 TensorRT-LLM

- PR #19727 "[TRTLLM-16621][feat] Support DeepSeek-V4.1-Flash" は 2026-10-11 にマージされた。https://github.com/NVIDIA/TensorRT-LLM/pull/19727
  - 内容: CSA2、Engram、mHC、DSpark。
  - 検証: GB300×4、NVFP4、GSM8K 96.36%、DSpark 受理長 4.0。
- issue #20511 / PR #20512 (open, 2026-10-17): CSA2 の MXFP8 projection は SM100/SM103 限定で、SM90 では起動に失敗する。BF16 フォールバックを入れる PR。PR 本文に「SM120/121 ... I have not run those」。
- PR #20078 / #20071: DSpark confidence scheduling。
- 【推測】SM120 での実用はまだ先。

### 2.5 llama.cpp / ik_llama.cpp / その他

- **llama.cpp 本家**: PR #28696 "model : add DeepSeek V4.1" (open, 2026-10-10 更新)。https://github.com/ggml-org/llama.cpp/pull/28696
  - Engram は `-ot "engram_embd.weight=CPU"` で CPU に置く。GB10 の Q2_K で 2.8 tok/s。
  - Q2_K は 246.3 GiB、Q3_K_M は 323.4 GiB。
- **ik_llama.cpp**
  - `deepseek41` は PR #2455 で 2026-09-21 にマージされた。Engram は file mapping と行 prefetch で扱う。
  - vision は #2545、DSpark は #2522 / #2546。
  - **issue #2519 (open)**: SM120 (2x RTX PRO 6000) に MXFP4 専門家を載せると NaN になる、または illegal memory access。
  - #2683 (open): 200K 超で文脈を失う ("amnesia")。
  - https://github.com/ikawrakow/ik_llama.cpp/issues/2519
- **DwarfStar (antirez/ds4)**: Metal 専用。Engram を SSD streaming する。https://huggingface.co/antirez/deepseek-v4.1-flash-gguf
- **ktransformers**: issue #2213 "Support for Deepseek V4.1 Flash" が open のまま (2026-09-21)。対応 PR は無い。https://github.com/kvcache-ai/ktransformers/issues/2213
- **deepseek-ai/deepseek-recipe**: 推論エンジンではない。リクエスト変換とプロンプトエンコーディング (V4 / V4.1) を行う Rust と Python のライブラリ。https://github.com/deepseek-ai/deepseek-recipe

---

## 3. コミュニティの量子化・軽量版 (HF)

HF API の検索 (`/api/models?search=DeepSeek-V4.1-Flash` と `search=V4.1-Flash-1017`) と各カードから。サイズは HF の usedStorage。

### 3.1 1017 ベース (2026-10-17 公開。すべて初日のもの)

| repo | 形式 | サイズ | 備考 |
|---|---|---|---|
| [LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4) | 専門家は MXFP4→NVFP4 の**ビット厳密な変換**。Engram は MXFP4 | 399.51 GiB (Engram 97.28 GiB) | activation scale は NVIDIA の **V4.1-Flash** 校正値を流用。NVIDIA 版と同じ layout で、vLLM #61530 が必要 |
| [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | EXL3 2.9bpw (専門家 3bpw、層 18〜22 は 2bpw) | 196.3 GiB。**Engram は含まない** (1017 本体の shard 47/48 を使う) | PPL: code 1.8749→1.9094、prose 6.4277→6.4520。**Engram を切ると 4.16 / 9.06**。MiaAI-Lab の kit 向け |
| [mrfakename/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) | Q8_0 473 / Q4_K_M 414 / Q3_K_M 323 / Q2_K 246 GiB | | llama.cpp PR #28696 のブランチが必要。text のみ |
| [Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF) | 専門家は 8 次元 VQ | 本体 113.6 GB と 1017 Engram 2×101.5GB (無改変) | ds4-dfm-rs (GB10) 向け。1017 版の速度は未測定 |
| mlx-community/…-1017-oQ{2,2.5,3,4}e-mtp、ほかに重みの改変版 | MLX 版、改変版 (範囲外) | | 詳細は未確認 |

### 3.2 V4.1-Flash ベース (主なもの)

| repo | 内容 | サイズ | 品質や速度 |
|---|---|---|---|
| [nvidia/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4) | 専門家 NVFP4。10/14 改訂で Engram を MXFP4 化 (相対 L2 誤差約 12%) | 約 400 GiB (Engram を FP8 に戻した版は rev `3431dde3`) | GPQA 91.0→90.8 など同等。GB300×4 で検証 |
| [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) | **2x RTX PRO 6000 Max-Q の TP2 専用**。専門家 EXL3 3.0bpw (204.5GB)、Engram int4 (111GB)、drafter 3bit | DL 332GB。**host RAM 約 230〜250GiB が必要** | KL 0.0545 (NVFP4 は 0.0352)、GSM8K 98%、HumanEval 95.7%。§4 参照 |
| 同 repo の旧 2.0bpw (rev `28b7ab71`) | EXL3 2.0bpw | | KL 0.3613 (top-1 80.4%) と劣化が大きい。GSM8K-200 97.5% |
| [satgeze/…-EXL3-2.0bpw-Ablit-EngramQ4-SM120-Dual-RTX-Pro-6000](https://huggingface.co/satgeze/DeepSeek-v4.1-Flash-EXL3-2.0bpw-Ablit-EngramQ4-SM120-Dual-RTX-Pro-6000) | diffbot 2.0bpw の派生 (重みの改変版。改変の内容は範囲外)。Engram は MXINT-4 (97.6 GiB) で NVMe から読む | 259.8GB | 2x RTX PRO 6000 の TP2 で decode 89〜94 tok/s。「125 GiB boxes cannot pin the Engram tables」 |
| [nktlabs/dsv41-flash-engram-disk-prestage](https://huggingface.co/nktlabs/dsv41-flash-engram-disk-prestage) | 重みなし。**Engram を NVMe から prestage するパッチ** (diffbot 2.0bpw 用) | | 2x96GB、host 125GB (RSS 18GB)。c1 80.2、4 並列 229、**8 並列 449.7 tok/s**。GSM8K-200 97.5% |
| [coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw](https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw) / GRAFT-3bpw | EXL3 3bpw (校正あり) | 215GB 前後 | diffbot の元になった重み |
| Mia-AiLab EXL3 2.9 / 3.0bpw | 2x DGX Spark の TP2 kit | 210.7GB | Spark ×2 で c1 31.6 tok/s |
| [LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E) | **REAP で専門家を 384→256 に削減** (text と画像のバランスで順位づけ)。Engram は無改変 | 385.5 GiB (Engram 込み) | text PPL +14.1%、caption +8.9%。272E は vLLM の expert 数テーブルに無く、ロードできない |
| amd Quark-MXFP4、intel AutoRound (W4A16/MXFP4 Engram) | AMD 向けや INC の形式 | 410〜510GB | 未確認 |
| xwu-intel/DeepSeek-V4.1-Flash-7layers-compact-engram | 7 層だけの検証用 (README は空) | 63.5GB | 実用ではない |

### 3.3 Engram の圧縮や除去 (まとめ)

- **除去版**: 実用的なものは見つからなかった。
  - 1017 EXL3 のカードに、Engram を切ったときの PPL 悪化が記録されている (code 1.87→4.16)。【推測】Engram を外すのは品質上ありえない。
- **圧縮版**:
  - MXFP4 (NVIDIA / LibertAIDAI): 97.28 GiB、cos 0.9934。
  - int4 + fp16 per-32 scale (diffbot): 111GB。
  - MXINT-4 + e4m3 block scale (satgeze): 97.6 GiB。satgeze のツールは MXINT-3 にも対応。
  - NVIDIA の測定では、MXFP4 Engram で精度は同等 (GSM8K 0.966 vs 0.967、vLLM #61530)。

---

## 4. 2〜4 枚の 96GB GPU、または少数 GPU と大容量 RAM での実績

| 構成 | 重みと Engram の置き場所 | 性能 | 出典 |
|---|---|---|---|
| **2x RTX PRO 6000 Max-Q (300W)、TP2、vLLM nightly + EXL3 plugin** | 専門家 3bpw のうち 30% を pinned host、Engram int4 を pinned host (UVA)。**host 約 230GiB** | agentic: 1 本 159 tok/s、4 並列合計 232。2K decode 74〜124 (plan による)。46K prefill 10K tok/s (4 並列で 18K)。1M prefill 123 s | diffbot (§3.2) |
| **2x 96GB SM120、host 125GB、TP2** | EXL3 2.0bpw の専門家を全部 VRAM に置く。Engram FP8 は NVMe、`prepare_inputs` で preadv→pinned→H2D | **c1 80.2、c4 229.2、c8 449.7 tok/s**。prefill 1.7K (1 本)、3.3K (合計)。pinned 版 (host 約 280GB) は 113 tok/s | nktlabs |
| 2x RTX PRO 6000 (Workstation と Max-Q の混在)、TP2 | EXL3 2.0bpw、Engram MXINT-4 を NVMe から読む (page cache warm) | 2K decode 89〜94 tok/s。46K prefill 1.6〜1.8K。NCCL_P2P_DISABLE=1 が必要 (非対称ペアで deadlock) | satgeze |
| 4x RTX PRO 6000 Max-Q、TP4、LIL b12x | ネイティブ (CSF)。Engram は RAM (pinned 188.83GiB) | C1 249、C8 805、32K prefill 18K | LIL rtx6kpro |
| 4x Max-Q、TP4、LIL (NVMe Engram) | ネイティブ | C1 229.6、C32 1,220 | LIL vllm #736 |
| **4x RTX PRO 6000 (275W)、128GB DDR5**、SGLang 改造 | ネイティブ **1017**。Engram は NVMe と DDR5 の 64GiB キャッシュ | C1 約 200〜230、C8 600〜748。32K prefill 7.4K | sybil-solutions / 0xSero |
| 8x RTX PRO 6000 Max-Q、TP8、vLLM dev253 | ネイティブ FP8 / MXFP4 | c1 165 tok/s (2K)、1M 文脈で約 120 | vLLM #56892 |
| 8x RTX PRO 6000 Max-Q、TP8/EP8、SGLang | ネイティブ | decode 94〜134 (DSpark なしで 36.5)、prefill 6.85K | SGLang #40877 |
| 1x RTX 3090 + 503GiB DDR4 + NVMe | EXL3 3bpw の専門家を全部 pinned DDR4 (190.5GiB)、Engram は NVMe の mmap | decode 約 20 tok/s、prefill 約 800 | https://github.com/0xSero/dsv41-flash-offload |
| 4x DGX Spark (GB10)、TP4 | EXL3 3.5bpw、Engram は NVMe / NFS を on-demand で読む | C1 約 42、C12 合計 475 (coding) | https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark |
| 2x GB10、TP2、REAP-256E | Engram は NVMe | (カードに表あり) | https://github.com/Libertai/dsv41-flash-vllm-2x-spark |

Reddit の r/LocalLLaMA には、V4.1-Flash の 2x RTX PRO 6000 での数値を載せたスレッドが検索では見つからなかった (WebSearch extended を 2 回実行)。X の投稿 (4 枚の RTX Pro と SSD Engram で c8 約 1000 tok/s。https://x.com/Oluwaphilemon1/status/2100203085893157170 ) は 402 エラーで本文を確認できなかった。

---

## 5. このマシンへの当てはめ【推測を含む】

ハードウェア: 2x96GB SM120、RAM 128GB、NVMe 1 本 (空き約 1TB)、PCIe Gen5 x8 (約 20GB/s)。

1. **ネイティブ重みでは TP2 に収まらない。**
   - Engram 以外の合計は約 286 GiB。LibertAIDAI のカードの内訳から計算した。専門家 268.95 + MTP 専門家 6.72 + 注意 4.80 + 共有 1.32 + embed/head 2.46 + ViT 0.90 など。
   - TP2 だと 1 GPU あたり約 143 GiB になり、96GB を超える。
   - 本家の recipe も、MI300X (192GB) で TP2 は 1 rank あたり約 145 GiB で「does not fit」としている。
   - NVFP4 に変換しても専門家は 284.77 GiB で、かえって大きくなる。
   - 専門家の約 100GiB 超を host に出す方法も、RAM 128GB では Engram や OS と両立しない。
   - **結論 (推測)**: ネイティブまたは NVFP4 での 2 GPU 構成は非現実的。
2. **Engram 表の置き場所**
   - FP8 で 189 GiB、MXFP4 / MXINT4 でも約 97 GiB あり、128GB RAM に pin できない。satgeze は 125GiB の箱で 81GiB の pin でも wedge したと報告している。
   - したがって **NVMe から読む方式が必須**。prestage (nktlabs) や io_uring (LIL `table_memory: disk`)。
   - /dev/shm は GLM の LMCache が 49GB 使っているので、GLM を止める前提になる。
   - 帯域の見積もり: 1 トークンあたり 48 行 × 264B ≈ 12.7KB なので帯域は小さい。IOPS が律速になる。decode 450 tok/s なら約 2.2 万 IOPS で問題ない。prefill 10K tok/s なら約 48 万 IOPS で NVMe 1 本の限界に近い (重複除去と page cache で緩和できる)。
3. **現実的な候補**
   - (a) **1017 の EXL3**
     - 既存の LibertAIDAI 2.9bpw は 196 GiB で、VRAM に全部は入らない。約 25〜30% の専門家を host に置く必要があり、diffbot 方式の host 専門家版になる。
     - Engram は NVMe に置く。host 専門家 pinned が約 60GiB なら 128GB RAM に入る可能性はあるが、未実証。
     - あるいは 1017 から自分で EXL3 2.0〜2.5bpw を作り、全部を VRAM に載せる (nktlabs 型)。2.0bpw は KL の劣化が大きい点に注意。
   - (b) LIL b12x の `ds41-flash` は TP4 前提で、host 専門家の機能は SM103 Grace だけ。2 GPU には向かない。
   - (c) SGLang は SM120 で TP8/EP8 しか起動しないという報告があり、2 GPU 構成の実績は無い。
4. **期待値 (推測)**
   - nktlabs と diffbot の 2 GPU 実績では、c1 は 70〜120 tok/s、c8 合計は約 450 tok/s。
   - GLM-5.3-Flash の並列 16 合計約 384 tok/s と同程度か、少し上になる。ただし量子化 (EXL3) が前提。
5. **DSpark の効き**: 2 GPU では受理長が約 4.1 (diffbot)。SM120 では adaptive verification が hang するという報告がある (FlashInfer #5015、tonyd2wild)。

---

## 6. 未確認事項 (open questions)

- vLLM #61530 (MXFP4 Engram) がいつマージされるか。本家に Engram disk / prestage 経路を入れる予定があるか (現状は LIL と個人パッチのみ)。
- 1017 の EXL3 で 2.0〜2.5bpw の版が出るか (2026-10-18 時点では 2.9bpw だけ)。diffbot が 1017 版を出すか。
- LIL の MXFP4-CSF / X4T reader の TP2 geometry を、host 専門家なしで 2x96GB に載せる方法があるか (たぶん無い)。
- SM120 TP2 で DSpark adaptive verification が安定するか (#56771、FlashInfer #5015)。
- NVIDIA の calibration を流用した 1017 NVFP4 の品質 (LibertAIDAI は KL などを未測定)。

## 参考 URL (主なもの)

- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 / https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- https://api-docs.deepseek.com/updates/
- https://arxiv.org/abs/2609.19969
- https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash
- https://vllm.ai/blog/2026-10-07-deepseek-v41-flash
- https://github.com/vllm-project/vllm/issues/56400 , /56700 , /56892 , /56771 , /pull/61530 , /pull/59846
- https://github.com/local-inference-lab/rtx6kpro/blob/master/models/deepseek-v4.1-flash.md
- https://github.com/local-inference-lab/vllm/issues/745 , /pull/736 , /pull/949 , /pull/982 , /pull/888 , /issues/808
- https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx
- https://github.com/sgl-project/sglang/issues/40877 , /issues/42170 , /pull/43838 , /pull/43405
- https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization
- https://github.com/NVIDIA/TensorRT-LLM/pull/19727 , /pull/20512
- https://github.com/ggml-org/llama.cpp/pull/28696
- https://github.com/ikawrakow/ik_llama.cpp/pull/2455 , /issues/2519 , /issues/2683
- https://github.com/kvcache-ai/ktransformers/issues/2213
- https://github.com/deepseek-ai/deepseek-recipe , https://github.com/deepseek-ai/Engram
- https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4
- https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4 , https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
- https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000
- https://huggingface.co/nktlabs/dsv41-flash-engram-disk-prestage
- https://huggingface.co/satgeze/DeepSeek-v4.1-Flash-EXL3-2.0bpw-Ablit-EngramQ4-SM120-Dual-RTX-Pro-6000
- https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E
- https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF
- https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF , https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF
- https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 , https://github.com/0xSero/dsv41-flash-offload
- https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark , https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks
