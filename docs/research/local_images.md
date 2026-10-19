# ローカル推論エンジンイメージの DeepSeek-V4 / V4.1 対応調査

調査日: 2026-10-18。GPU は一切使っていない (`docker run --rm --entrypoint bash <image>`、`--network none`、GPU 割り当てなし)。
稼働中の `glm53v` に対しては `docker exec` / `docker inspect` で読み取りだけを行った。
抽出したソースは `<scratchpad (非公開)>/` 以下 (`lil/`, `b12x/`, `sgl_dev/`, `sgl_orm/`, `sgl_latest/`) に置いてある。

## 0. 結論 (要約)

| イメージ | V4 | V4.1 (`DeepseekV41ForCausalLM`) | Engram | DSpark | SM120 経路 | 1017 の config |
|---|---|---|---|---|---|---|
| `ghcr.io/local-inference-lab/vllm:release-10152026` (LIL vLLM) | あり | **あり (SM120 専用 b12x 実装)** | device / ram / **disk (io_uring)** | あり | b12x 専用経路 | **読める (実測)** |
| `ghcr.io/ormandj/sglang-glm53-flash-sm120:v0.5.0` | あり | あり | device / host RAM のみ | あり | flash_mla_sm120 など | 読める (実測) |
| `lmsysorg/sglang:dev-cu13` | あり | あり (ormandj と V4.1 部分はほぼ同一) | device / host RAM のみ | あり | 同上 | 読める (実測) |
| `lmsysorg/sglang:latest` (0.5.20) | あり | **なし** (Engram カーネルなし) | なし | V4 のみ | - | 不可 |

- LIL vLLM には **DeepSeek-V4.1-Flash の実行用プロファイル `ds41-flash` が入っている**。「native Engram RAM or disk tables and DSpark verification」と説明されている。ただし既定は **TP4**。TP2 用のプリセットは無い。
- **このマシンで最大の障害は Engram ではなく routed experts である。** `headers.json` を集計すると、Engram 以外の重みが 307.5GB ある (routed experts だけで 288.8GB)。TP2 では 1 GPU あたり約 154GB になり、96GB に収まらない。どのエンジンも V4.1 の routed experts を CPU や NVMe へ退避する経路を持っていない (後述 §5)。
- Engram 表 (202.8GB) は LIL vLLM の `table_memory=disk` を使えば NVMe から直接引ける。RAM モードは 121GiB の RAM に収まらないので起動前検査で拒否される。device モードは TP2 だと 1 GPU あたり 101GB で入らない。SGLang には device と host RAM のモードしかないため、このマシンでは Engram を置く場所が無い。
- 参照実装の依存 (tilelang==0.1.12, torch>=2.10, sympy など) は 4 イメージすべてに既に入っている。ただし参照実装は既定 MP=8 で重みを全部 GPU に載せる設計なので、2 GPU では重量的に動かない (§3)。
- GLM-5.3 と共通の b12x 部品は mHC、DSA indexer、FP4 MoE、PCIe allreduce である。LIL の GLM-5.3 実装は mHC を `vllm.models.deepseek_v4.nvidia.b12x.B12xMHCResidual` から借りており、カーネルは文字どおり共有されている (§4)。

---

## 1. 各イメージのモデル実装と V4.1 config キー対応

### 1.1 LIL vLLM (`release-10152026`, vllm `0.1.dev21830+g8244d1c44.cu134`)

`vllm/model_executor/models/` には `deepseek_v4*` が無い。V4 系は新しいトップレベルパッケージ **`vllm/models/`** にある。

- `vllm/models/deepseek_v4/` (V4 と V4-Vision)、`vllm/models/deepseek_v41/` (汎用 NVIDIA / AMD)、`vllm/models/deepseek_v4_1/` (**SM120 専用の b12x 実装**)、`vllm/models/deepseek_v32/`、`vllm/models/glm5next/` (GLM-5.3) など。
- registry (`vllm/model_executor/models/registry.py`):
  - 98 行: `"DeepseekV4ForCausalLM": ("vllm.models.deepseek_v4", ...)`
  - 383-385 行: `"DeepseekV41ForCausalLM": ("vllm.models.deepseek_v41", "DeepseekV41ForCausalLM")`
  - 658-661 行: `"DSparkDraftModel"` と `DSparkV41DraftModel` → `vllm.models.deepseek_v41.DSparkDeepseekV4ForCausalLM`
- 振り分け (`vllm/models/deepseek_v41/__init__.py:16`): `current_platform.is_device_capability_family(120)` の場合は `vllm.models.deepseek_v4_1` (b12x 専用) を import する。SM120 では SM120 専用経路が選ばれる。
- SM120 専用経路のファイル: `deepseek_v4_1/{attention.py, b12x_layers.py, ced.py, compressor.py, mxfp4_csf.py, sparse_mla.py, l2_prefetch.py, common/engram.py, nvidia/{model.py, dspark.py, b12x_vision.py, vl_model.py, model_state.py}}`。
- 周辺: `vllm/config/engram.py` (EngramConfig)、`vllm/transformers_utils/configs/deepseek_v41.py`、`vllm/tokenizers/deepseek_v41*.py`、`vllm/parser/deepseek_v41.py`、`vllm/tool_parsers/deepseekv41_engine_tool_parser.py`、`v1/worker/gpu/spec_decode/dspark/*`。

**config キーの参照状況** (SM120 経路 `deepseek_v4_1` と汎用経路 `deepseek_v41` を grep した結果):

| キー | 参照箇所 (抜粋) |
|---|---|
| `compress_ratios` | `deepseek_v4_1/attention.py`, `ced.py` / `deepseek_v41/attention.py`, `sparse_mla.py` |
| `compress_rope_theta` | `deepseek_v41/common/rope.py` |
| `kv_source_layer_ids`, `index_source_layer_ids`, `index_n_heads` | `deepseek_v4_1/attention.py` |
| `candidate_source_layer_id`, `candidate_topk_blocks` | `deepseek_v4_1/attention.py`, `nvidia/model.py` |
| `candidate_block_size` | `deepseek_v41/attention.py` |
| `hc_mult`, `hc_sinkhorn_iters` | `deepseek_v4_1/b12x_layers.py`, `nvidia/model.py`, `common/engram.py` |
| `engram_layer_ids`, `engram_num_embeddings`, `engram_compressed_vocab_size` | `deepseek_v4_1/common/engram.py` |
| `dspark_markov_rank`, `dspark_n_routed_experts`, `dspark_target_layer_ids`, `num_nextn_predict_layers` | `deepseek_v4_1/nvidia/dspark.py` |
| `dspark_block_size`, `dspark_noise_token_id` | `vllm/config/speculative.py:1581-1585`, `338-357` |
| `o_groups`, `o_lora_rank`, `sliding_window` | `deepseek_v4_1/attention.py` |
| `swiglu_limit` | `deepseek_v4_1/mxfp4_csf.py` |
| `expert_dtype`, `scale_fmt` | `deepseek_v41/quant_config.py`, `deepseek_v4_1/nvidia/model.py` |
| `image_token_id` | `deepseek_v41/common/mm_preprocess.py`, `nvidia/vl_model.py` |

**V4.1-1017 の config を実際に読ませた結果**: `ref/config.json` を read-only でマウントし、`vllm.transformers_utils.config.get_config` に渡した。

```
<class 'vllm.transformers_utils.configs.deepseek_v41.DeepseekV41Config'> deepseek_v41 ['DeepseekV41ForCausalLM']
compress_ratios 43 / kv_source_layer_ids [2, 8, 14, 20] / index_source_layer_ids [2, 8, 14, 20, 24, 28, 32, 36]
candidate_source_layer_id 20 / engram_layer_ids [1, 14] / engram_num_embeddings [384006168, 384016682]
dspark_block_size 5 / dspark_markov_rank 256 / dspark_target_layer_ids [37, 38, 39] / hc_mult 4 / o_groups 8
quant {'quant_method': 'fp8', ..., 'weight_block_size': [32, 32], 'scale_fmt': 'ue8m0', 'expert_dtype': 'fp4'}
```

**Engram のハッシュ幾何が 1017 の表サイズと一致するか**も CPU で確認した。

```
b12x.sequence.engram.build_geometry(layer_ids=(1,14), base_table_size=16000000, compressed_vocab_size=99092)
→ geometry num_embeddings (384006168, 384016682) match True
```

`deepseek_v4_1/common/engram.py:58-66` は `(max_ngram, heads, head_dim) == (4, 8, 256)` を要求し、幾何と checkpoint の行数が一致しなければ例外を投げる。1017 はこの検査を通る。

その他のハード制約 (SM120 経路):

- `attention.py:483-495`: `head_dim == 512`、`rope_head_dim == 64`、`compress_ratio ∈ {0,1,2}` でなければ例外になる。DCP / PCP (context parallel) も禁止される。GLM で使っている DCP2 は V4.1 では使えない。
- `nvidia/model.py:82`: B12X 以外は不可 ("DeepSeek V4.1 requires B12X")。`:130`: sqrtsoftplus + norm_topk 必須。`:371`: LoRA 不可。`:376`: 量子化は `deepseek_v41_fp8` か `mxfp4_csf` に限る。
- 元の HF checkpoint (`quant_method: fp8`) は `deepseek_v4_1/quant_config.py` の `override_quantization_method` が `deepseek_v41_fp8` に読み替える。experts は `Mxfp4MoEMethod`、Linear は `B12xFP8LinearMethod` で処理される。1017 は元形式なので CSF 変換なしで読める見込みである (ロード自体は未検証)。
- `mxfp4_csf.py:117`: CSF の experts は EP / DP 非対応で、TP のみ。

### 1.2 SGLang 系

| | ormandj v0.5.0 | dev-cu13 | latest (0.5.20) |
|---|---|---|---|
| sglang ソース | `/opt/glm53/sources/sglang` (`0.0.0.dev1+gc7be3e935.d20260930`) | `/sgl-workspace/sglang` (`gcae69be58`) | `0.5.20` |
| `srt/models/deepseek_v4.py` | あり | あり | あり |
| `srt/models/deepseek_v41_vit.py` | あり | あり | **なし** |
| `srt/configs/deepseek_v41.py` (`DeepseekV41Config`) | あり | あり | **なし** |
| `srt/layers/engram.py`, `kernels/ops/embeddings/engram_{hash,gather,gate}.py` | あり | あり | **なし** |
| `srt/layers/attention/dsv4/dsv41_sparse.py`, `v41_indexer/` | あり | あり | なし |
| `deepseek_v4_dspark.py` (DSpark) | あり (V4.1 用 `is_dsv41` 分岐) | あり | V4 のみ |

- V4.1 は `normalize_deepseek_v41_config` (`srt/configs/deepseek_v41.py:23-40`) で text_config を平坦化し、architecture を `DeepseekV4ForCausalLM` に書き換えて V4 のモデルクラスで動かす。
- ormandj と dev-cu13 の差分は、`engram.py`、`configs/deepseek_v41.py`、`dsv41_sparse.py`、`deepseek_v41_vit.py` が 0 行、`deepseek_v4.py` が 15 行、`deepseek_v4_dspark.py` が 4 行だった。V4.1 部分は実質同じである。
- 1017 の config を `DeepseekV41Config(**json)` に通すと、両イメージとも `deepseek_v41 ['DeepseekV4ForCausalLM'] 43 [1, 14] [384006168, 384016682] [2, 8, 14, 20] 20 5 32` が出力された (実測)。
- SM120 の扱い:
  - `srt/layers/attention/deepseek_v4_backend.py:3444` で SM120 のとき `sglang.kernels.ops.attention.flash_mla_sm120.flash_mla_with_kvcache_sm120` を使う。
  - indexer は SM120 で `fp8_paged_mqa_logits_torch_sm120` という torch 実装になる (`dsv4/indexer.py:187, 834`)。
  - MXFP4 MoE は `cutlass_sm120` を使う (`layers/quantization/mxfp4.py:412`)。
- **Engram の置き場所** (`srt/layers/engram.py:667-700`, `environ.py:1549,1554`): 既定は TP シャードした GPU 常駐である。`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` でホストメモリに置ける (memfd + `cudaHostRegister`、layout `shared` か `per_rank`)。**ディスクに置くモードは無い。**

---

## 2. LIL launcher (`/opt/lil/runtime`) のプリセットと b12x

### 2.1 プロファイルとプリセット

- `presets.yaml` のプリセットは `glm53-tp2`, `glm53-csf-tp8`, `glm53-csf-tp6`, `glm53-tp3`, `qwen38-tp2`, `qwen38-tp1x2`, `mimo26-flash-tp2`, `glm53-dgx-spark-x2`, `qwen38-dgx-spark-x2` で、**DeepSeek 系のプリセットは無い**。
- `profiles/` には `ds4-flash.yaml`, `ds4-vision.yaml`, **`ds41-flash.yaml`** がある。
- `profiles/ds41-flash.yaml` の主な行:
  - 5: `description: DeepSeek-V4.1-Flash with native Engram RAM or disk tables and DSpark verification.`
  - 7: 根拠 URL `https://github.com/local-inference-lab/rtx6kpro/blob/master/models/deepseek-v4.1-flash.md`
  - 14: `tensor-parallel-size: 4`、15: `mode: dspark`、20: `max-num-seqs: 32`、22: `gpu-memory-utilization: 0.95`
  - 27-29: `attention-backend: B12X` / `moe-backend: b12x` / `linear-backend: b12x`
  - 36: `engram-table-memory: ram`、37: `engram-disk-resident-scales: false`、38: `engram-projection-tp: false`
  - 43-45: `tokenizer-mode / reasoning-parser / tool-call-parser: deepseek_v41`
  - 60-71: checkpoint は `csf` (= `local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF` rev `9c60bcbf`) または `original` (= `deepseek-ai/DeepSeek-V4.1-Flash`)。**1017 用の CSF は無い。** 1017 は `CHECKPOINT=original` と `MODEL=<ローカル>` の組み合わせで使うことになる (この場合 load-format は instanttensor)。
  - 74-76: `dspark: draft_tokens: 7`, `method: dspark`, `draft_sample_method: probabilistic`
- `options.yaml:534-544`: `engram-table-memory` の enum は `ram` と `disk` の 2 つ。`launcher.py:2468-2477` は profile 経由では `table_memory ∈ {ram, disk}` かつ `cpu_offload=False` しか許さない。
- `hardware/rtx-pro-6000-pcie.yaml` の `model_environment.ds41-flash`: `VLLM_PCIE_DMA_MIN_BYTES=6MB`, `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=96KB`, `..._FUSED_ADD_RMS_NORM_MAX_SIZE=96KB`, `TWOSHOT=off`。
- `generate.py:78-81`: ds41 では compose に `ulimits.memlock=-1` と **`security_opt: seccomp=unconfined`** を加える。コメントは "The published Engram disk path uses io_uring, blocked by Docker's default seccomp profile"。disk モードで起動するときは docker run に `--security-opt seccomp=unconfined --ulimit memlock=-1` が要る。
- **TP2 での計画解決 (CPU のみの dry resolve)**: `runtime.launcher.resolve("ds41-flash","rtx-pro-6000-pcie", env={TP:2, ENGRAM_TABLE_MEMORY:disk, CHECKPOINT:original, MAX_NUM_SEQS:16, MODEL:/models/ds41})` は例外を出さずに解決した。

  ```
  OK {'model': '/models/ds41', 'tensor-parallel-size': 2,
      'engram-config': {'cpu_offload': False, 'table_memory': 'disk', 'disk_resident_scales': False, 'projection_tp': False},
      'load-format': 'instanttensor'}
  ```

  launcher の段階では TP2 は拒否されない。ただし実際の起動は重みの容量不足で失敗する見込みが高い (§5)。

### 2.2 b12x (flashinfer `0.7.1+lil.cu134.sm120` に同梱: `flashinfer/experimental/b12x/`)

`b12x` パッケージ自体は `flashinfer.experimental.b12x` への alias である (`site-packages/b12x/__init__.py`)。V4.1 の SM120 経路が使う b12x モジュールは次のとおり。

- `b12x.attention.compressed_sparse_mla` (V4.1 用。`cache_format == "deepseek_v41"` の分岐あり)、`b12x.attention.dsa_indexer`、`b12x.attention.mla_compress`、`b12x.attention.varlen`
- `b12x.gemm.{block_fp8_linear, bf16_gemv, wo_projection}`、`b12x.moe.fused_moe`
- `b12x.norm.{mhc, hyperconnection, vision}` (`run_engram_mix_impl` など Engram 混合用の関数もある)
- `b12x.sequence.engram` (ハッシュ / lookup / DiskTable)、`b12x.sequence._shared.disk_table`

形状対応について確認できたこと:

- **head_dim 512 / rope 64 / kv_heads 1**: V4.1 専用経路の前提そのものである (`attention.py:483-488`)。
- **heads / TP**: `compressed_sparse_mla/_tuning.py:216-226` は V4.1 FP8 decode で `v41_heads_per_block ∈ {8,16}` を取り、H16 では「16 ヘッドの完全なグループ」を要求する。64 heads は TP2 で 32 heads/rank、TP4 で 16 heads/rank になり、どちらも条件を満たす。
- **MoE (experts 384, inter 2304, hidden 5120, top-6)**: `moe/fused_moe` に experts 数や inter を固定する検査は見当たらなかった。あるのは `trellis_layout.py:68` の「H divisible by 32」と `_tuning.py:767` の「top-k within 1024 experts」程度である。TP2 なら I_local は 1152 (=9×128)。V4.1 専用経路 (MXFP4 a8 / a16) として実装済みである。ただし LIL が性能を確認しているのは TP4 の構成だけである (プロファイル既定値からの推測)。
- **Engram disk**: `disk_table.py:172-174` で `B12X_DISK_BACKEND ∈ {io_uring, gds}` (既定は io_uring)。`loader/_row_plan.h:113` は `open(path, O_RDONLY|O_CLOEXEC|O_NONBLOCK|O_DIRECT)` で開くので **OS のページキャッシュは使われない**。読み出し単位は `PLE_BLOCK 4096u` (`_row_plan.h:6`)。ブロックはソートし、隣接するものはまとめて読む (`coalesced_reads`)。
  - disk モードは **checkpoint の safetensors を直接読む** (`deepseek_v4_1/common/engram.py:524-561` は `get_file_tensor_source` から path と offset を取り、`DiskTable.add_shard`)。1017 の配置では `layers.1.engram.embed.{weight F8_E4M3 [384006168,256], scale F8_E8M0 [384006168,8]}` が `model-00047`、layer 14 が `model-00048` にあり (`headers.json` で確認)、`(rows,256)` と `(rows,8)` の形状検査 (`engram.py:536-546`) と一致する。
  - `config/engram.py:76-86`: `disk_resident_scales=True` にすると、スケールだけを RAM に常駐させる (約 5.72GiB)。

---

## 3. ライブラリのバージョン (各イメージで `importlib.metadata` を使って実測)

| | LIL vLLM | ormandj sglang | sglang dev-cu13 | sglang latest |
|---|---|---|---|---|
| python | 3.12.3 (`/opt/venv`) | 3.12.3 (`/opt/sglang`) | 3.12.3 | 3.12.3 |
| torch | **2.14.0a0+nv26.8** (CUDA 13.4, arch に sm_120 あり) | 2.13.0+cu130 | 2.13.0+cu130 | 2.13.0+cu130 |
| tilelang | **0.1.12** | **0.1.12** | **0.1.12** | **0.1.12** |
| flashinfer-python | 0.7.1+lil.cu134.sm120 (b12x 同梱) | 0.7.0 | 0.6.18 | 0.6.18 |
| deep_gemm | **無し** | sgl-deep-gemm 0.2.0+gitc518ae0a | 0.2.0 | 0.2.0 |
| triton | 3.8.0+nv26.8 | 3.7.1 | 3.7.1 | 3.7.1 |
| transformers | 5.15.0 | 5.18.0.dev0 | 5.12.1 | 5.12.1 |
| nvidia-cutlass-dsl | 4.7.1 | 4.8.0 | 4.6.2 | 4.6.2 |
| その他 | lmcache 0.5.3.dev453+lil, instanttensor 0.1.9 | sglang-kernel 0.4.7, sgl-deep-ep 0.1.2, tokenspeed-mla 0.1.8 | 同左 | 同左 |
| nvcc | 13.4 | 13.0 | 13.0 | 13.0 |
| flash_mla モジュール | 無し | 無し (sgl_kernel.flash_mla 経由) | 同左 | 同左 |

参照実装の依存 (`ref/requirements.txt`: `torch>=2.10.0, transformers, tokenizers, safetensors>=0.7.0, numpy, sympy, Pillow, tilelang==0.1.12, tqdm`) は **4 イメージすべてで満たされている**。sympy 1.14.0、tqdm 4.70、pillow 12.3.0、numpy 2.1 / 2.3、tokenizers 0.22 / 0.23 が入っている。tilelang は全イメージにあるので pip での追加導入は不要だった。

- tilelang 0.1.12 は SM120 を識別できる (`tilelang/cuda/target.py:139 def target_is_sm120`)。参照の `kernel.py` は `TL_DISABLE_WARP_SPECIALIZED` と `TL_DISABLE_TMA_LOWER` を True にしており (`kernel.py:9-11`)、mma ベースで SM120 にも通る構成に見える。ただし **GPU 上でコンパイルや実行ができるかは未検証** (GPU 使用禁止のため)。
- 参照実装を動かす候補は LIL vLLM (torch 2.14) か sglang dev-cu13 / ormandj (torch 2.13) で、どれでも依存的には動く。とはいえ参照実装は `convert.py --model-parallel MP` で MP 個に分割し、全部を GPU に載せる (`run.sh` の既定は MP=8)。Engram 表も `ParallelEngramEmbedding` で GPU に置く。510GB を 2×96GB で動かすことは**できない**。部分的な層単位の検証 (数層だけ読み込む等) にしか使えない (推測)。

---

## 4. GLM-5.3 との共通点と既存カーネルの流用

GLM-5.3-Flash (`$MODELS/GLM-5.3-Flash-FP4-Local/config.json`) は `Glm5NextForConditionalGeneration` で、hidden 4096、45 層 (KDA と DSA)、64 heads、kv_lora 512 / NoPE 256、experts 288 top-8 inter 2048、`hc_mult 4`、indexer 32×128 top-k 2048。稼働中の `glm53v` は `PRESET=glm53-tp2` で `vllm.models.glm5next` を使っている (`docker inspect glm53v` の Env)。

| 部品 | GLM-5.3 (LIL) | V4.1 (LIL SM120) | 流用性 |
|---|---|---|---|
| mHC (hc_mult 4, sinkhorn) | `glm5next/nvidia/model.py:497-503` で **`vllm.models.deepseek_v4.nvidia.b12x.B12xMHCResidual` を import して使用** | `b12x.norm.mhc` / `b12x.norm.hyperconnection` (`b12x_layers.py:683` "V4.1 mHC requires four streams") | **同じ b12x mHC 系。GLM 側が DSv4 のコードを借りている** |
| DSA indexer | `b12x.attention.dsa_indexer` (`_glm_pool_kernels.py`, `glm_pool.py`)。`deep_gemm.fp8_fp4_*mqa_logits` も参照するが、LIL イメージに deep_gemm は無い | `b12x.attention.dsa_indexer` + V4.1 の 2 段 indexer (candidate blocks 2048×8 → top-k 512) | モジュールは共通、経路は別 |
| Sparse MLA | `b12x_mla_sparse` (`B12xGLM5NextMLASparseBackend`)、head 256 | `b12x.attention.compressed_sparse_mla` (head 512、kv 1 head、圧縮 KV、SWA 128) | 別カーネル。V4.1 用は実装済み |
| FP4 MoE | `b12x.moe.fused_moe` (NVFP4、CSF scale) | `b12x.moe.fused_moe` (MXFP4、UE8M0 /32、CSF 可) | 同じフレームワークで量子化形式だけ違う |
| PCIe allreduce | `VLLM_PCIE_ALLREDUCE_BACKEND=b12x` | 同じ (hardware yaml の ds41 用設定あり) | そのまま使える |
| 投機デコード | MTP3 | DSpark (block 5、markov、confidence head、`draft_tokens 7`、adaptive verification) | 別物。V4.1 は MTP 方式を明示的に拒否する (`config/speculative.py:1488-1500`) |
| 文脈並列 | TP2 + DCP2 | **DCP / PCP 禁止** (`attention.py:489-495`) | GLM の DCP2 構成は使えない |
| ホスト側の埋め込み | `VLLM_GLM53_EMBED_HOST=1` | Engram ram / disk (専用実装) | 別物 |

まとめると、**LIL vLLM のイメージには V4.1 用の専用カーネル群が既に揃っている**。GLM-5.3 から新たに流用する必要は無い。mHC と MoE と allreduce は GLM-5.3 と同じ b12x 系なので、GLM-5.3 で確認した PCIe 周りの設定や JIT キャッシュの運用はそのまま応用できる (推測)。

---

## 5. このマシン (2×96GB, RAM 121GiB, NVMe 1 本) での実現性

### 5.1 重み容量 (`ref/headers.json` の `data_offsets` を集計して実測)

| 区分 | GB |
|---|---:|
| routed experts (本体 40 層) | 288.78 |
| Engram 表 (embed.weight と scale、2 層) | 202.76 |
| DSpark (mtp) routed experts | 7.22 |
| 密層その他 (注意・共有専門家・gate など) | 7.03 |
| embed / head | 2.65 |
| vision + aligner | 0.82 |
| mtp その他 | 0.71 |
| Engram 投影 (wkv, q / k) | 0.32 |
| **合計** | **510.29** |

- **Engram を除いた重みは 307.5GB**。TP2 では 1 GPU あたり 153.7GB になり、96GB を大きく超える。DSpark と vision を外しても約 299GB (150GB/GPU) である。KV やワークスペースを除いて重みに使えるのを 1 GPU 約 85GB とすると (推測)、合計 170GB 前後に対して **約 130〜140GB が溢れる**。
- LIL の V4.1 実装には routed experts を CPU や NVMe に退避する経路が無い (`deepseek_v4_1` と `deepseek_v41/nvidia` を grep しても offload は Engram 関連だけ)。vLLM 汎用の `OffloadConfig` (`config/offload.py`: `cpu_offload_gb`, prefetch) はあるが、b12x の prepared weight と組み合わせられるかは不明 (未検証)。仮に組み合わせられても、溢れる約 140GB をホスト RAM (121GiB、現在 available 41GiB) に置けない。PCIe 20GB/s で毎 step 転送すると 1 step に数秒かかる (推測計算)。
- LIL の既定は TP4 (4×96 = 384GB → 約 77GB/GPU) で、作者のリグは 4 GPU 以上と考えられる (推測)。

### 5.2 Engram 表の置き場所

| モード | 必要量 | このマシン |
|---|---|---|
| device (TP2) | 1 GPU あたり 101.4GB | 不可 |
| ram (LIL) | 202.75GB と予備 4GiB (`config/engram.py:48-55, 154`) | 不可 (MemTotal 121GiB)。起動前検査で拒否される |
| SGLang host table (shared) | 202.75GB | 不可 |
| **disk (LIL, io_uring O_DIRECT)** | RAM はほぼ不要 (`disk_resident_scales=true` で約 5.7GiB) | **可** (NVMe 空き 922G。safetensors を直接読む) |

disk モードの I/O 量 (推測計算): 1 トークンあたり 2 層 × 24 行 = 48 行を引き、1 行ごとに 4KiB の O_DIRECT 読み出しになる (scale を RAM 常駐にした場合)。

- decode を並列 16、DSpark で検証 8 トークンとすると、1 step は 128 トークン × 48 = 約 6,100 回のランダム 4KiB 読み出しになる。PCIe4 NVMe の 4K ランダム読み出しを数十万〜100 万 IOPS とすると、1 step あたり 6〜20ms。
- prefill 4096 トークンのチャンクでは約 19.7 万回で、0.2〜0.6 秒になる。
- ページキャッシュを使わないため、ホットな n-gram も毎回 NVMe まで読みに行く (重複行は除去される)。

### 5.3 結論

既存のイメージをそのまま 2 GPU で使い、V4.1-1017 の全重みを載せてサーブすることはできない。Engram は LIL の disk モードで解決できるが、routed experts 307.5GB が VRAM 192GB に入らないことが根本の障害である。考えられる方向は次の 3 つ (いずれも推測)。

1. GPU を 4 枚以上にして LIL の `ds41-flash` を TP4 + Engram disk で動かす (RAM 128GB なので ram モードは不可)。
2. experts を NVMe や RAM から流し込む独自実装を作る (速度は大きく落ちる)。
3. experts の剪定や再量子化 (品質が劣化する。公式には無い)。

---

## 6. 未確認事項

- LIL V4.1 経路の TP2 での実ロードと、b12x の JIT が TP2 形状 (32 heads/rank, I_local 1152) を通るか (GPU 不要の範囲では確認できない)。
- 1017 の重み名が `original` (V4.1-Flash) と完全に同じか。config は同一形式で、Engram の幾何は一致した。
- tokenizer ファイルはダウンロード未完了で、Engram の `compressed_vocab_size` 照合 (`engram.py:84-86`, tokenizer から 99092 を再計算) は未実施。
- tilelang 0.1.12 で参照 kernel.py が SM120 上でコンパイルできるか。
- `rtx6kpro` リポジトリの V4.1 実測値 (profile の根拠 URL。ネットでの確認は別担当)。
