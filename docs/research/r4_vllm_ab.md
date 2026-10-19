# r4: DeepSeek-V4.1-Flash-1017 EXL3 2.0bpw (pack v0) を本家 vLLM + vllm-exl3 で動かす準備 (ds41rt との A/B 用)

作成日: 2026-10-18 / GPU は使っていない (評価ジョブが両 GPU を使用中)。docker は GPU 無し (`--network none`、`--gpus` 無し) でだけ動かした。
凡例: 【実測】はこの作業で手元で確かめた値、【資料】はリポジトリや HF の記載、【計算】は式から出した値、【推測】は根拠が間接的な値。

---

## 0. 結論

1. **使うレシピは diffbot の 2.0bpw 版 (HF `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000` の rev `28b7ab71`)。** 旧 repo 名 `…-EXL3-2.0bpw-2x-RTX-PRO-6000` は今は 3bpw repo へ 307 で転送される。2.0bpw 版のレシピは repo の履歴にしか無い【実測: HF API】。
   - pack v0 の `config.json` は diffbot 2.0bpw 版の `config.json` と完全に一致した【実測: sort_keys で diff して差分 0】。テンソル名も `layers.N.ffn.experts.E.w{1,2,3}.{trellis,suh,svh,mcg}` の形で、diffbot 形式と同じ。
   - 1017 はアーキ・config・tokenizer が V4.1-Flash と同一なので (web_engines §1.1)、0909 の day-0 image のモデルコードでそのまま読めるはず【推測。ロードは GPU で確認する】。
2. **nktlabs の prestage は diffbot 2.0bpw の image (`deepseekv41-flash-0909` + sfxnz `460ac89c` の `apply_engram_disk.py`) 専用に作られており、そのまま適用できる。** nktlabs の `engram_disk.py` は sfxnz `460ac89c` 版に関数 2 個 (`engram_prestage_enabled` / `engram_prestage_check_enabled`) を足しただけだった【実測: diff】。残り 3 ファイルの image との差分は §3.3。
   - 4 ファイルを image に焼き込んだ派生イメージ **`dsv41-flash-exl3-sm120-prestage`** を作った (§2)。
3. **当機 (RAM 121GiB 実効) で現実的なのは「Engram disk + prestage + CUDA graph」だけである。** 代替 2 案は数値で劣る (§4)。
   - diffbot の pinned 経路: Engram 190GiB を pin。当機では不可能。
   - Engram disk + eager (sfxnz `DSV41_ENGRAM_DISK=1`): 起動はできる。ただし c1 26〜31 tok/s【資料】。
   - Engram を MXFP4 化して pin (~97GiB): pin の上限 (~90GB) を超える。超えなくても OS・プロセス・ページキャッシュに残るのは 20GiB 前後になる。加えて、MXFP4 Engram の読み手 (vLLM PR #61530、open) は 0909 image に無く、移植が要る。
4. **起動スクリプト**: `build/serve_vllm_exl3.sh` (TP2、DSpark、MAX_NUM_SEQS 16、口 8001、`deepseek-v4.1-flash`、container の RAM 上限 112g で swap 無し、Engram disk + prestage)。`DRY_RUN=1` で GPU 無しの検査ができる (§5)。
5. **予想性能 (thinking off、2K 文脈)**: c1 75〜90 tok/s、c4 210〜240、c8 420〜470、c16 500〜650 (aggregate)。prefill は 1.5〜2.2k tok/s【推測、根拠 §6】。ds41rt (c1 code 180〜210 の見込み、prefill 3.5〜5k) より decode・prefill とも遅い見込み。vLLM 側の強みは API 機能 (§8)。

---

## 1. 取得したもの

| 置き場所 | 中身 | 出所 |
|---|---|---|
| `ext/diffbot_exl3_2bpw_recipe/recipe/` | **2.0bpw 版レシピ一式** (docker/, kernels/ (xmoe 源), patches/{vllm,flashinfer}/, serve/, bench/, quantize/, results/)。72 ファイル。重みは取っていない | HF `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000` @ `28b7ab71` |
| `ext/diffbot_exl3_2bpw_recipe/ref_3bpw_main_6062be5f/recipe/` | 参考: 現行 3bpw 版レシピ (2026-10-15 に vLLM main `nightly-5555a2d8` へ rebase。host experts、int4 Engram、vllm-exl3 AGPL パッチ、PR #59585/#61039/#61142/#60792 の diff 等)。97 ファイル | 同 repo @ `6062be5f` |
| `ext/diffbot_exl3_2bpw_recipe/recipe/third_party/sfxnz-recipe/` | sfxnz recipe の checkout (`460ac89c`、diffbot の `docker/build.sh` が要求する commit) | ローカルの `ext/sfxnz_recipe` から clone し、origin を GitHub に向けて fetch |
| `ext/nktlabs_prestage/` | `patches/prestage/{engram.py,engram_disk.py,model.py,model_state.py}`、`serve/serve-prestage.sh`、`PATCH-SUMMARY.md`、`README.md`、`NOTICE.md` | HF `nktlabs/dsv41-flash-engram-disk-prestage` @ main |

- 既存の部分コピー (README、recipe/README.md、quantize/、results/RESULTS.md) は rev `28b7ab71` の取得物と一致した【実測: diff 差分 0】。
- diffbot の `recipe/docker/build.sh` には 1 か所だけ手を入れた。exllamav3 のビルド並列数 `MAX_JOBS=48` を環境変数で変えられるようにした (元は `build.sh.orig`)。理由: 7950X (32 thread) で nvcc を 48 並列に走らせると、評価ジョブと合わせて RAM 128GB を超えるおそれがある。

### 1.1 nktlabs prestage の要点 (README / PATCH-SUMMARY を読んだ結果)

- 対象: diffbot 2.0bpw pack + `vllm/vllm-openai:deepseekv41-flash-0909` + diffbot の sm_120 overlay。**これ以外の image tag では試されていない**【資料】。
- 方式: `DeepseekV41ModelState.prepare_inputs` (graph の外で eager に走る) で、hash kernel を先に実行する。`.cpu()` で同期して row id をホストに取り、`preadv` で pinned buffer に集める。それを 1 回の H2D で常駐 GPU tensor (`staged_rows`) へ送る。graph 化した forward は staged の行を読むだけになる。
- 安全装置: prestage 有効時に `lookup()` に来たら raise する。`prestaged_tokens != T` でも raise する。`DSV41_ENGRAM_PRESTAGE_CHECK=1` (eager 専用) は、従来の lookup と prestage の行が byte 一致するかを検査する。
- 制約: Engram DP=1、ubatch 1 個 (DBO 無し) のみ。TP2 は問題ない。
- 実測 (2x RTX PRO 6000、host 125GB、PCIe4 x4 NVMe、thinking off、2K): c1 80.2、c4 229.2、c8 449.7 tok/s、prefill 1,719 (1 本) / 3,324 (合計) tok/s、GSM8K-200 97.5%、RSS 18GB、page cache 約 107GB【資料】。
- 残りの差 (pinned 比 71%) の原因は、行の遅延ではなく 1 step ごとの固定費 (hash kernel、ホスト同期、Python の gather ループ、CPU dequant) と書かれている【資料】。
- 3234 token を超える長文 prefill は測られていない【資料】。
- 補足: sfxnz の現 HEAD (`3de9146`) にも別実装の `apply_engram_prestage.py` がある (GB10 向けの Tony/Kai 方式。arm64 の 0909 digest に当てる前提)。今回は使っていない。

---

## 2. イメージ

| イメージ | 作り方 | サイズ |
|---|---|---|
| `vllm/vllm-openai:deepseekv41-flash-0909@sha256:4f3c8bcf…a9b` (ベース、x86_64) | `docker pull` | 30.4GB 展開後 / 8.64GB 圧縮 (amd64)。docker hub から約 1〜1.5MB/s しか出ず、取得に約 2 時間かかった |
| `dsv41-flash-exl3-sm120` | `NOBUILDKIT=1 MAX_JOBS=12 recipe/docker/build.sh` (diffbot そのまま) | 32.4GB / 9.39GB (ベースとの差 +2GB) |
| `dsv41-flash-exl3-sm120-prestage` | `build/build_vllm_exl3_prestage.sh` (docker run + commit) | 32.5GB / 9.43GB (+0.1GB) |

- `dsv41-flash-exl3-sm120` の中身は、0909 day-0 image に次を加えたもの。
  - sfxnz `apply_engram_disk.py` (`engram.py` と `weight_utils.py` を書き換え、`engram_disk.py` を置く)
  - vllm-exl3 `d3cfd394` (Python plugin のみ。`VLLM_EXL3_NO_CUDA=1`)
  - ExLlamaV3 `5be88657` (v1.4.9) を `TORCH_CUDA_ARCH_LIST=12.0a` でビルドしたもの
  - sfxnz の `sitecustomize.py`
- xmoe: `IMAGE=dsv41-flash-exl3-sm120 recipe/kernels/build.sh` → `recipe/kernels/build/xmoe/` (K=2 MCG のみに絞った `exl3_moe` 拡張)。
- `dsv41-flash-exl3-sm120-prestage` は、diffbot の `serve.sh` と nktlabs の `serve-prestage.sh` が bind mount していたものを焼き込んだもの。
  - nktlabs の 4 ファイル
  - diffbot の `patches/vllm/sitecustomize.py`
  - FlashInfer `sparse_mla_sm120_prefill.cu` (32-token extra page 対応版)
  - `flashinfer_jit_cache/jit_cache/sparse_mla_sm120` を空にしたもの。prebuilt を消し、初回起動で JIT させる (diffbot の `empty-aot` mount と同じ効果)
  - `/opt/xmoe/build/xmoe`
  - `/opt/dsv41-ab/dryrun.py`
  - 元のファイルは `/opt/dsv41-ab/orig/` に残した
  - ENV: `VLLM_PLUGINS=vllm_exl3 DSV41_ENGRAM_DISK=1 DSV41_ENGRAM_PRESTAGE=1 XMOE_EXT=xmoe XMOE_DIR=/opt/xmoe/build`

ビルドの記録 (ログ: `build/vllm_exl3_build.log`, `vllm_exl3_prestage_build.log`)【実測】:

- diffbot `build.sh` は `NOBUILDKIT=1 MAX_JOBS=12` で一度で通った。`apply_engram_disk: patched …engram.py and …weight_utils.py`、`vllm-exl3-0.4.2`、`exllamav3-1.4.9` (sm_120a、wheel 92MB) が入り、`exllamav3_ext.exl3_moe` があることも確かめた。apt の取得と ExLlamaV3 のビルドを合わせて約 15 分かかった。
- `docker` は sudo が要る。そこで `exec sudo docker "$@"` の shim (`build/vllm_exl3_overlay/docker-sudo-shim.sh` を `docker` という名前で置いたもの) を作り、PATH の先頭に足して `build.sh` / `kernels/build.sh` を無改造で走らせた。
- xmoe: `kernels/build.sh` で 15 秒。`recipe/kernels/build/xmoe/xmoe.so` ができた。
- 派生イメージで 1 回だけ失敗した。原因は、image の sitecustomize (sfxnz 版) が `python3 -c` の起動時に vLLM のログを stdout に出し、パッケージの path 取得を汚したこと。`find_spec` の結果に接頭辞を付けて `sed` で抜き出すように直した。
- 本番では使わないが、GPU 無しの container では `vllm._C` が無いという WARNING が出る (この image は `_C_stable_libtorch` を使う)。`libcuda.so.1` が無いため `vllm.platforms.cuda` も import できない。dry-run では `LD_LIBRARY_PATH=/usr/local/cuda/compat` (image 内の CUDA 13.0 compat lib) を付けて回避した。
- ディスク: この作業で増えたのは約 45GB (ベースの展開 30GB と圧縮 9GB、差分 2GB 強)。作業中に空きが 352GB から 210GB に減ったが、大半は並行して走っている requant ジョブ (`models/dsv41-v1-work` 111GB) の分である。

---

## 3. 互換性の確認

### 3.1 pack v0 と diffbot 2.0bpw の形式

- `config.json`: 完全一致 (`quant_method: exl3`、`bits: 2`、`codebook: mcg`、`mtp_experts: source`、`mtp_experts_start_layer: 40`、`non_routed_quantization` = `deepseek_v4_fp8` / ue8m0 / [32,32] / `expert_dtype: fp4`)【実測】。
- index: 188,245 テンソル。Engram は `layers.1.engram.*` が shard 47、`layers.14.engram.*` が shard 48 にある (diffbot の README と同じ配置)。MTP (DSpark) の experts は原本のまま (`mtp.0.ffn.experts.0.w1.weight/scale`)【実測】。

### 3.2 1017 Engram 表を disk reader で読めるか

`DRY_RUN=1 serve_vllm_exl3.sh` (GPU 無しの container) の結果【実測、ログ `build/vllm_exl3_dryrun.log`】:

```
[OK]   vllm serve 引数の解析: … tp=2 quant=exl3 max_num_seqs=16 spec={'method': 'dspark', 'num_speculative_tokens': 5, …}
       capture_sizes=[1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96] cudagraph_mode=FULL_DECODE_ONLY prefix_cache=None max_model_len=524288 kv_mem=8589934592
[OK]   ModelConfig の生成 (config.json + quantization=exl3): arch=['DeepseekV41ForCausalLM'] quant=exl3 dtype=torch.bfloat16 max_model_len=524288 engram_layers=[1, 14]
[OK]   vllm_exl3 / exllamav3_ext / xmoe の import: exllamav3_ext.exl3_moe = xmoe.exl3_moe (/opt/xmoe/build/xmoe/xmoe.so)
[OK]   prestage パッチ済みモジュールの import: prestage enabled=True disk=True
[OK]   Engram disk reader で 1017 の行を読む: L1 rows=384006168 shape=(384006168, 256) scale=(384006168, 8) ok; L14 rows=384016682 shape=(384016682, 256) scale=(384016682, 8) ok
DRY-RUN PASS
```

- Engram の検査では、各層の乱択 64 行と先頭・末尾の行を `DiskEngramTable.gather_dequant` で読んだ。それを safetensors の直接読み (FP8 × ue8m0 scale → bf16) と比べ、bit 一致した。行数 384,006,168 / 384,016,682 は 1017 の config の `engram_num_embeddings` と一致する。
- `prefix_cache=None` は「既定に任せる」という意味である (0909 の `CacheConfig.enable_prefix_caching` の既定は True)。
- dry-run で確かめられないもの: 重みのロード、`create_engine_config` (GPU の推定が要る)、DSpark drafter の構築、graph capture、FlashInfer の JIT。

### 3.3 nktlabs の 4 ファイルと image の差分

`PATCH-SUMMARY.md` の手順どおり、0909 image (amd64、`4f3c8bcf`) から `vllm/models/deepseek_v4_1` を取り出して比べた。engram.py は、sfxnz `460ac89c` の `apply_engram_disk.py` を当てた後のものと比べた【実測。diff は `research/tmp/r4/diff_*.txt`】。

| ファイル | diff 行数 | 削除行 | 内容 |
|---|---:|---:|---|
| `common/engram.py` | 117 | 5 | `_disk_lookup` を `disk_gather_local` + 薄い wrapper に分けた (削除 5 行はこの分割だけ)。`prestage()` / `check_prestaged()`、lookup 到達時の raise、pinned staging の確保を追加 |
| `common/engram_disk.py` | +12 | 0 | `engram_prestage_enabled()` / `engram_prestage_check_enabled()` だけ (sfxnz `460ac89c` 版との差) |
| `nvidia/model.py` | 244 | 0 | 追加のみ (`engram_prestage`、`engram_prestage_dummy`、`_check_engram_prestage`、`_PrestageTimer`、forward の分岐) |
| `nvidia/model_state.py` | 65 | 0 | 追加のみ (`_engram_prestage_model`、`prepare_inputs` の hook、capture 用 dummy) |

- 合計 ~440 行で、README の「~450 diff lines」と合う。**既存のコードを上書きしている箇所は無い。0909 image + sfxnz `460ac89c` の組み合わせにはそのまま当たる。**
- pinned staging は `max_num_batched_tokens` × `part_n_hash_cols` (12) × 256 × bf16 = 4096 × 12 × 256 × 2 = 24MiB/層・rank (2 層 × 2 rank で計 96MiB)【計算】。

---

## 4. Engram の置き方: 3 案の比較 (当機の数値)

当機の前提【実測】: `MemTotal` 121.5GiB。評価ジョブ稼働中で used 15GiB。swap 121GiB (使用 4〜7GiB)。NVMe は Fanxiang S880 2TB (MAXIO MAP1602、DRAM-less、PCIe4)。

NVMe 実測 (shard 47/48 に対する 4KiB O_DIRECT のランダム読み、Python 32 thread、`build/vllm_exl3_overlay/nvme_rows.py`)【実測】:

| 1 回にまとめて読む行数 | 当てはまる場面 | p50 | p99 | 実効 IOPS |
|---:|---|---:|---:|---:|
| 48 | 1 token (c1、DSpark 無し) | 0.64 ms | 1.24 ms | 74k |
| 288 | c1 + DSpark (6 token) | 2.72 ms | 7.98 ms | 106k |
| 1,536 | c~5 + DSpark | 11.6 ms | 19.6 ms | 132k |
| 4,608 | c16 + DSpark (96 token) | 36.1 ms | 51.3 ms | 128k |

nktlabs の機材は 48 行 cold で p50 1.41 ms だった。当機の NVMe はそれより約 2 倍速い。ただし上の値はすべてページキャッシュに無い場合 (cold) の値である。

| 案 | host RAM | pinned | decode c1 | c8〜c16 | prefill | 実装の手間 | 判定 |
|---|---|---|---|---|---|---|---|
| A. pinned (diffbot 既定) | ~280GB | ~190GiB | 113 | c4 277 | 4,044 (46K) | 0 | **不可** (RAM 121GiB) |
| B. disk + eager (sfxnz `DSV41_ENGRAM_DISK=1`, `EAGER=1`) | ~12GB RSS | ≈0 | 26〜31 | c4 108 (線形) | 1,805 (46K) | 0 | 起動は可。遅い |
| C. MXFP4 Engram を pin | 97GiB + RSS ~18 = ~115GiB | ~97GiB | ~100〜110【推測】 | pinned 並み | pinned 並み | 大 | **不可に近い**。pin の上限 ~90GB を超える。page cache が ~5GiB しか残らず、モデルロード (~160GB の mmap 読み) とも競合する。読み手 (PR #61530) の 0909 への移植と、変換スクリプトも要る |
| **D. disk + prestage + graph (nktlabs)** | ~18GB RSS + page cache | 数 MiB | **80** | **c8 450** | 1,719 (2K) | 小 (焼き込み済み) | **採用** |

- B (eager) の根拠: diffbot README の値 (46K で decode 31.5、prefill 1,805) と nktlabs の実測 (26.1 / c4 107.6)【資料】。
- C は、RAM が 128GB に収まるかどうかの瀬戸際になる。pin は swap できないので、OOM killer の対象を減らす逃げ道も無い。D の page cache 方式 (r2_engram_hit: 4KiB page LRU 64GB で decode ヒット ~85%、prefill 1.11 IO/token) のほうが同じ RAM で柔軟に効く。
- 結論: **D を既定にし、B は比較・切り分け用に残す** (`ENGRAM_PRESTAGE=0 EAGER=1`)。

---

## 5. 起動

### 5.1 手順 (GPU が空いてから)

```bash
# 0. ds41rt と評価ジョブを止める (VRAM 2x96GB をほぼ全部使う。page cache も奪い合う)
# 1. 初回だけ: parity 検査 (eager、prestage の行 = 従来 lookup の行を毎 step で照合)
PARITY=1 THINKING=off build/serve_vllm_exl3.sh
#    → 別端末で短い会話と 3K〜12K の prompt を数本流し、"parity failed" が出ないことを確かめて Ctrl-C
# 2. 本番 (CUDA graph + DSpark + prestage)
build/serve_vllm_exl3.sh
# 3. smoke (diffbot の bench を口 8001 に向ける。API_KEY は未設定なら任意文字列で可)
```

- 生成される docker コマンド (`DRY_RUN=1` の出力、本番も同じ):

```
sudo docker run --rm --name dsv41-vllm-exl3 --gpus all --network host --ipc host --shm-size 16g --memory 112g --memory-swap 112g --cap-add IPC_LOCK --ulimit memlock=-1:-1 -v $MODELS:$MODELS:ro -v build/vllm_exl3_cache:/root/.cache -e VLLM_PLUGINS=vllm_exl3 -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 -e DSV41_ENGRAM_PRESTAGE=1 -e DSV41_ENGRAM_PRESTAGE_CHECK=0 -e XMOE_EXT=xmoe -e XMOE_DIR=/opt/xmoe/build -e TORCH_CUDA_ARCH_LIST=12.0a -e FLASHINFER_CUDA_ARCH_LIST=12.0f -e FLASHINFER_DISABLE_VERSION_CHECK=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e NCCL_P2P_DISABLE=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 --entrypoint vllm dsv41-flash-exl3-sm120-prestage serve $MODELS/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw --tensor-parallel-size 2 --host 0.0.0.0 --port 8001 --max-model-len 524288 --kv-cache-dtype fp8 --kv-cache-memory 8589934592 --gpu-memory-utilization 0.92 --max-num-seqs 16 --max-num-batched-tokens 4096 --kernel-config \{\"enable_flashinfer_autotune\":false\,\"enable_jit_warmup\":false\} --block-size 64 --quantization exl3 --language-model-only --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 --served-model-name deepseek-v4.1-flash --trust-remote-code --compilation-config \{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"\,\"custom_ops\":\[\"all\"\]\} --cudagraph-capture-sizes 1 2 4 6 8 12 16 24 32 48 64 80 96 --disable-custom-all-reduce --attention-config \{\"indexer_kv_dtype\":\"mxfp4\"\} --linear-backend marlin --speculative-config \{\"method\":\"dspark\"\,\"num_speculative_tokens\":5\,\"draft_sample_method\":\"probabilistic\"\} 
```

### 5.2 RAM を 128GB に収める設定

| 項目 | 設定 | 効果 |
|---|---|---|
| Engram | `DSV41_ENGRAM_DISK=1` + `DSV41_ENGRAM_PRESTAGE=1` | 190GiB の pin を 0 にする。prestage の pinned staging は 4096 token × 12 列 × 256 × bf16 = 24MiB/層・rank (計 96MiB)【計算】 |
| weight loader | Engram の embed / scale を skip (`apply_engram_disk` の `weight_utils` パッチ) | ロード中に 203GB を読まない |
| KV | `--kv-cache-memory 8GiB/GPU`、KV の CPU offload 無し、V1 なので CPU swap space も無し | ホスト側の KV は 0 |
| container | `--memory 112g --memory-swap 112g` | page cache 込みで 112GiB を上限にする。cgroup v2 の `memory.swap.max=0` で swap を使わせない (当機で `memory.max=120259084288`, `memory.swap.max=0` を確認【実測】。`--memory-swappiness` は cgroup v2 では無視されるので付けない)。熱い Engram 行は、この内側の page cache が持つ |
| shm | `--ipc host` (NCCL 用)。`--shm-size` は ipc host では効かない | — |
| 起動前検査 | GPU の使用が 4GB 超、口 8001 が使用中、`MemAvailable` が 60GiB 未満なら中止 (`FORCE=1` で無視) | 評価ジョブや ds41rt との同居を防ぐ |

- ホストの swap そのもの (121GiB) は止めていない。止めたい場合は `sudo swapoff -a` を手で実行する。container の swap は上の設定で無効になる。

### 5.3 serve.sh からの変更点 (diffbot 2.0bpw 版との差)

- `MAX_NUM_SEQS` 8 → **16**。capture size は `1 2 4 6 8 12 16 24 32 48 64 80 96` (16 × (1+5))。
- 口 8000 → **8001** (8000 は ds41rt)。`--api-key` は `API_KEY` を設定したときだけ付ける (ds41rt に認証が無いので、A/B は認証無しで揃える)。
- bind mount を廃止して派生イメージに焼き込んだ。キャッシュ (`/root/.cache`) は `build/vllm_exl3_cache` に固定した (FlashInfer の JIT と torch.compile を 2 回目以降に再利用するため)。
- `HF_HUB_OFFLINE=1` を付けた。

---

## 6. 予想性能 (根拠付き)

条件: thinking off、2K 文脈、DSpark 5、xmoe flat scheduler、Marlin dense、Max-Q 300W。

| 指標 | 予想 | 根拠 |
|---|---:|---|
| decode c1 | 75〜90 tok/s | nktlabs 80.2【資料】。NVMe は当機が約 2 倍速い (48 行 p50 0.64 vs 1.41 ms)。ただし差の主因は固定費なので、伸びは数 % にとどまる【推測】 |
| decode c4 | 210〜240 | nktlabs 229.2【資料】 |
| decode c8 | 420〜470 | nktlabs 449.7【資料】 |
| decode c16 | 500〜650 | 誰も測っていない。step あたり 96 token × 48 = 4,608 行を集める。cold なら 36 ms【実測】、page cache で 85% 当たれば ~700 IO ≈ 6 ms【計算】。これに Python の gather ループ (16 行 chunk × 288 個) が 10〜20 ms 乗る【推測】。c8 は 50 ms/step 前後 (450 tok/s ÷ (8 × τ2.8)) と読めるので、c16 は 75〜90 ms/step、16 × 2.8 / 0.08 ≈ 560【計算】 |
| decode 46K c1 | 60〜80 | pinned 116.6 × prestage 比 0.71 ≈ 83。長文で hash の固定費が増える分を引く【推測】 |
| prefill (1 本) | 1.5〜2.2k tok/s | nktlabs 1,719【資料】。4096 token の chunk ごとに約 19.7 万行を Python で集める。日本語は IO が 3 倍多い (r2_engram_hit: prefill 1.80 vs code 0.59 IO/token) |
| prefill (4 本合計) | 3〜3.5k | nktlabs 3,324【資料】 |
| TTFT 46K 新規 | 20〜30 s | 上の prefill から【計算】 |
| 品質 | GSM8K-200 ~97.5% (thinking off) | 0909 pack の値【資料】。1017 は再測定が要る |

ds41rt (0-spark、2.0bpw) の見込み (r2_ds41rt): c1 code 180〜210 / 加重 115〜135、c16 合計 code 1,100〜1,350 / 混在 330〜380、prefill 3.5〜5k。**decode と prefill は ds41rt の方が速いと予想する。** vLLM 側が勝ちうるのは混在負荷の c16 (500 超を出せれば) と、API の機能である。

---

## 7. 既知の問題・懸念

1. **#56771 (open): SM120 + DSpark + 長文 prefill で illegal memory access。**
   - 発生条件は、CUDA graph が prefill にも効く `FULL_AND_PIECEWISE` のとき。`--enforce-eager` と `cudagraph_mode: NONE` では起きない【資料】。
   - このスクリプトは diffbot と同じ `FULL_DECODE_ONLY` (prefill は eager) を使う。diffbot は 46K × 4 本を DSpark で流して問題が出ていない【資料】。
   - 2026-10-17 に、新しい nightly (`49cb2bbf` + #60762) で再現しないという報告がある。ただし 0909 image はその修正より古い。**最初に 12K / 46K / 120K の prompt で確かめる。**
2. **prestage は 3,234 token を超える prefill と c8 超で未検証**【資料】。c16 では staging (4096 token 分) は足りるが、Python の gather が律速になるおそれがある。
3. **Engram の IO が rank 1 に偏る** (miss の約 75%、r2_engram_hit §0-5)。rank 0 が待たされ、step 時間は遅い方の rank で決まる。
4. **page cache の奪い合い**: ds41rt (Engram を mmap で持つ) や評価ジョブと同時に動かすと、熱い行が追い出される。A/B は必ず片方ずつ動かし、各回の前に `vmtouch` / `/proc/meminfo` の Cached を記録する。
5. 4 本の 46K prompt を同時に prefill すると、他の stream の decode が合計 ~30 tok/s まで落ちる (chunked prefill の干渉)【資料】。
6. custom all-reduce は無効 (graph capture で IPC handle が落ちる)。NCCL で 1 step あたり ~1.6 ms かかる【資料】。
7. tool call の parser は 0909 版のまま。diffbot が後に取り込んだ修正は入っていない (#59585: NaN/Infinity 引数の JSON 化、#61142: 崩れた DSML opener)。ds41rt より tool call の頑健性が低いかもしれない。
8. vision は無効 (`--language-model-only`。SM120 の FlashInfer decode に top-k 1152 の dispatch が無い)【資料】。
9. 初回起動では FlashInfer `sparse_mla_sm120` prefill の JIT、torch.compile、graph capture が走る。ロードは nktlabs で 159 s【資料】。初回は +数分を見込む。
10. diffbot 2.0bpw (0909) の KL は neutral 0.361 / code 0.215 / agentic 0.077 と大きい【資料】。pack v0 も同じ作り方 (校正なし、beam 16) なので、品質は同程度と見込む。
11. MAX_NUM_SEQS 16 は diffbot・nktlabs のどちらも測っていない。capture 96 まで増やすと graph 用の VRAM が増える。`UTIL 0.92` / `KV_MEM 8GiB` で足りなければ、`KV_MEM=6442450944` に下げる。

---

## 8. ds41rt との機能差

| 機能 | vLLM (この構成) | ds41rt v15 (r2_ds41rt §0-5) |
|---|---|---|
| `/v1/chat/completions` (SSE, tools, 並列 tool calls) | あり (`deepseek_v41` tool / reasoning parser) | あり |
| `/v1/completions` | あり | **無し** |
| `/v1/responses` | あり (0909 image に `entrypoints/openai/responses/` と `/v1/responses` の router がある【実測】)。reasoning / tool parser との組み合わせは要確認 | **無し** |
| logprobs / top_logprobs | あり (vLLM 標準)。DSpark (投機) 併用時の値が正しいかは要確認 | **無し** |
| n > 1、presence / frequency / repetition penalty | あり (vLLM 標準) | **無し** |
| 構造化出力 (JSON Schema) | あり (`response_format` / structured outputs、xgrammar)。DSpark との併用は要確認 | あり |
| API キー認証 | あり (`API_KEY` を設定したとき) | 無し |
| prefix cache | APC (block 64 のハッシュ)。diffbot の実負荷でヒット率 87%【資料】 | token radix + COW。部分一致では encoder の末尾 128 token を作り直す |
| 画像入力 | **無効** (`--language-model-only`) | 16 枚まで |
| 最大文脈 | 524,288 (KV 8GiB で ~3.8M token 分。1M も可) | 1,048,576 |
| 並列数 | 16 (設定で増やせる) | 16 が上限 |
| メトリクス | `/metrics` (Prometheus)、`/tokenize` | ブラウザのコンソール |
| Engram | FP8、NVMe preadv + page cache、prestage | FP8、mmap (MAP_PRIVATE + MADV_RANDOM) |

---

## 9. GPU で最初に確かめること (順番)

1. `DRY_RUN=1` が PASS すること (済み。§3.2)。ds41rt・評価ジョブ・requant ジョブが止まっていて、`MemAvailable` が 100GiB 以上あることも確かめる。
2. `PARITY=1` (eager) で起動し、ロードが通るか確かめる (0909 コード × 1017 重み × pack v0)。あわせて `nvidia-smi` で各 GPU の残り VRAM、`/proc/meminfo` で RSS と Cached を見る。短い prompt と 3K〜12K の prompt で "parity failed" が出ないこと、出力が筋の通った文になることを確かめる。
3. 本番 (graph) で起動し、graph capture (96 まで) が OOM しないか確かめる。12K / 46K / 120K の prompt で #56771 が出ないことを確かめる。
4. GSM8K-200 (thinking off) を diffbot の `bench/quality-probe.py` で流し、97% 前後か確かめる (prestage と 1017 の両方の検算になる)。
5. `bench-sbs.py` と `build/bench_conc.py` で c1 / c4 / c8 / c16 を測る。ds41rt と同じ負荷にそろえ、片方ずつ動かす。`_PrestageTimer` のログ (hash / gather の ms)、`iostat -x` の r/s、`/proc/vmstat` の pgmajfault も記録する。
