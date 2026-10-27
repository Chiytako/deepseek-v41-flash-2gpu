# ds41rt-patch: ds41rt v15 への改修

[ds41rt](https://github.com/tpurtell/ds41rt) (MIT、T.J. Purtell) の upstream `3067d06` (v15) に対する改修。中身は 3 つある。

1. **reasoning effort を整数 1〜100 で受ける。** DeepSeek-V4.1 のプロンプトは `Reasoning Effort: N` で思考の深さを受けるが、
   公式の deepseek-recipe 0.1.0 と ds41rt v15 は段階名 (`low` / `high` / `max` など) しか受けない。
   改修版は、整数 (`60`)・数値文字列 (`"60"`)・段階名のどれでも受け、V4.1 には数値をそのまま描く。
2. **1 台 (2x RTX PRO 6000、DGX Spark なし) で動かすための起動の差分と設定の雛形。**
3. **raw EXL3 パックで、専門家ごとに K2 と K3 を混ぜられるようにする。** 上流の v15 は `config.json` だけで量子化を宣言するパック (raw publication) を
   全専門家同じビット数としてしか読まない。改修版は `quantization_config.expert_bits` (`{"<層>.<専門家>": 3}`) を読み、
   その専門家の w1/w3/w2 を K3 として扱う。K2/K3 の 2 段の配置とカーネル (`exl3-k23`) は上流にもともとあるので、変えたのは読み込みだけ。
   2026-10-27 以降の HF の重み (日本語 OCR で使われる専門家 1,280 個を K3 にしたもの) は、**この改修が無いと読めない**。
   この重みは K3 のぶん GPU メモリを 5.3GiB 多く使う。2x 96GB では設定に `KV_POOL_SIZE=7.5GB` (既定は約 1,400 万トークン = 13.1GB) を足し、
   CUDA グラフと画像エンコーダの余白を 1 枚あたり約 3GB 残す。既定のままだと、16 本同時の長い要求のあとに画像の要求がメモリ不足で失敗した。

ds41rt の Rust のバイナリだけを差し替えた派生イメージ `ds41rt-coordinator:v15-effort` を作る。ネイティブライブラリ (`libds41rt_native.so`)、
AOT カーネル、Python のパッケージ、v15 のラベルは公開イメージ `ghcr.io/tpurtell/ds41rt-coordinator:v15` のまま引き継ぐ。

## 中身

| ファイル | 内容 |
|---|---|
| `patches/0001-ds41rt-api-accept-numeric-reasoning-effort-1-100.patch` | `rust/Cargo.toml` に `[patch.crates-io]` を足して改修版の recipe を使う。`ds41rt-api` の `request.rs` (effort の解析)、`encoding.rs` (V4 の段階への丸め)、`native_v41.rs` (試験) |
| `patches/0002-Add-a-derived-coordinator-image-with-the-numeric-eff.patch` | `docker/Dockerfile.effort` と `scripts/build-effort-image.sh` (GPU なしで派生イメージを作る) |
| `patches/0003-Accept-per-expert-K-tiers-in-raw-EXL3-publications.patch` | `ds41rt-loader` の `v41_exl3.rs` (`expert_bits` の解釈と試験) |
| `third_party/deepseek-recipe-patched/` | deepseek-recipe / -core / -encoding 0.1.0 (MIT、DeepSeek) の改修版。ds41rt の `third_party/` にそのまま置く。中身は [その README](third_party/deepseek-recipe-patched/README.md) |
| `run-local.sh.diff` | `run.sh` → `run-local.sh` の差分。docker の `--device=/dev/infiniband` を外す (InfiniBand / RoCE の無い機械では、これがあるとコンテナが起動しない) |
| `local.config.example` | 1 台用の設定の雛形 (`SPARK_COUNT=0`、`RTX_GPUS=2`、`RTX_EXPERT_LAYERS=40`、`EXPERT_FORMAT=exl3`、`SPARKINFER_EXL3=force`、`HOST_CACHE_BYTES=4GiB`、`PREFIX_CACHE_ENTRIES=8`) |
| `local.config.diff` | 雛形と upstream の `ds41rt.config` の差分 |

## 当て方とイメージの作り方

```sh
git clone https://github.com/Chiytako/deepseek-v41-flash-2gpu   # このリポジトリ
git clone https://github.com/tpurtell/ds41rt && cd ds41rt
git checkout 3067d06
cp -r ../deepseek-v41-flash-2gpu/ds41rt-patch/third_party/deepseek-recipe-patched third_party/
git am ../deepseek-v41-flash-2gpu/ds41rt-patch/patches/*.patch
cp run.sh run-local.sh && patch run-local.sh ../deepseek-v41-flash-2gpu/ds41rt-patch/run-local.sh.diff

sudo docker pull ghcr.io/tpurtell/ds41rt-coordinator:v15
scripts/build-effort-image.sh
```

`build-effort-image.sh` がすること:

- v15 の coordinator イメージの中で (同じ Ubuntu 24.04 / glibc 2.39、rustc 1.98) Rust のバイナリをビルドする。GPU は使わない。
  初回は rustup と crates.io に接続する。キャッシュは `~/.cache/ds41rt/builds/effort`。
- ビルドの前に試験を回す (recipe のワークスペースと `ds41rt-api`)。省くときは `DS41RT_EFFORT_SKIP_TESTS=1`。
- できたバイナリを `docker/Dockerfile.effort` で v15 に重ね、`ds41rt-coordinator:v15-effort` にする。
- 動いているコンテナには触れない。docker は `sudo` で呼ぶ。

| 環境変数 | 既定 |
|---|---|
| `DS41RT_EFFORT_BASE_IMAGE` | `ghcr.io/tpurtell/ds41rt-coordinator:v15` |
| `DS41RT_EFFORT_IMAGE` | `ds41rt-coordinator:v15-effort` |
| `DS41RT_EFFORT_CACHE` | `~/.cache/ds41rt/builds/effort` |
| `DS41RT_EFFORT_CONTEXT` | `$DS41RT_EFFORT_CACHE/context` (docker build の文脈。バイナリだけを置く) |
| `DS41RT_EFFORT_RUST` | `1.98` |

## 起動

```sh
hf download ChiTako/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-ja-calibrated             # 重み (334GB) を HF キャッシュ ($HF_HOME、既定 ~/.cache/huggingface) に
cp ../deepseek-v41-flash-2gpu/ds41rt-patch/local.config.example local.config
$EDITOR local.config                        # MODEL_ID を重みの repo にする。MODEL_REVISION は空なら refs/main を読む
./run-local.sh --config local.config        # 0.0.0.0:8000
./run-local.sh --config local.config --restart   # 動いているものを入れ替える
```

- `COORDINATOR_GPU_UUID` と `COORDINATOR_GPU_PCI_BUS_ID` は、`nvidia-smi` で調べて固定しておくと、再起動で GPU の番号が変わっても同じカードが RTX0 になる。
- 改修していない v15 のまま使うときは、`COORDINATOR_DOCKER_INFERENCE=ghcr.io/tpurtell/ds41rt-coordinator:v15` にする (effort は段階名だけになり、`medium` と `xhigh` は 75 になる)。
  ただし K2/K3 混在の重み (2026-10-27 以降の HF の main) は上流の v15 では読めない (`unexpected quantization_config key expert_bits` で止まる)。それより前の全 K2 の版を使う。
- 起動は 50 秒ほど。`curl -s localhost:8000/health` で確かめる。

## effort の写し方

| 要求の値 | V4.1 のプロンプト | 公式 (deepseek-recipe 0.1.0、ds41rt v15) |
|---|---|---|
| `none` / `off` / `disabled` | 思考なし | `none` のみ |
| `minimal` / `low` | 50 | 50 |
| **`medium`** | **62 (独自)** | 75 |
| `high`、未指定 | 75 | 75 |
| **`xhigh`** | **88 (独自)** | 75 |
| `max` (Anthropic は `ultra` も) | 100 | 100 |
| 整数 1〜100、数値文字列 `"60"` | そのまま | 受けない |

**`medium`→62 と `xhigh`→88 は公式と違う、この改修だけの対応である。**
DeepSeek の公式の encoding が段階名で定めるのは low (50)・high (75)・max (100) の 3 つだけで、
公式の deepseek-recipe は OpenAI / Anthropic の API の名前を受けるために `medium` と `xhigh` を high と同じ 75 に写す
(crates.io 版 0.1.0 の `dsv41.rs` と各 API の `convert.rs`)。そのままでは `medium`・`high`・`xhigh` が同じプロンプトになる。
V4.1 の尺度は 1〜100 なので、`medium` を low と high の間 (62.5 の切り捨て)、`xhigh` を high と max の間 (87.5 の切り上げ) に置いた。

V4 (数値の尺度が無いモデル) では ≤61 を low、62〜88 を high、89 以上を max に丸める。`medium` と `xhigh` は公式どおり high になる。

公式どおりの値がよいときは、整数 75 を送るか、プロキシ ds41gw の `thinking.effort_map` で `medium = 75, xhigh = 75` にする。

## 確かめたこと (手元)

- 試験: recipe のワークスペース 4 + 8 + 2 件、`ds41rt-api` 239 件がすべて合格。
- `render-effort` (`cargo run --manifest-path third_party/deepseek-recipe-patched/Cargo.toml -p render-effort -- '"medium"' 62 '"xhigh"' 88`):
  `"medium"`・`62`・`"62"` はどれも `Reasoning Effort: 62`、`"xhigh"` と `88` は `Reasoning Effort: 88` になる。
- 実機 (temperature 0、seed 7): ds41rt 直で `medium` と `62`・`"62"`、`xhigh` と `88`・`"88"` の出力が一致し、`low`=50、`high`=75、`max`=100 と一致した。
  `medium` の出力は `high` とも `low` とも違い、`xhigh` も `high` とも `max` とも違った。

## 注意

- `scripts/build-effort-image.sh` の docker build の文脈の既定は、手元の版 (作業場の `build/ds41rt-effort`) から `$DS41RT_EFFORT_CACHE/context` に変えてある。それ以外は手元で動かしているものと同じである。
- ds41rt の HTTP の口は `0.0.0.0:8000` で待ち受ける。外に出したくないときは `ADDR=127.0.0.1:8000` にして、プロキシだけを外に出す。
