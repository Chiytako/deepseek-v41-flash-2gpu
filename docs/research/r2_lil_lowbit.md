# R2: LIL vLLM (release-10152026) の V4.1 経路で experts を 2〜3bit にして 2 GPU で動かす道

調査日: 2026-10-18。GPU は使っていない。稼働中の `glm53v` には触れていない。

- 対象イメージは `ghcr.io/local-inference-lab/vllm:release-10152026` (image id `745feb916ee8`)。
- ソースは `ext/lil_src/{vllm,flashinfer}` に展開済みのものを読んだ。
  - これはイメージ内の `/opt/venv/lib/python3.12/site-packages/{vllm,flashinfer}` と同一である。`sudo docker run --rm --network none --entrypoint bash <image> -c 'md5sum …'` で 3 ファイル (`deepseek_v4_1/nvidia/model.py`、`b12x/moe/checkpoints/exl3.py`、`b12x/sequence/_shared/disk_table.py`) の md5 が一致することを確かめた。
  - 以下のパスは、`vllm/…` がイメージの `site-packages/vllm/…`、`b12x/…` がイメージの `site-packages/flashinfer/experimental/b12x/…` を指す。
- Web から小さく取得したものは `research/tmp/r2l/` に置いた: CSF リポジトリの README とファイルサイズ一覧、LIL `rtx6kpro` の V4.1 ページ、LIL vllm PR #888、b12x PR #425 / #426。
- 前提の PLAN.md §0 の要点: 置き場が決まらないのは experts 288.8GB である。VRAM の expert 予算は 2 枚で約 152.7GB なので、全 experts を残すなら平均 2.25bit/重みが上限になる。

---

## 0. 結論

1. **LIL の b12x MoE には、2〜3bit の expert 形式が 3 種類実装済みである。** trellis (EXL3 コンテナ `exl3-v1`)、IQ2_XS / IQ2_XXS、Q8_0 である。ただしどれも Kimi-K3 など別モデル用に配線されている。DeepSeek-V4.1 の経路 (`deepseek_v41_fp8` / `mxfp4_csf`) からは呼ばれない。
   - 配線に必要な改修は、vLLM 側の 6〜8 か所 (各数十行) と b12x カーネルの 3 か所 (SwiGLU clamp) が中心になる。表は §1.1 に示す。
2. **swiglu_limit=10 の clamp には、形式ごとに次の問題がある。**
   - MXFP4 と IQ2 / Q8_0 では、カーネルの epilogue が clamp を実装済みである (`_clamp_swiglu_inputs`)。ただし IQ2 系では vLLM 側が clamp を拒否しているか、値を渡していない。2 か所直せば済む。
   - **trellis (EXL3) では、カーネルが「回転付き epilogue と clamp の併用」を明示的に拒否する** (`w4a16/kernel.py:6805-6808`)。clamp を入れる位置は、回転を戻して物理座標に戻った直後である。そこに `min/max` を 3 関数分 (約 30 行) 足せば、数学的に正しく扱える。
3. **TP2 の intermediate 1152 は、形式によってはタイル制約に合わない。**
   - 合う: MXFP4 (現行)、Q8_0、trellis の K2 で intermediate Hadamard を使う場合 (タイル 128)。
   - 合わない: IQ2_XS / XXS は I%256 を要求する。trellis の一律ビット (タイル (64,256,64,256)) は I%256 を要求する。1152 = 4.5×256 なので、どちらも通らない。
   - EP2 (I=2304 のまま) なら、すべての形式が通る。
4. **LIL の SM120 V4.1 経路では、EP は使えない。**
   - `B12xExperts`、MXFP4-CSF、EXL3、IQ2 の各 MoE method が EP を拒否する。
   - 一方で、b12x 側には expert map の仕組みがある (`fused_moe.bind(route_expert_map=…)`、`b12x.moe.ep_moe`)。DP なしの「入力複製型 EP」なら、vLLM 側の数か所を直せば配線できる見込みである。
5. **exl3-v1 は exllamav3 の EXL3 とタイル内配置が一致する。変換はバイトコピーで済む見込みである。**
   - ただし公開済みの V4.1 EXL3 はすべて `mul1` codebook で作られている。b12x は `mcg`、`lut_e4m3`、`lut_fp16` しか受け付けない。
   - さらに、**mcg は 3bit 以上、2bit は LIL 独自の `lut_e4m3` だけ**である。
   - 一律ビットの制約もある。モデル全体で 1 つの bits しか使えず、層ごとや expert ごとにビットを変える形式は DS の形状では使えない。
   - **2.0〜2.25bit を実現するには、lut_e4m3 用の量子化器を自作する必要がある。** LIL の `trellis-quant` は非公開 (GitHub 404) だが、デコード表は b12x に公開されている。
6. **Engram disk は同期読みで、行キャッシュも先読みも無い。**
   - io_uring + O_DIRECT で読む。ページキャッシュは使わない。バッチ内の 4KiB ブロック単位の重複除去だけがある。
   - TP では行範囲を半分ずつ持ち、読んだ後に all-reduce する。
   - 必要 RAM は約 0.1GB と小さい。scale を RAM に常駐させる場合は +6.1GB。
   - コンテナは `seccomp=unconfined` と `memlock=-1` が必須であることを実測した。
   - 改修点 (行キャッシュ、非同期の先読み、4KiB 整列) は §3.5 に挙げる。
7. **MXFP4-CSF で縮むのは scale だけである。**
   - scale 16.99GB → 2.31GB で、experts は 288.8GB → **約 274.1GB**。192GB を大きく超えるので、**量子化は避けられない**。
   - CSF 変換ツール (`trellis_quant.lossless_scale_checkpoint`) は非公開で、イメージにも入っていない。
8. **LIL の expert residency は、このイメージに入っていない。**
   - 中身は PR #888 と b12x #425 / #426 で、SM103 の tcgen05 カーネルである。PR はどれも未マージで closed。
   - x86 / SM120 に移すには、カーネルとローダとプレースメントを作り直す必要があり、4〜8 週かかる。
   - 加えて PCIe Gen5 x8 (約 20GB/s) は GB300 の C2C (約 300GB/s) の 1/15 しかない。cold 側に当たる割合を 1〜2% 未満に抑えない限り遅い。単独の方式としては勧めない。

**推奨 (LIL 経路の中で)**

- 本線: **TP2 + trellis `lut_e4m3` K2 (+intermediate Hadamard) の一律 2bit、全 experts を VRAM に置く**。experts 約 136.6GB、1 GPU あたり 68.3GB。
  - 必要なもの: (a) lut_e4m3 量子化器の自作、(b) DS 用の量子化 config と MoE method、(c) カーネルへの clamp パッチ、(d) Engram への行キャッシュ追加。
- 余った VRAM 予算 (約 16GB) は、層ごとに K3 を混ぜる案 (manifest を層別に分ける改修) に回せる。混ぜられるのは約 9 層。
- 比較用の次善案: **EP2 + IQ2_XS / IQ2_XXS の層別混合** (140〜157GB)。量子化器が既にある (ggml) ので作る手間は少ないが、品質は劣る見込み。

---

## 1. deepseek_v4_1 の量子化分岐に trellis / IQ2 / Q8_0 を配線する

### 1.0 現状の分岐

- `vllm/models/deepseek_v41/__init__.py:16-22`: SM120 では必ず `vllm.models.deepseek_v4_1` (b12x 専用) が選ばれる。
- `vllm/models/deepseek_v4_1/nvidia/model.py:372-376`: 量子化 config の名前が `deepseek_v41_fp8` か `mxfp4_csf` 以外なら例外になる。
- `vllm/models/deepseek_v4_1/quant_config.py:13-47` (`DeepseekV41FP8Config`): Linear は `B12xFP8LinearMethod`、Embedding は `B12xEmbeddingMethod`、`RoutedExperts` は `Mxfp4MoEMethod` に振り分ける (45-46 行)。
- `vllm/models/deepseek_v4_1/mxfp4_csf.py:76-97`: 主層 (`layer < num_hidden_layers`) の experts だけを `Mxfp4CsfMoEMethod` に差し替える。DSpark の層 (40〜42) は親クラスの MXFP4 のままにする。**新しい形式もこの作り方を真似るのが最も素直である。**
- MoE 層を作る場所は `vllm/models/deepseek_v4/nvidia/model.py:780-1035` (`DeepseekV4MoE`)。swiglu_limit は 825 行、`FusedMoEFactory(..., swiglu_limit=…)` は 1010-1032 行。DSpark 層は expert 数だけを差し替えている (`deepseek_v4_1/nvidia/model.py:134-160`)。

### 1.1 改修箇所の一覧

| # | ファイル:行 | 現状 | 改修 |
|---|---|---|---|
| 1 | `vllm/model_executor/layers/quantization/__init__.py:15-53` (`QuantizationMethods` の Literal)、`:162-195` (`method_to_config`)、`vllm/config/model.py:1272-1296` (overrides の順序) | `exl3` と `modelopt_mixed` は Kimi 用、DS 用の低ビット名は無い | 新しい名前 (例 `deepseek_v41_lowbit`) を登録する。checkpoint の `quant_method` を `exl3` にすると Kimi 用の `Exl3Config` が先に拾ってしまう (`exl3.py:71-81`)。そのため別名にする |
| 2 | `vllm/models/deepseek_v4_1/nvidia/model.py:372-376` | 2 つの名前しか許さない | 新しい名前を許可リストに加える |
| 3 | 新規 `vllm/models/deepseek_v4_1/lowbit.py` (`mxfp4_csf.py:37-97` を雛形にする) | — | `DeepseekV41FP8Config` を継承し、主層の `RoutedExperts` だけを新しい MoE method に回す。Linear / Embedding / DSpark は親クラスのまま。層ごとのコーデック (EXL3 の bits、IQ2_XS / XXS) を config から引けるようにする |
| 4 | `vllm/model_executor/layers/quantization/exl3.py:131-282` (`Exl3MoEMethod`)。流用するか複製する | EP / DP を拒否する (136-146)。**活性化は SiTU(4,25) 必須** (147-156)。`ActivationSpec(nonlinearity="situ")` を固定で渡し、swiglu_limit を渡していない (220-227)。manifest の場所は `model_config.model` (198-199) | SiLU を許可し、`nonlinearity="silu"` と `swiglu_limit=moe.swiglu_limit` (=10.0) を渡す。EP にする場合は、自 rank の expert だけを切り出すようにする (§2.4) |
| 5 | `vllm/model_executor/layers/quantization/utils/exl3.py:40-85` (`plan_exl3_extent`) | intermediate の barrier が中央に 1 つあることを要求する。TP は 2〜24 | TP2 ならそのまま使える (rank0 が slot [0,36)、rank1 が [36,72)、各 1152ch)。EP 用には barrier 無しで全 slot を持つ extent を別に書く |
| 6 | `vllm/model_executor/layers/quantization/modelopt_iq2_xs.py:36-80` (`ModelOptIQ2XSMoEMethod`) | EP を拒否する (47)。**swiglu_limit / alpha / beta が設定されていると拒否する** (56-65)。H と I_local は `max(128, 256)` の倍数でなければならない (67-80) | clamp の拒否を外す。EP 時は、ローカルの expert 番号への写像を weight_loader (127-170) に入れる。現状は `param.data[expert_id]` を大域番号のまま使っている (152 行付近) |
| 7 | `vllm/model_executor/layers/fused_moe/b12x.py:472-492` (`_prepare_experts` のブロックコーデック分岐) | IQ2 / Q8_0 の `ActivationSpec` に swiglu_limit を渡していない。packed 分岐 (536-555) は渡している | `swiglu_limit=self._swiglu_params(activation)[0]` を足す |
| 8 | `vllm/model_executor/layers/fused_moe/b12x.py:667-695` (`install_prepared_experts`) | trellis では、`prepared.plan.activation.swiglu_limit != moe_config.swiglu_limit` なら例外になる (686) | #4 で limit を渡せば通る。EP では `num_experts` の比較 (682 付近) をローカルの数で行う |
| 9 | `vllm/model_executor/layers/fused_moe/b12x.py:815-824` (`_supports_parallel_config`)、`:833-834` (`supports_expert_map` が False)、`:1118-1119` (`apply` で expert_map が来ると例外) | EP 不可 | EP 時 (`use_ep and not use_all2all_kernels`) を許す。`plan_execution(capacity=ExecutionCapacity(route_num_experts=384))` と、`fused_moe.bind(route_expert_map=expert_map)` で配線する (b12x 側の `fused_moe/_impl.py:1001-1002, 1068-1069` に既に引数がある) |
| 10 | `b12x/moe/_shared/kernels/w4a16/kernel.py:6799-6808` | `intermediate_rotation` と swiglu_limit の併用を拒否する。trellis では必ず `full_rotation=intermediate_rotation=True` になる (`fused_moe/_impl.py:3413`) | 拒否を外し、#11 を実装する |
| 11 | `b12x/.../w4a16/kernel.py:8819-8903` (`_run_activation` の回転分岐)、`:8394-~8560` (`_run_activation_compact`)、`:8263-~8393` (`_run_activation_intermediate_hadamard`) | `ig = had128(gate)*svh_gate`、`iu = had128(up)*svh_up` の直後に `silu(ig)*iu` を計算しており、clamp が無い | `ig = min(ig, L)`、`iu = clamp(iu, -L, L)` を挿入する (各 4 要素×3 関数)。参照実装 `ref/model.py:847-849` と同じ形 (up は両側、gate は上側だけ) にする。回転を戻した物理座標で作用するので、意味的に正しい |
| 12 | `b12x/moe/fused_moe/planning.py:336-346` と `b12x/.../w4a16/exl3.py:89-94` (trellis のタイル既定値) | K2 で intermediate Hadamard を使うときは (128,128,128,128)、それ以外は (64,256,64,256) | TP2 で一律 K3 / K4 を使うなら fc1_n=128 系のタイルに変え、チューニングを取り直す (§1.3) |
| 13 | `b12x/moe/fused_moe/planning.py:312-313` (任意) | trellis は A16 だけ | lut_e4m3 の一律ビットには W4A8 trellis の下位実装がある (`b12x/moe/_shared/execution.py:869-902`)。A8 (FP8 MMA) を開けば prefill が速くなる余地がある |
| 14 | `vllm/models/deepseek_v4_1/nvidia/model.py:775-860` (`load_weights`) | `.experts.` の重みは `expert_mapping` から `param.weight_loader(..., expert_id=global)` で読む | EXL3 では experts の safetensors を index に含めず、manifest と層別ファイルから読む (Kimi と同じ方式)。IQ2 では checkpoint の `experts.E.w{1,2,3}.weight` を `uint8 [N, K/256, 74]` に置き換える (scale のテンソルは無し) |

### 1.2 SiLU + swiglu_limit=10 の扱い

- **V4.1 の定義** (`ref/model.py:845-850`): up は ±10 に、gate は上側だけ 10 に clamp し、その後 `silu(gate)*up` を計算する。
- **b12x の汎用 epilogue**
  - `_clamp_swiglu_inputs` (`w4a16/kernel.py:7077-7092` と `9030-9045`) が、同じ意味の clamp を `has_swiglu_limit` の const_expr で実装している。
  - スカラー経路 (8927)、x8 ベクトル経路 (`_gated_activation_elem` 8735-8737 経由)、W4A8 系 (`micro.py:5314-5322`、`dynamic.py:1836`、`w4a8_phase1.py:564`) も同じ clamp を持つ。
  - **MXFP4 (現行の V4.1) と IQ2 / Q8_0 は、この epilogue を通る。**
- **swiglu_limit の受け渡し**
  - execution plan は、weight plan の `ActivationSpec.swiglu_limit` を引き継ぐ (`b12x/moe/fused_moe/_preparation.py:244`)。
  - したがって、weight plan に limit を入れれば実行時にも効く。入れていないのが改修 #4 と #7 の箇所である。
- **trellis (EXL3)**
  - 回転付きの epilogue は、FC1 出力に 128 次元の Hadamard を掛け、svh で物理座標に戻してから活性化を計算する (`kernel.py:8829-8838` のコメント)。
  - clamp もこの時点の値に掛ければよい。現状はこれが未実装で、`kernel.py:6805-6808` で明示的に拒否している (改修 #10 と #11)。
  - **実装量は小さいが、カーネルの JIT キャッシュ鍵 (`has_swiglu_limit`, `swiglu_limit` は既に鍵に入っている: 7035-7036) が変わるので、再コンパイルとチューニングが要る。**

### 1.3 TP2 (I_local=1152) のタイル制約

| 形式 | 制約 (ファイル:行) | 1152 | 2304 (EP) |
|---|---|---|---|
| MXFP4 W4A8 (現行) | H%256、I%32 (`execution.py:905-908`)。n%128==64 には特別経路がある | 通る (9×128) | 通る |
| trellis の一律ビット (mcg K3〜、lut_e4m3 で Hadamard 無し) | タイル (64,256,64,256) (`planning.py:340-346`)。trellis は `trellis_t256_proj` 固定 (`fused_moe/trellis.py:453`、`_impl.py:8435`) で、I % fc1_tile_n(256) を要求する (`w4a16/prepare.py:1546-1556`、`kernel.py:1481-1490`) | **通らない** (1152%256=128) | 通る |
| trellis lut_e4m3 K2 + intermediate Hadamard | タイル (128,128,128,128)。H%pre_block(512)、I%post_block(128) (`exl3_schema.py:397-414`)。extent の先頭は 128 境界 (`planning.py:325-331`) | 通る (5120%512=0、1152%128=0、rank1 の先頭 1152%128=0) | 通る |
| trellis の回転付き epilogue | I%128 (`kernel.py:6809-6812`) | 通る | 通る |
| IQ2_XS / IQ2_XXS | H と I を `max(128,256)` で割り切れること (`execution.py:962-975`、`w4a16/iq2_xs.py:140-145`、`modelopt_iq2_xs.py:67-80`) | **通らない** (4.5 ブロック) | 通る (9 ブロック) |
| Q8_0 | `max(128,32)`=128 | 通る | 通る |

- TP2 のまま IQ2 を使うには、rank ごとに 1280 / 1024 の非均等分割が要る。Kimi の EXL3 は層ごとに持ち主を回して非均等を均している (`utils/exl3.py:43-51`, 72-85) ので、前例はある。ただし `moe_config.intermediate_size_per_partition` が rank ごとに違う状態を vLLM 全体に通す必要があり、手間が大きい。EP の方が素直である。

### 1.4 EP (expert parallel) について

- **LIL の SM120 V4.1 では使えない (現状)。** 根拠は次のとおり。
  - `B12xExperts._supports_parallel_config` が `use_ep` を拒否する (`b12x.py:815-824`)。`supports_expert_map()` は False (833-834)。apply で expert_map が来ると例外になる (1118-1119)。
  - MXFP4-CSF (`mxfp4_csf.py:110-117`)、EXL3 (`exl3.py:136-146`)、IQ2 (`modelopt_iq2_xs.py:46-47`) もそれぞれ EP を拒否する。
  - profile は `moe-backend: b12x` を指定する (LI §2.1)。EP にすると b12x のバックエンドが選べず、起動できない。
  - V4.1 の汎用経路 (`deepseek_v41/nvidia/model.py:160, 466`) の EP は MegaMoE (DeepGEMM 系、SM100) 用で、SM120 では選ばれない。
- **配線はできる見込みである。**
  - vLLM の `FusedMoEParallelConfig` は、DP=1 で EP を使うとき `use_all2all_kernels=False` になる (`fused_moe/config.py:1061-1064`)。この場合、入力は複製されたままで、expert_map とその後の TP all-reduce で処理される。
  - b12x 側にも、同じ形の「入力複製型 EP」が用意されている。
    - `b12x/moe/ep_moe/__init__.py:1-12` (W4A16 専用、`_impl.py:193`) と、`fused_moe.bind(route_expert_map, output_expert_map)` (`fused_moe/_impl.py:1001-1002`)。
    - カーネルにも `use_expert_map`、`direct_topk_routes` がある (`w4a16/kernel.py:6616-6617, 7495, 8435-8445`)。
  - trellis、IQ2、Q8_0、MXFP4-A16 はすべて W4A16 なので、どれも対象に入る。
- **EP2 の得失【推測】**
  - 利点: I=2304 のまま使えるので、§1.3 のタイル制約が全部消える。
  - 欠点 1: 負荷の偏り。c1 (B=6、1 層あたり約 35 experts) では、2 枚への振り分けの偏りで MoE 時間が +10〜15% 増える。c16 (1 層あたり約 243 experts) では、偏りはほぼ無視できる。
  - 欠点 2: rank あたりの読み出し量は TP と同じ。
  - DSpark の 128 experts も 64/64 に割り切れる。

---

## 2. exl3-v1 manifest の仕様と exllamav3 出力からの変換

### 2.1 仕様 (`b12x/moe/_shared/exl3_schema.py`)

- **ファイル構成** (11-36 行): `exl3-manifest.json` と、MoE 層ごとの `exl3-layer-<NNNNN>.safetensors`。
- **層ファイルの中身**
  - `codes` は `u8 [num_slots, row_stride]`。1 slot は intermediate の 32 チャネル (`SLOT_CHANNELS=32`, 58 行)。各行に全 experts 分の gate‖up‖down の符号語が expert 番号順に並び、行末は 0 で埋める。
  - 各行列の区画は low plane ‖ high plane の順。一律ビットのときは、slot の 2 つの N16 タイル (FC1) または K16 タイル (FC2) がそのまま 2 つの plane になる。1 plane は `[H/16][16·bits] int16`。
  - `rotations` は `fp16 [num_slots, E, 3, 32]`。intermediate 側の値 (gate svh、up svh、down suh) を入れる。
  - `gate_suh` / `up_suh` / `down_svh` は `fp16 [H]`、または `[E,H]` (`per_expert_input_rotations=true` のとき)。
  - `rates_fc1` / `rates_fc2` は `u8 [num_slots/8, E]` で、`per_expert_pair` のときだけある。
  - `sign_pattern` は `u8 [E]` (0..7) で、intermediate Hadamard のときだけある。
- **manifest のキー**
  - `kind="exl3-manifest"`、`schema="exl3-v1"`、`codebook ∈ {mcg, lut_e4m3, lut_fp16}`。mcg なら `codebook_seed=0xCBAC1FED` が必須 (205-221)。
  - geometry: `num_experts`, `hidden_size`, `intermediate_size`, `slot_channels=32`, `num_slots`, `moe_layer_indices` (266-322)。
  - rates: `{"structure":"uniform","bits":2..6}` か `{"structure":"per_expert_pair","pair_kinds":[P22,P33,P24,P43,P44]}` (325-366)。
  - hadamard: `intermediate_hadamard`、`per_expert_input_rotations`、`pre_block` / `post_block` (369-420)。
  - layout: `row_alignment`、`extent_alignment_slots`、`extent_barriers` (423-453)。
  - layers: `{"<idx>": {"file", "sha256"}}` (456-482)。
- 層ファイルの safetensors metadata (`schema`, `codebook`, `layer`, `num_experts`, `hidden_size`, `intermediate_size`, `slot_channels`) が manifest と一致するかも検査される (`checkpoints/exl3.py:93-109`)。
- **書き出し器**: テスト用の合成チェックポイントの writer `b12x/moe/_shared/kernels/w4a16/exl3_synth.py:276-311` (`write_exl3_checkpoint`) が、そのまま変換器の出力部に流用できる。行の組み立ては `assemble_code_rows` (182-212)、manifest は `_manifest_dict` (215-261)。

### 2.2 一律ビット (uniform rate) の制約

- `checkpoints/exl3.py:1-7, 239-240`: 実行側のアダプタ `trellis_from_exl3` は `uniform` しか受け付けない。`per_expert_pair` は低水準の reader では読めるが、実行経路には乗らない。
- `per_expert_pair` の経路 (`w4a16/exl3.py:113-…`) は、1 rank の extent がちょうど 1 pair (8 slot = 256ch) であることを要求する。これは Kimi の TP10 用の形で、DS (TP2 で 1152ch、EP で 2304ch) では使えない。
  - また、pair の静的な種類は {P24, P33, P43, P44} だけで、**P22 は無い** (`w4a16/kernel.py:6772` の `static_kinds`)。2bit の pair は作れない。
- **ビット数の範囲** (`trellis_codebooks.py:54-64`、`fused_moe/source.py:93-106`)
  - `mcg` は 3〜6bit。**「MCG trellis execution requires at least K3」** (source.py:105-106)。
  - `lut_e4m3` は K2 / K3 / K4。`lut_fp16` は K5 / K6。
  - **したがって 2bit は lut_e4m3 だけで、mcg の最小は 3bit。**
- **層ごとにビットを変える方法**
  - manifest 1 つにつき bits は 1 つだが、MoE method は層ごとに作られ、カーネルもビット数ごとにコンパイルされる (`trellis_bits` は const)。
  - そこで、層ごとに別の manifest (別のディレクトリ) を置き、`load_exl3_manifest(root)` (`utils/exl3.py:26-37`、root ごとにキャッシュ) を層ごとの root で呼ぶよう改修 #4 を変えれば、**層単位の混合は小改修でできる。**
  - expert 単位の混合はできない。
- **容量の計算**【計算】
  - experts は 40 層 × 384 × 3 × 5120 × 2304 = 543.6G 重み。
  - K2: 135.9GB + rotations / suh 約 0.7GB = **136.6GB** (1 GPU あたり 68.3GB)。
  - K3: 204.6GB。
  - 予算 152.7GB に収めるなら、K3 にできるのは約 9 層 (平均 2.24bit)。

### 2.3 exllamav3 の出力から変換できるか

- **タイル内の配置は一致する。** exllamav3 の `tensor_core_perm` (`ext/exllamav3/exllamav3/modules/quant/exl3_lib/quantize.py:43-66`) では、lane t、要素 j の位置は k={2(t%4), +1, +8, +9}、n=t//4 (j≥4 なら +8) である。b12x の `_tile_position_map` (`b12x/testing/reference/trellis_moe.py:59-71`) は `n0 = 2·(lane//8) + ((lane>>2)&1) = lane>>2` で、同じ写像になる。
- **mcg の定数も一致する。** 乗数 `0xCBAC1FED`、lop3 の mask `0x8FFF8FFF`、or `0x3B603B60` が、exllamav3 の `codebook.cuh` と同じ値である (`b12x/_lib/intrinsics.py:6553-6557` のコメントは codebook.cuh の行番号まで引いている)。
- **テンソルの対応** (exllamav3 では 1 Linear が `{key}.trellis int16 [K/16, N/16, 16·bits]`, `.suh fp16 [K]`, `.svh fp16 [N]`, `.mcg`。`ext/exllamav3/exllamav3/modules/quant/exl3.py:28-114`。DS-V4 の expert の key は `experts.{e}.w1/w3/w2`、`architecture/deepseek_v4.py:175-184`)
  - gate (w1): `trellis[:, 2s, :]` と `[:, 2s+1, :]` → slot s の gate の 2 plane。up (w3) も同様。
  - down (w2): `trellis[2s, :, :]` と `[2s+1, :, :]` → slot s の down の 2 plane。
  - `w1.suh` → `gate_suh[e]`、`w3.suh` → `up_suh[e]`、`w2.svh` → `down_svh[e]`。いずれも `per_expert_input_rotations=true`。
  - `w1.svh[32s:32s+32]`、`w3.svh[…]`、`w2.suh[…]` → `rotations[s, e, 0/1/2, :]`。
  - b12x 側の説明「No codeword or scale is numerically converted」(`checkpoints/exl3.py:230-233`) とも合う。**変換はバイトの並べ替えだけで済み、CPU で数十分程度と見込む【推測】。**
- **ただし、既存の出力はそのまま使えない。**
  - 公開済みの V4.1 / 1017 の EXL3 (LibertAIDAI 2.9bpw、coolbho3k / diffbot 3bpw) は、すべて `mul1` codebook で作られている (`research/tmp/r2q/cards/LibertAIDAI_…md:19,23`、`diffbot_…md:177`)。
  - b12x の W4A16 は `mul1` を明示的に拒否する: 「the MUL1 runtime marker is not supported; use the mcg or lut_e4m3 codebook」(`w4a16/prepare.py:1871-1873`)。
  - さらに exllamav3 の出力はテンソルごとにビット数が違う (2.9bpw は 2 / 3 / 4bit の混合)。一律ビットの制約に反する。
- **作り方は 2 通りある。**
  1. exllamav3 v1.6.3 (`ext/exllamav3`、コミット `448f5ea7`) で `mcg` codebook と一律 3bit を指定し、experts だけを量子化して変換する。
     - 3bit は 204.6GB で VRAM に入らない。刈り込み (例 288 experts で約 153GB) か host 段 (§5) との併用が前提になる。
     - TP2 で使うにはタイルの改修 #12 も要る。EP2 なら不要。
  2. **lut_e4m3 の量子化器を自作する** (推奨の本線)。
     - デコード表は公開されている (`b12x/_lib/quant/lut_e4m3.py`。`lut_e4m3_direct_table_cpu()` が 3×65536 個の E4M3 値を返す)。
     - exllamav3 の Viterbi 量子化カーネル (codebook はテンプレート `cb`、`exllamav3_ext/quant/codebook.cuh:115-…`) に、「16bit 窓 → 置換 → E4M3 表」の codebook を 1 種類足す。表は E4M3 なので 64KiB で、共有メモリに載る。
     - intermediate Hadamard (pre 512 / post 128、sign_pattern) を使うなら、量子化の前に重みに同じ変換を掛ける必要がある (定義は `b12x/testing/reference/trellis_moe.py:148-206`)。
     - LIL 自身の量子化器 `trellis-quant` は非公開で、`https://api.github.com/repos/local-inference-lab/trellis-quant` が 404 を返す。

### 2.4 EP にした場合の EXL3

- codes の各行には全 384 experts のバンドルが expert 番号順に並ぶ。一律ビットではバンドルの大きさが固定 (`exl3_schema.py:99-106`) なので、自 rank の 192 experts 分は行ごとに連続したバイト範囲として切り出せる。
- `rotations[:, e0:e1]`、`*_suh[e0:e1]`、`sign_pattern[e0:e1]` も同じように切り出す。そのうえで `Exl3Geometry(num_experts=192)` 相当で `trellis_from_exl3` に渡す (`checkpoints/exl3.py:244-246` の payload 計算は E に比例する)。

---

## 3. Engram の disk モードの実装

### 3.1 データの流れ (1 ステップ)

1. `model_state.prepare_inputs` (`vllm/models/deepseek_v4_1/nvidia/model_state.py:258-262`) が、毎ステップ forward の前に `prepare_disk_engram` を呼ぶ。
2. `prepare_disk_engram` (`nvidia/model.py:487-526`) が GPU でハッシュ (`engram_hash.run_native`) を計算する。続いて層 1 と層 14 の 2 層分の lookup を `engram_native.run_lookups` でまとめて実行する。
3. `run_lookups` (`b12x/sequence/engram/_impl.py:377-427`)
   - 各層の `transaction` に入り、`_stage_ids` で GPU 上の ID を pinned host へ非同期にコピーする。
   - 1 層目は呼び出したスレッドで、2 層目は 3 スレッドのプールで `_read_staged` を同時に実行する。
   - 終わったら、GPU の gather (`lookup_op`) を enqueue する。
4. `_read_staged` (`b12x/sequence/_shared/disk_table.py:354-367`) は、**`self._ids_ready.synchronize()` で CPU を止め、GPU がハッシュを出し終えるまで待つ。** その後、native の `ple_reader_run` で io_uring の読み出しを最後まで待つ。
5. 読んだ行は mapped pinned host のステージング領域 (`cudaHostAllocMapped|WriteCombined`、disk_table.py:80-84) に書かれる。GPU の gather はそこを UVA で直接読み、FP8 と E8M0 を BF16 に復号して `staged_rows [max_tokens, 6144]` に置く。
6. forward の中では、`rows = tensor_model_parallel_all_reduce(staged_rows)` (`common/engram.py:878`) の後に wkv と mix を計算する。

### 3.2 同期読み、ステージング、行キャッシュ

- **同期である。** `b12x/sequence/engram/_disk.py:33-36` の docstring に「Disk lookup is synchronous. The prefetch argument and methods are accepted as no-ops」とある。`prefetch()` と `abort_prefetch()` は空実装である (90-99)。
  - ステップの critical path 上で、GPU の完了待ち → ディスク待ち → forward の enqueue が直列に並ぶ。
- **行キャッシュは無い。**
  - `DiskRowCache` は名前に反して、バッチ 1 回分のステージング領域である。docstring は「Neither file data nor full table allocations are retained」(disk_table.py:132-139)。
  - ファイルは `O_DIRECT` で開く (`b12x/loader/_row_plan.h:113`) ので、OS のページキャッシュも効かない。
  - 効くのは、**バッチ内での 4KiB ブロック単位の重複除去と隣接読みの結合**だけである。断片をオフセット順に radix ソートし、同じブロックはまとめ、隣接するブロックは最大 64KiB (`PLE_READ_MAX`) まで 1 回の read にする (`_row_plan.h:6-7, 238-312`)。
- **io_uring**
  - `io_uring_queue_init(depth, flags=0)` で初期化する。SQPOLL は使わない。
  - 固定バッファ `depth × 64KiB` を `io_uring_register_buffers` で登録し、ファイルも登録する (`IOSQE_FIXED_FILE`)。場所は `b12x/loader/_ple_reader.c:144-170, 178-199`。
  - キューの深さは `DiskTable(queue_depth=128)` (`_disk.py:43`)。リーダは 1 層・1 rank ごとに 1 つ。
  - 完了した分から補充していく (`ple_uring_run` 270-305)。
  - バックエンドは `B12X_DISK_BACKEND ∈ {io_uring, gds}` で、既定は io_uring (disk_table.py:172-174)。

### 3.3 TP での分割

- 行は TP で ceil 分割する。rank r が `[r·⌈N/TP⌉, (r+1)·⌈N/TP⌉)` を持つ (`common/engram.py:419-421`)。
- リーダは自分の範囲外の ID を 0 で埋めて飛ばす (`_row_plan.h:243-250`)。2 rank の結果を all-reduce で足し合わせる (engram.py:878)。
- **rank あたりの IO は半分になるが、2 つのプロセスが同じ 1 本の NVMe を同時に読む。** 合計の IOPS 上限 (約 23 万) は変わらない。
- all-reduce の量は 1 トークン 12KiB で、4096 トークンのチャンクでも 1 層 50MB である。PCIe では 2〜3ms で済む。

### 3.4 必要な RAM と、コンテナの要件 (`max_num_batched_tokens=4096` のとき)

| 項目 | 大きさ | 根拠 |
|---|---|---|
| 行ステージング (weight) | 4096×24×256B = 25.2MB / 層 / rank → 合計 101MB (mapped pinned) | `_disk.py:68-78` (`max_lookups = max_tokens×24`) |
| scale ステージング | 0.8MB / 層 / rank | 同上 (`scale_row_bytes=8`) |
| io_uring 固定バッファ | 128×64KiB = 8MiB / リーダ。4 リーダで 32MiB (locked) | `_ple_reader.c:136-137, 164` |
| 断片と job の配列 | 数十 MB / リーダ | `_row_plan.h:74-90` |
| scale の RAM 常駐 (任意) | ⌈384,006,168/2⌉×8B = 1.54GB / 層 / rank → **合計 6.14GB** | `_disk.py:81-88`、`config/engram.py:76-86` |
| GPU 側の `staged_rows` | 4096×6144×2B = 50MB / 層 / rank | `common/engram.py` (`staged_rows`) |

- **seccomp と memlock を実測した** (`research/` の外、scratchpad の `uring_probe.py` を `--network none` で実行)。
  - 既定の seccomp では `io_uring_setup: -1 EPERM`、memlock は `(8388608, 8388608)`。
  - `--security-opt seccomp=unconfined --ulimit memlock=-1` では `io_uring_setup: 3 ok`、memlock は `(-1, -1)`。
  - ホスト側は `/proc/sys/kernel/io_uring_disabled = 0` で、io_uring は有効。docker は 29.9.1。
  - 1 プロセス 2 リーダで固定バッファが 16MiB になり、既定の 8MiB を超える。**memlock の解除も必須**である。
  - LIL の launcher も compose に両方を加えている (`/opt/lil/runtime/generate.py:79-81`)。
  - seccomp を全部外したくなければ、`io_uring_setup`、`io_uring_enter`、`io_uring_register` だけを許すカスタムプロファイルでもよい。

### 3.5 先読みと行キャッシュを足す改修点

| 優先 | 改修 | 場所 | 見込み |
|---|---|---|---|
| 1 | **アプリ側の行キャッシュ** (FP8 の 256B 行、CLOCK / LFU、頻度表で初期化) | native: `_row_plan.h` の `ple_plan` (238-312) で、断片を作る前にハッシュ表を引く。ヒットはステージングへ memcpy、ミスだけ io_uring に回し、`ple_scatter` (`_ple_reader.c`) の完了時にキャッシュへ入れる。Python: `DiskRowCache.__init__` (disk_table.py:141-229) でキャッシュ容量の引数を受ける | rank ごとに担当行が分かれているので、キャッシュも自然に分割される。60GB (rank あたり 30GB、約 1.2 億行) でヒット率 65% なら、prefill の IOPS 上限は 6.7k → 約 19k tok/s (PLAN §2.4 D3) |
| 2 | **非同期の先読み (prefill)** | `DiskTable.prefetch()` の空実装 (`_disk.py:90-99`) を実装する。`DiskRowCache` のステージングを 2 面にし (disk_table.py:204-220)、`ple_uring_run` を submit と wait に分ける (`_ple_reader.c:270-305`)。vLLM 側: スケジューラがチャンク N を計算している間に、チャンク N+1 のハッシュを CPU で計算して投入する (ハッシュは入力トークンだけで決まる。`b12x/sequence/engram/geometry.py`、`reference.py` を移植)。呼び出しは `model_state.py:258-262` と `nvidia/model.py:487-526` を「投入」と「完了」に分ける | 現状は、4096 トークンのチャンクで読み出し (約 13 万 IO で 0.57s) と計算 (約 0.4s) が直列になり、約 4k tok/s で頭打ちになる。重ねれば IOPS の上限 (約 7k、キャッシュ併用で約 10k) まで上がる【計算】 |
| 3 | decode での隠蔽 | 層 14 の行を別の transaction にし、GPU 側で mapped flag を待つ (`cuStreamWaitValue32` など)。CUDA graph に入れる必要があり難しい | c16 で 1 ステップ約 11〜13ms の待ちを、層 1 分 (約半分) まで縮める。キャッシュだけでも約 4.6ms に減る【推測】 |
| 4 | 4KiB 整列 | Engram の 2 テンソルだけを、データ開始位置が 4096 の倍数になる safetensors (ヘッダの JSON を空白で詰める) に書き直す。`get_file_tensor_source` がそのオフセットを返す (`common/engram.py:524-561`)。`disk_resident_scales=true` と組み合わせると、1 行が必ず 1 IO になる | 現状はデータ開始が 664B で、16 行に 1 行が 8KiB 読みになる (PLAN §1.3)。IO が 6.25% 減る。再書き出しは 203GB で、NVMe の空き容量に注意 |
| 5 | io_uring の調整 | `IORING_SETUP_SQPOLL` / `IOPOLL` (nvme poll queue が必要)、QD 64〜256 の測定 (PLAN X3) | 遅延が数十 µs 縮む【推測】 |

---

## 4. MXFP4-CSF (ロスレス圧縮の experts)

- **縮むのは scale だけである。** LIL の CSF リポジトリの README (`research/tmp/r2l/csf/README.md`) によれば:
  - 「CSF stores the routed-expert block-scale planes in a compressed form. Nothing else changes: FP4 weight nibbles … are all byte-identical.」
  - 「Weight files: 510.30 GB in the source, 495.63 GB here (14.67 GB saved).」
  - 「Compressed scales: 46,080 matrices … 16.99 GB -> 2.31 GB (13.6%).」
- 自分で `ref/headers.json` を集計した値も一致した。主層 experts の内訳は weight 271.791GB、scale 16.987GB。DSpark は weight 6.795GB、scale 0.425GB。HF API で得た `tensors/` 48 本の合計は 495.63GB。
- **CSF にしても experts は 271.8 + 2.3 = 約 274.1GB で、192GB 未満にはならない。** scale を 0 にできたとしても 271.8GB が残る。量子化は避けられない。
  - CSF の効果は GPU 上の 14.7GB (1 GPU あたり約 7.3GB) で、KV が増える程度である (`mxfp4_csf.py:3` と profile の説明「they stay compressed in GPU memory」)。
- **変換ツールは使えない。**
  - 作成には `trellis_quant.lossless_scale_checkpoint` (README の Provenance と Verify 節) を使うが、イメージには無い (`import trellis_quant` が ModuleNotFoundError)。GitHub の `local-inference-lab/trellis-quant` も 404 で非公開。
  - 1017 用の CSF は公開されていない (LI §2.1)。
  - 読み出し側 (`vllm/model_executor/model_loader/mxfp4_csf_loader.py`、`csf_utils.py`、`b12x/_lib/quant/mxfp4_csf.py`) はイメージにある。ここから符号器を逆算することはできる。ただ、得るものが 1 GPU あたり 7GB しかないので、優先度は低い。
  - CSF のリポジトリは LIL-1.0 ライセンスで、再配布や変換物にも制約がある (README の License 節)。
- 制約: CSF の MoE は EP / DP 非対応 (`mxfp4_csf.py:110-117`)。PP1 で ubatching 無しが条件 (84-89)。

---

## 5. LIL の expert residency (host 側に experts を置く仕組み)

### 5.1 現状

- このイメージには入っていない。`vllm` と `b12x` を `residency` で grep しても、MoE には該当が無い。
- 中身は LIL vllm **PR #888「Serve DeepSeek-V4.1-Flash on one GB300 with b12x expert residency」** (`research/tmp/r2l/web/pr888.json`) で、2026-09-24 に closed、`merged: False`。前提の b12x **#425 (SM103 カーネル)** と **#426 (HBM / Grace の 2 層 MoE)** もどちらも closed で、未マージ。
- 方式の要点 (PR 本文)
  - よく使われる hot experts を HBM に、残りを coherent な Grace メモリに置く。GPU は NVLink-C2C で直接読み、1 回の operator 呼び出しで両方の層を扱う。
  - SM103 の tcgen05 / TMEM カーネル。
  - 「Grace-served route costs about 60 µs (… about 300 GB/s) against about 15 µs from HBM」。
  - ホットな集合の偏りについて: 「the top 305 experts per layer take about 97.6% of routes」。
  - 実測: 1 GPU、DSpark 無しで C1 59.7、C8 136.3 tok/s。後続ブランチでは C1 135 / C16 520。

### 5.2 x86 / PCIe で使えるようにするための改修量【推測】

1. **SM120 カーネル**
   - b12x の SM120 W4A8 / W4A16 MoE に、「hot スラブ (device) と cold スラブ (mapped host) の 2 つのベースと、expert ごとのスロット写像」を足す。b12x #426 の SM103 版を SM120 の mma.sync 系に移植することになる。
   - 代わりに、CUDA VMM を使う手もある。`cuMemCreate` (location=HOST) と device メモリを 1 つの VA 範囲に `cuMemMap` で並べれば、既存カーネルは無改修で済む。
     - その場合、prepare 後の配置が expert 単位で 2MiB に揃っている必要がある。TP2 の FP4 で w13 は 5.9MB / expert なので、6MiB に詰め直すと約 7% の無駄になる。
2. **vLLM 側**: PR #888 と同じ部品が要る。
   - experts を host で段取りする (`Mxfp4MoEMethod.create_weights` の `requires_device_loading=False`)。
   - `B12xExperts` で、device 上の準備をせず宣言だけにする。
   - モデル全体の配置コントローラと予算の検査。
   - 固定の `--kv-cache-memory-bytes`。
3. 規模は 4〜8 週。

### 5.3 x86 での効き方【計算】

- PCIe Gen5 x8 は約 20GB/s / GPU (GLM で実測) で、C2C の約 1/15 である。
- cold の expert を 1 回読むのに、TP2 の FP4 (9.4MB / rank) で約 0.47ms かかる。
- c1 (B=6、1 層あたり約 35 experts) で cold の割合が 5% なら、40 層 × 1.7 回 × 0.47ms で 1 ステップあたり約 31ms 増え、ステップ時間が約 2 倍になる。
- FP4 のまま VRAM に置けるのは 8,122 / 15,360 experts (53%) なので、cold に当たる割合は 15〜25% と見込む (PLAN 案 C)。実用にならない。
- **使い道は「2bit 化した上で予算から少しはみ出す分 (1〜2% のルート)」の退避先に限る。** 本線の K2 一律 (136.6GB) は予算に収まるので、residency は不要である。

---

## 6. 推奨する経路、手順、工数、性能

### 6.1 本線: TP2 + trellis lut_e4m3 K2 (+intermediate Hadamard)、全 experts を VRAM、Engram disk + 行キャッシュ

**手順**

1. 量子化器 (GPU 1 枚、GLM を止めた後)
   - exllamav3 (`ext/exllamav3`) に lut_e4m3 codebook を追加する (§2.3 の作り方 2)。
   - 1017 の MXFP4 experts を BF16 に展開し、校正データ (PLAN X6 の層ストリーミング・ハーネスで得る活性の H 行列) で K2 一律に量子化する。
   - intermediate Hadamard と sign_pattern は b12x の参照実装 (`testing/reference/trellis_moe.py:148-206`) に合わせる。
2. 変換器 (CPU)
   - exllamav3 の出力から `exl3-manifest.json` と `exl3-layer-000{00..39}.safetensors` を作る。
   - 設定値: `codebook=lut_e4m3`、`rates={uniform, bits:2}`、`hadamard={intermediate_hadamard:true, pre_block:512, post_block:128, per_expert_input_rotations:true}`、`layout={row_alignment:4096, extent_alignment_slots:4, extent_barriers:[36]}`。
   - 書き出しは `exl3_synth.write_exl3_checkpoint` の書き出し部を流用する。
3. サービング用のディレクトリ
   - 1017 の非 experts テンソル (FP8 のまま。Engram の 47/48 番は symlink または 4KiB に揃えた版) と、experts を除いた `model.safetensors.index.json` を置く。
   - config の `quantization_config` は `{"quant_method":"deepseek_v41_lowbit", "exl3":{"manifest":"exl3-manifest.json"}, …}` にする。
4. vLLM / b12x のパッチ: §1.1 の #1〜#5、#8、#10、#11、#14。TP2 + K2 + Hadamard ならタイル改修 #12 は不要。
   - 適用はイメージを書き換えず、glm53v と同じく `-v patch.py:/opt/venv/…:ro` の bind mount で行う。
5. 起動: `ds41-flash` profile に TP=2、`ENGRAM_TABLE_MEMORY=disk`、`CHECKPOINT=original`、`MAX_NUM_SEQS=16`。docker には `--security-opt seccomp=unconfined --ulimit memlock=-1` を付ける。
6. Engram の改修: §3.5 の 1 (行キャッシュ) と 4 (整列) から始め、2 (prefill の先読み) を足す。

**工数【推測】**

| 作業 | 期間 |
|---|---|
| 量子化器 | 1.5〜3 週 |
| 変換器 | 2〜3 日 |
| vLLM / b12x の配線と clamp | 1 週 |
| Engram のキャッシュと先読み | 1〜2 週 |
| 合計 | **4〜6 週** |

量子化の実行は、1 版あたり十数時間 (GPU 1 枚) と見込む。

**性能【推測、PLAN §1.4 の式に A16 trellis の係数 kq=1.25 と、Engram の同期待ちを加えた値】**

| 指標 | 見込み | 計算の内訳 |
|---|---|---|
| c1 | 110〜160 tok/s (式で約 170) | 1 GPU あたり 6.4GB の expert 読み + 5.1GB の密層 → 9.9ms。固定費 9ms と Engram 1.5ms を足して 20.4ms。τ=3.5 |
| c16 | 450〜700 tok/s (式で約 640。行キャッシュ込みで約 720) | 1 GPU あたり 45GB 読み → 43ms。固定費 9ms + 計算 4.5ms + Engram 約 13ms (キャッシュで約 4.6ms) |
| prefill (現状の同期読み) | 約 4k tok/s | §3.5 の 2 |
| prefill (先読み + キャッシュ) | 7〜10k tok/s | 同上 |
| 文脈 | 1M / 本 (KV プール約 4M トークン) | PLAN §2.2 |

GLM-5.3 の現行値 (c1 224 / c16 384) と比べると、c16 は上回り、c1 は下回る見込みである。

### 6.2 次善 / 比較用: EP2 + IQ2_XS / IQ2_XXS の層別混合

- 容量: IQ2_XXS 2.0625bpw で 140.2GB、IQ2_XS 2.3125bpw で 157.1GB【計算】。予算 152.7GB に収めるなら、29 層を XS、11 層を XXS にする。
- **量子化器は ggml に既にある。** MXFP4 → BF16 → `ggml_quantize_chunk` (imatrix 付き) で作れる。
  - 既存の 1017 GGUF には imatrix が付いていない (`cards/mondk_…md:23`、`mrfakename_…md:23`)。imatrix は PLAN X6 のハーネスで自作する。
  - 格納形式は ModelOpt の mixed で `{"quant_algo":"IQ2_XS","group_size":256,"block_payload_bytes":74,"packing":"ggml"}` (`modelopt.py:1555-1572`)。GGUF のブロックをそのまま `uint8 [N, K/256, 74]` にする。
- 改修: §1.1 の #1〜#3、#6、#7、#9 (EP)、#14。カーネルの clamp パッチは不要。
- 工数: 2〜3 週。品質は trellis K2 より劣る見込み (4bit の重みを再量子化するので誤差が重なる)。

### 6.3 採らない案

- mcg K3 の一律ビット (exllamav3 でそのまま作れる): 204.6GB で入らない。刈り込みか residency の併用になる。
- CSF: 縮むのは 15GB だけ。
- x86 への residency の移植: 費用に見合わない (§5)。

---

## 7. 未確認事項

1. lut_e4m3 K2 の品質 (KL、タスク評価)。PLAN X7 で、EXL3 mcg K3 / mul1 2〜3bit、IQ2 と並べて比べる。
2. b12x の W4A16 trellis カーネルが、DS の形状 (E=384, H=5120, I=1152, top-6) で JIT コンパイルとチューニングを通るか (GPU が必要)。Kimi 以外での実績は不明。
3. exllamav3 mcg の trellis が b12x の mcg デコードとビット単位で一致するか。タイル配置と定数の一致は確認した。plane 内のワード順 (`_pack_native_tiles` の pair flip) は、exllamav3 の `pack_trellis` CUDA 実装と照合していない。1 expert を両方で復元して比べる (GPU 1 枚で数分)。
4. EP2 の配線後に、vLLM の TP all-reduce と b12x の `output_expert_map` で出力が正しく合算されるか。DSpark (128E) を EP で動かしても問題が無いか。
5. 2 つのプロセスから同じ NVMe へ io_uring で同時に読んだときの実 IOPS (PLAN X3 の拡張)。
6. vLLM の async scheduling と同期 Engram の組み合わせで、実際に何 ms が critical path に出るか。
