# DeepSeek-V4.1-Flash-1017 を 2x RTX PRO 6000 Max-Q で動かすための決定事項 (第 2 ラウンド後)

作成日: 2026-10-18
入力: `PLAN.md` (§0, §2.1, §2.3, §2.4, §4)、`r2_ds41rt.md`、`r2_lil_lowbit.md`、`r2_quant_tools.md`、`r2_engram_hit.md`
凡例: 【実測】はこの調査で手元で確かめた値、【資料】は出典付きの値、【計算】は式から出した値、【推測】は根拠が間接的な値。

---

## 0. 結論

1. **今日中に出力を見る最短経路は「T1: 無量子化の層ストリーミング参照ハーネス」である。**
   - 既存の `harness/stream_ref.py` を、手元にある LIL イメージ (`ghcr.io/local-inference-lab/vllm:release-10152026`) で動かす。追加のダウンロードは要らない。
   - 重みは公式の FP8/FP4 のままで、Engram も FP8 のまま NVMe から引く。速度は 1 tok/s 前後と遅い【推測】。ただし 1017 本来の品質で出力が出るので、後で量子化版を評価するときの教師にもなる。
   - GLM を長めに止められるなら、同じ日のうちに **T2 (ds41rt + 自作 K2)** の初回起動まで進められる。T1 と T2 の内容は §2 に書いた。
2. **GLM 置き換え水準を狙う本命は「E3: ds41rt の 0-spark 構成 + 1017 から自作した EXL3 K2 MCG」である。** 品質は 3 段階で上げる (§3)。
   - 当環境での速度の見込みは次のとおり【推測、r2_ds41rt §10】。
     - c1: code 180〜210、加重平均 115〜135 tok/s
     - c16 合計: code 1,100〜1,350、topic 600〜720、混在 330〜380 tok/s
     - 文脈は 1M まで。Engram は FP8 のまま
   - **c16 ≥384 と c1 ≥150 を満たせるかは、負荷の内容で決まる。** code 系なら両方満たす。混在負荷の c16 と加重平均の c1 は境界付近になる。したがって GLM と同じ負荷 (要求ログの再生) で測り直して判定する。
   - 最大のリスクは品質である。0909 の一律 2.0bpw (校正なし) は KL 0.36、HumanEval は 95.7→91.5 だった【資料、r2_quant_tools §5】。
3. **Engram の懸念は X2 でほぼ解消した。** ds41rt は Engram を mmap とページキャッシュで扱うが、それでも NVMe は律速しない見込みである。
   - X2 のシミュレーションでは、4KiB ページの LRU 64GB で全体のヒット率が 91.3%、prefill の IO が 1.11/token だった。230k IOPS なら prefill の上限は約 20 万 tok/s になる【実測 + 計算、r2_engram_hit §2.2, §3.2】。
   - 行キャッシュの改修 (D3) は、ds41rt で gather の待ちを実測して問題が出たときだけ行う。
4. **重みのダウンロードは完了している。ディスクの空きは見込みより多い。**
   - 48 shard、合計 510,311,114,111B で、最終書き込みは 01:21。`.incomplete` は無い【実測: `ls`、`du -sb`】。
   - 空きは 560.9GB (522GiB) で、課題文の「約 490GB」より 70GB 多い【実測: `df -B1 /`】。
5. **ユーザーの判断が要る点は §6 にまとめた。** 主なものは、glm53v を止める時間帯、2.0bpw の品質の合格線、ds41rt イメージ (圧縮 10.5GB) の取得、API の差の許容である。

---

## 1. 第 2 ラウンドで分かったことと、各案の位置づけ

### 1.1 4 つの調査結果

| 調査 | 判定 | この決定で使う事実 |
|---|---|---|
| ds41rt (r2_ds41rt) | 条件付きで可 | 1017 は ds41rt のローダ契約を満たす。config は全項目一致、96,085 テンソルも全一致【実測、§2.2】。0-spark の公表値は C1 code 225.20、c16 混在 390【資料、§5.2】。experts をホストに退避する機能は無く、2.0bpw がほぼ上限 (RTX0 90.94GiB / RTX1 91.53GiB)【資料、§2.1】。`run.sh:468` が `--device=/dev/infiniband` を要求する【実測】 |
| LIL lowbit (r2_lil_lowbit) | 条件付きで可 | 本線 (TP2 + lut_e4m3 K2) は 4〜6 週かかる。lut_e4m3 の量子化器が非公開で、自作が要る。b12x の trellis は clamp を拒否し (`w4a16/kernel.py:6805-6808`)、mul1 も拒否する (`prepare.py:1871-1873`)【資料、§0, §6.1】 |
| 量子化ツール (r2_quant_tools) | 条件付きで可 | 1017 の既製量子化物で、experts が 150GB 以下かつ SM120 で読めるものは無い【実測: HF 再スキャン 2026-10-17 18:52Z】。使える量子化器は sfxnz (MIT)、sudoingX (Apache-2.0)、exllamav3 PR #449 の 3 系統。1 shard には 1 層しか入っていない【実測】 |
| Engram ヒット率 (r2_engram_hit) | 可 | 行 LRU 32GB で prefill 98.2% / decode 87.6%。IO/token は prefill 0.58、decode 5.97。page LRU 64GB でも IO は 1.11 / 7.25【実測】。公式の TP 分割では miss IO の 75% が rank 1 に集まる【実測】 |

### 1.2 各案の位置づけ (PLAN §2.1 を更新)

| 案 | 中身 | 役割 | 理由 |
|---|---|---|---|
| **T1** | 層ストリーミング参照ハーネス (無量子化) | **今日の動作確認と、品質の教師** | 追加のダウンロードが要らない。品質は 1017 そのもの |
| **E3** | ds41rt 0-spark + 自作 K2 MCG | **本命** | 既製のエンジンで、API、ツール、画像、DSpark、1M 文脈、Engram FP8 が揃っている。c1 の見込みが 150 を超えるのはこの案だけ |
| A | vLLM nightly + vllm-exl3 (AGPL) + expert 別 mixed-K + Engram D3 | E3 の予備 | 1.5〜2.5 週かかる。見込みは c1 100〜140 で、150 に届かない |
| A' / LIL lowbit | LIL b12x + lut_e4m3 K2 | 長期の選択肢 | 4〜6 週かかり、量子化器を一から作る必要がある |
| E2 | LibertAIDAI 2.9bpw + host experts | 任意 (品質の比較用) | 211GB のダウンロードが要る。ds41rt では動かない (host 退避が無い) |
| B (REAP) | 刈り込み | 保留 | 1017 版が無い。192E で PPL +14.7%。vLLM の expert 数テーブルの制約もある |
| GGUF | ik_llama.cpp ほか | 不採用 | SM120 で NaN になる不具合 (#2519) がある。中間ファイルが約 508GB でディスクに入らない |

---

## 2. 課題 1: 今日中に 1017 を 2 GPU で推論して出力を見る

### 2.1 経路の比較

| 経路 | 品質 | 速度 | 今日できるか | 追加の取得 | GPU の専有 |
|---|---|---|---|---|---|
| **T1 参照ハーネス** (`harness/stream_ref.py`) | **1017 そのまま** (FP8/FP4、Engram FP8) | decode 0.5〜2 tok/s、短い prompt の prefill は数十秒〜2 分【推測】 | **可** (GLM を止めてから 1.5〜3 時間) | 無し (LIL イメージは手元にある) | 1 プロセスに 1 枚。2 枚で 2 プロセスを並列に回せる |
| T2 ds41rt + 自作 K2 | 2.0bpw の一律量子化 (KL 0.36 級) | c1 100 tok/s 超 | 条件付きで可 (GLM の停止が合計 8〜12 時間) | ds41rt v15 イメージ 10.5GB (圧縮)。量子化イメージは LIL からの派生で作れれば不要 | 2 枚 (TP2) |
| E2 LibertAIDAI 2.9bpw | 良い (PPL +1.8%) | c1 50〜100 | 不可 (211GB の取得に 1Gbps で 30 分以上かかり、エンジンの移植も要る) | 211GB | 2 枚 |

**判断**: T1 を先に実行する。
- 「出力を見る」と「品質を知る」を同時に満たせるのは T1 だけである。
- T1 の教師出力が無いと、T2 (2.0bpw) の劣化を測れない。ds41rt の API には logprobs が無い (r2_ds41rt §7)。
- 時間に余裕があれば、同じ日に T2 へ進む。

### 2.2 T1 の手順

**段 0 (CPU のみ。GLM は動かしたまま、今すぐできる)**

1. X1: sha256 を照合する。期待値は HF のダウンロード metadata の 2 行目にある。shard 47 の値 `76a05106…` は LibertAIDAI のカードと一致する【実測】。
   ```zsh
   mkdir -p work/x1
   bash -c 'cd $MODELS/DeepSeek-V4.1-Flash-1017/.cache/huggingface/download && for f in model-*.safetensors.metadata; do printf "%s  %s\n" "$(sed -n 2p "$f")" "${f%.metadata}"; done | sort -k2' > work/x1/expected.txt
   bash -c 'cd $MODELS/DeepSeek-V4.1-Flash-1017 && ls model-*.safetensors | xargs -P4 -n1 sha256sum | sort -k2' > work/x1/actual.txt
   diff work/x1/expected.txt work/x1/actual.txt && echo ALL_OK
   ```
   - 全体で 510GB を読むので 5〜10 分かかる【推測】。
2. X3: NVMe の IOPS を測る。ダウンロードの書き込みが終わったので、単独の値が取れる。`research/tmp/plan/iops.py` を shard 47 に対して、QD 32〜256、1 プロセスと 2 プロセスで実行する。

**段 1 (ユーザーが GLM の停止を許可してから)**

3. GLM を止めて、資源が空いたか確かめる。glm53v は `AutoRemove=false`、restart policy は `no` なので、後で `sudo docker start glm53v` で戻せる【実測: `docker inspect`】。再起動には 6〜9 分かかる (PLAN §3)。
   ```zsh
   sudo docker stop glm53v
   nvidia-smi --query-gpu=index,memory.used --format=csv
   free -g; df -h /dev/shm
   ```
   - glm53v は `--ipc host` で動いている。LMCache の 49GB が `/dev/shm` にファイルとして残っていないか確かめる。残っている場合に消すかどうかは、ユーザーが決める (§6)。
4. X5: 参照カーネルの単体試験を行う。所要は 10〜20 分。
   ```zsh
   sudo docker run --rm --gpus '"device=0"' --ipc host --network none \
     -v $DSV41/ref:/ref:ro -v $DSV41/tools:/tools:ro \
     --entrypoint /opt/venv/bin/python3 ghcr.io/local-inference-lab/vllm:release-10152026 /tools/x5_kernels.py
   ```
   - `act_quant`、`fp8_gemm`、`fp4_gemm` が通れば次へ進む。
   - 通らない場合は、ハーネス側を torch の実装に差し替える (dequant と BF16 の matmul)。作業は約 1 時間【推測】。`sparse_attn` は、ハーネスで既に torch 版に置き換えてある (`stream_ref.py` の `sparse_attn_torch`)。
5. T1 を実行する。最初は GPU0 で、短い出力を見る。
   ```zsh
   mkdir -p work/t1
   sudo docker run --rm --gpus '"device=0"' --ipc host --network none \
     -v $DSV41/ref:/ref:ro \
     -v $MODELS/DeepSeek-V4.1-Flash-1017:/model:ro \
     -v $DSV41/harness:/h:ro -v $DSV41/work/t1:/out \
     -e DSV41_REF=/ref -e DSV41_CKPT=/model \
     --entrypoint /opt/venv/bin/python3 ghcr.io/local-inference-lab/vllm:release-10152026 \
     /h/stream_ref.py --prompt "日本の首都はどこですか。一文で答えてください。" \
       --prompt "Write a Python function that returns the n-th Fibonacci number." \
       --max-new 64 --expert-budget-gb 60 --out /out/first.json
   ```
   - 出力: 生成文、最初の位置の top-5、prefill と decode の秒数、expert と Engram の統計 (`[experts]` と `[engram]`)。
   - 速度の見込み【推測】: decode は 1 トークンで 40 層 × 6 experts × 18.8MB = 最大 4.5GB を読む。expert キャッシュ 60GB (全体の約 21%) と、ページキャッシュ (GLM を止めた後の空き RAM 約 100GB) を併用して、0.5〜2 tok/s になる。
   - 失敗しやすい箇所【推測】
     - `wo_a` の扱い。1017 で追加された FP8 の再量子化は、`ref/model.py:787, 1073` にある。
     - DSpark 層 (`mtp.*`) の読み込み。
     - KV の `max_seq_len`。
     - ハーネスは未実行のコードなので、ここで 1〜2 時間の修正を見込む。
6. 2 枚目の GPU を使う。`--gpus '"device=1"'` で 2 つ目のプロセスを起動し、評価用の prompt を半分ずつ受け持たせる。ページキャッシュは 2 プロセスで共有される。
   - T1 は TP2 ではない。「2 GPU を使って推論する」のは、厳密には T2 から始まる。

**段 2 (T1 の成果物。T2 の評価に使う)**

7. 教師の出力を保存する。評価セットの候補は次のとおり。
   - GLM の要求ログから 50 件 (エージェント型、ツール付き)
   - 日本語 QA 30 件
   - コード 30 件
   - 一般 30 件
   - それぞれ、teacher-forced で top-k logits (k=64 程度) を保存する。数十 MB で済む。
   - `stream_ref.py` には今のところ「1 回の forward で全位置の logits を保存する」機能が無い。`greedy()` を teacher-forced に拡張する改修が約 1 時間要る【推測】。
8. routing の頻度を集める。`STORE.counts` (層 × expert) を、上の評価セットと要求ログからの 5〜10 万トークンで記録する。これは 1017 と自分のトラフィックでの routing の偏りの初めての実測になる (§4 M5)。

### 2.3 T2 (同じ日に進む場合。GLM の停止を延ばす必要がある)

r2_ds41rt §11 の手順を、当機向けに次のように具体化する。

1. **量子化イメージ** (CPU のみ。今すぐ作れる)
   - 第 1 案: diffbot の `recipe/docker_build.sh` を使う。ただし vLLM の day-0 image の取得が要る。
   - 第 2 案 (取得なし): 手元の LIL イメージ (torch 2.14、nvcc 13.4、`local_images.md` §3) に exllamav3 `5be88657` を `TORCH_CUDA_ARCH_LIST=12.0a` で build し、`docker commit` する。commit `5be88657` は `ext/exllamav3` の履歴にある【実測: `git cat-file`】。
   - 第 2 案は、`--no-deps` で入れたときに exllamav3 が要求する依存が欠けるおそれがある【推測】。欠けた場合は小さな pip 取得で補う。
2. **量子化** (GPU 2 枚、GLM の停止中)
   - コマンド: `MODELS_DIR=$MODELS SRC=DeepSeek-V4.1-Flash-1017 DST=DeepSeek-V4.1-Flash-1017-EXL3-K2-mcg-v0 IMAGE=<量子化イメージ> recipe/quantize/quantize.sh`
   - 事前の準備
     - スクリプトは docker を sudo なしで呼ぶので、sudo で呼ぶように直す。
     - `third_party/sfxnz-recipe` を `460ac89c` で用意する。
   - 前提の確認
     - shard の割り当て (MoE 層 N は shard N+3) は 1017 でも成り立つ【実測、r2_ds41rt §9.2】。
     - hardlink は、同じファイルシステム `/` の上なので作れる【実測: `df`】。
   - 所要は 2 GPU で 1.5〜4 時間【推測。K2 は約 3 GPU 時間、r2_quant_tools §1.2】。
   - 出力の `config.json` で、`codebook=mcg` と `mtp_experts=source` を確かめる。
3. **ds41rt を起動する**
   - `sudo docker pull ghcr.io/tpurtell/ds41rt-coordinator:v15` (圧縮 10.5GB。ユーザーの許可が要る)
   - `hf/hub/models--local--DeepSeek-V4.1-Flash-1017-EXL3-K2-mcg-v0/snapshots/<40 桁の hex>/` に hardlink で配置する。
   - `local.config` を書く。値は r2_ds41rt §11-6 のとおりで、`SPARK_COUNT=0`、`RTX_EXPERT_LAYERS=40`、`HOST_CACHE_BYTES=4GiB` などである。
   - `run.sh` を `run-local.sh` に複製し、468 行目から `--device=/dev/infiniband` を外す。
   - `--dry-run` で確かめてから本番を起動する。
4. **最初の確認**
   - `scripts/api-smoke.sh` を実行する。
   - T1 と同じ prompt で出力を比べる。
   - 起動直後の VRAM が 2 枚とも約 91GiB になっていることを確かめる。

**今日の時間割の目安 (GLM の停止が 1 回の場合)【推測】**

| 時間 | 作業 |
|---|---|
| 0:00〜0:30 | X5 と、GLM 停止後の資源の確認 |
| 0:30〜3:00 | T1 の初回出力と修正、教師の保存を開始 |
| 3:00〜7:00 | K2 v0 の量子化 (2 GPU)。この間 T1 は止まる |
| 7:00〜9:00 | ds41rt の起動、smoke、T1 との比較 |
| 9:00〜 | GLM に戻す (`sudo docker start glm53v`) か、そのまま評価を続ける |

---

## 3. 課題 2: GLM 置き換え水準を狙う本命 (E3) の工程

### 3.1 目標と判定

| 指標 | 目標 | E3 の見込み【推測、r2_ds41rt §10】 | 判定 |
|---|---|---|---|
| c16 合計 | ≥384 tok/s | code 1,100〜1,350 / topic 600〜720 / 混在 330〜380 | 負荷次第。混在は境界付近 |
| c1 | ≥150 tok/s | code 180〜210 / 加重平均 115〜135 | 負荷次第 |
| 長文脈 | 1M | 1M まで。256K 保持での decode (加重) は公表値 121 | 可 |
| Engram | FP8 を維持 | FP8 の公式表を mmap で使う (raw 経路では FP8 しか許されない、`v41_exl3.rs:219-339`) | 可 |
| 画像、ツール、OpenAI 互換 | GLM と同等 | 画像 16 枚、tools、strict、JSON Schema。`/v1/completions`、`/v1/responses`、認証、logprobs は無い | 一部不足 (§6) |

- GLM の 384 と 224 は、測った負荷が ds41rt の公表値と違う。**両方を同じ再生負荷で測ってから判定する** (§4 M8)。
- sampling を有効にすると約 5% 遅くなる【資料、r2_ds41rt §5.3】。

### 3.2 工程

| 段 | 内容 | 形式の変化 | 日数【推測】 | 品質の見込み |
|---|---|---|---|---|
| **E3-0** | diffbot のレシピを 1017 にそのまま適用する (K2 MCG、校正なし、greedy beam 16) | ds41rt raw にそのまま載る | 1〜2 日 (GPU は半日) | 0909 の値で KL 0.361 / 0.215 / 0.077 (neutral / code / agentic)、HumanEval 91.5、tool eval 154〜159/176【資料】 |
| **E3-1** | 同じ K2 MCG を **tail-biting Viterbi + scale refit** で符号化し直す (sfxnz `requant_full.py`) | 不変 (同じ shard 配置とテンソル名。`requant_full.py:3-9`) | +2〜3 日 (Spark 2 台向けの分担処理を、ローカルの 2 GPU 用に書き直す) | relerr 0.377→0.262、NLL 0.243→0.139 nats/token【資料、`requant_full.py:16`、r2_quant_tools §5】 |
| E3-2 (任意) | 自分のトラフィックで取った Hessian で校正して K2 MCG にする (exllamav3 の LDLQ。H は T1 の活性から取る) | 不変 | +1〜2 週 | 校正ありの 2bit 級は KL 0.13〜0.20 (尺度は出典ごとに違う)【資料】 |
| E3-3 (任意) | ds41rt の staged manifest で K2/K3 を混ぜる (≤2.2bpw)。routing の頻度と感度で配分する | staged 形式に変わる。DSpark の experts も EXL3 にする必要がある | +1〜2 週 | 混合配分は効く (上位 25% の experts が routing の 56% を持つ)【資料】。KV を約 2GiB まで削る必要がある |
| E3-E (条件付き) | ds41rt の `EngramPipeline` にアプリ側の行キャッシュ (CLOCK 32GB) を足す | 不変 | +1〜2 週 | 数値は変わらない |

- E3-0 で動作と速度を確かめ、E3-1 で品質を上げる。ここまでで **合計 3〜5 日**【推測】。
- E3-2 と E3-3 は、§4 の G1 (品質の判定) に通らなかった場合にだけ進める。
- E3-E は、§4 の G3 (Engram の待ち) に通らなかった場合にだけ進める。

### 3.3 品質のリスク

| リスク | 根拠 | 対処 |
|---|---|---|
| 2.0bpw の一律量子化で劣化する (KL 0.36 級、PPL +13%) | diffbot (0909)【資料】。1017 の 2bit 一律でも KL 0.265 (Lucebox。0909 でも 0.266 で、量子化への耐性は同程度)【資料】 | E3-1 → E3-2 → E3-3 の順に対処する。合格線は §6 でユーザーが決める |
| reasoning code が止まらない (30 件中 2 件が 4,096 トークンの上限に到達) | ds41rt v7 の完了チェック【資料】 | 評価で max_tokens を大きくして再現性を確かめる。reasoning_effort の既定 (high) を見直す |
| 日本語の劣化 | 公開の校正データと評価は英語とコードが中心 | T1 の教師で、日本語の KL と日本語 QA を別に測る |
| 1017 の `wo_a` FP8 化と、ds41rt の FP8 GEMM の粒度の違い | `ref/model.py:787, 1073`。ds41rt は grouped FP8 で、粒度は未確認【資料】 | T1 との層 0 の比較か、出力の top-1 一致で確かめる |
| ds41rt の EXL3 経路は v7 以降に測られていない | r2_ds41rt §1.2【資料】 | v15 で動かなければ v7 の image に切り替える (追加の取得になる) |
| 開発者が 1 人のプロジェクト | r2_ds41rt §12 | MIT なので、必要な部分は fork して保守する |

### 3.4 予備の経路に切り替える条件

- E3 が v15 でも v7 でも起動しない場合、または ds41rt の形式 (一律 K と、隣り合う 2 段の混合) では品質が足りない場合は、**案 A** (vLLM + vllm-exl3 + expert 別 mixed-K) に移る。工数は 1.5〜2.5 週で、c1 の見込みは 100〜140 である (r2_quant_tools §1.6, §6)。
- LIL lowbit (4〜6 週) は、案 A でも c1 が足りない場合の長期の選択肢とする。

---

## 4. 課題 3: 先に測る値とその順序

### 4.1 測定の順序

| 順 | ID | 測る値 | 方法 | GPU | 状態 | 何を決めるか |
|---|---|---|---|---|---|---|
| — | X2 | Engram の行ヒット率 | 要求ログ 30,526 件のシミュレーション | 不要 | **完了** (r2_engram_hit) | D3 は不要。ページキャッシュでも成り立つ |
| M1 | X1 | 重みの完全性 | §2.2 段 0 の sha256 照合 | 不要 | 未実施。今すぐできる | 全工程の前提 |
| M2 | X3 | NVMe 4K randread の IOPS と遅延 | `research/tmp/plan/iops.py` を shard 47 に、QD 32〜256、1 / 2 プロセスで。ダウンロードが終わった単独の状態で測る | 不要 | 未実施。今すぐできる | 230k IOPS の前提が正しいか (QD64 で 200k 以上、p99 1ms 未満) |
| M3 | X3' | **mmap + `MADV_RANDOM` で、ページキャッシュに上限を付けたときの Engram gather** | X2 の行 ID トレースの一部を、`docker run --memory=80g` (cgroup がページキャッシュを制限する) の中で shard 47/48 の mmap に再生する。メジャーフォールトの数、実 IOPS、1 step あたりの待ちを測る | 不要 | 未実施。GLM が動いていてもできる (RAM を 80GB 使えるのは GLM の停止後) | ds41rt の Engram 方式 (D2) のままでよいか。D3 の改修 (E3-E) が要るか |
| M4 | X5 | 参照カーネルの SM120 での動作 | §2.2 段 1-4 | 要 (1 枚) | GLM の停止後 | T1 の修正量 |
| M5 | X6 | **教師 logits と、1017 + 自分のトラフィックでの routing の偏り** | T1 で `STORE.counts` を集める (層 × expert、5〜10 万トークン)。Gini、上位 25% の割合、下位 30% の割合を出し、0909 の公開値 (Gini 0.46、下位 30% で 7%) と比べる | 要 | GLM の停止後 | E3-3 (混合配分) の効果の大きさ。校正データの構成 |
| M6 | X7a | **E3-0 の KL** | T1 のハーネスの ExpertStore を、EXL3 の experts を exllamav3 で BF16 に戻して使うように改修する。教師と同じ位置で top-k KL と top-1 一致を出し、neutral / code / agentic / 日本語の 4 区分に分ける。diffbot の値と比べるため、AtomicChat 方式 (top-512) にそろえる | 要 | E3-0 の後 | G1 |
| M7 | X7b | 実タスクの品質 | ds41rt の API で次を測る: `qualify-ds41-tool-eval.py` (176 件)、GSM8K-200、HumanEval、日本語 QA、要求ログのエージェント再生 (`glm53/replay_openai.py` のサンプル)、`qualify-ds41-native-needle.py` (1M) | 要 | E3-0 の後 | G1 |
| M8 | X8 | 速度 | 要求ログの再生 (c1 / c4 / c16) を、**GLM と ds41rt に同じ負荷で**かける。あわせて `bench-ds41-concurrent-api.py`、prefill 8k / 32k / 128k を測る。Engram は `RUST_LOG=info,ds41rt::timing=debug` の gather 待ち、`/proc/vmstat` の pgmajfault、`iostat -x` の r/s で見る | 要 | E3-0 の後 | G2、G3 |
| M9 | X7c | E3-1 の KL と実タスク | M6 と M7 を E3-1 で繰り返す | 要 | E3-1 の後 | E3-2 / E3-3 に進むか |
| M10 | X11 | DSpark の受理長 | ds41rt の統計 (`/v1/stats`) を実トラフィックで見る | 要 | M8 と同時 | 速度の見込みの補正 |

**順序の理由**
- M1〜M3 は GPU を使わないので、GLM を止める前に済ませる。
- M4〜M5 は教師を作る工程である。教師が無いと M6 と M9 を測れない。
- KL の測定 (M6) を ds41rt でなくハーネスで行うのは、ds41rt の API に logprobs が無いためである。ds41rt の top-1 診断 (`collect-ds41-top1-agreement.py`) は、v5 の診断トレースを入れた build が前提になる (同ファイル 3-5 行)。

### 4.2 判定の基準 (案。合格線そのものはユーザーが決める、§6)

| ID | 判定 | 合格の案 | 不合格のとき |
|---|---|---|---|
| G1 品質 | M6 / M7 / M9 | agentic と code の top-1 一致が 0909 の 3bpw (96〜98%) から 3 ポイント以内。tool eval ≥150/176。HumanEval の低下 ≤3 点。日本語 QA の低下 ≤3 点 | E3-1 → E3-2 → E3-3 の順に進む |
| G2 速度 | M8 | 同じ再生負荷で、c16 の合計が GLM 以上、c1 ≥150 | 負荷の内訳を見る。DSpark の draft 長、`--concurrency`、KV の大きさを調整する |
| G3 Engram | M3 / M8 | gather の待ちが step 時間の 5% 以下、NVMe の r/s が 50k 以下 | E3-E (行キャッシュ)。4KiB に揃えた表 (X4) も検討する |

**KL の合格線について**: PLAN X7 の「KL ≤0.10」は、平均 2.0〜2.2bpw では届かない可能性が高い (r2_quant_tools §5)。そのため上の G1 は、top-1 一致と実タスクを基準にした案にしてある。

---

## 5. 課題 4: ディスクの予算 (現在の空き 560.9GB)

### 5.1 前提【実測】

- 空き: 560,858,320,896B (`df -B1 /`)。`/var/lib/docker` も同じ `/dev/nvme0n1p2` 上にある。
- 1017 の原本は 510.3GB で、取得は完了している。
- docker のイメージは合計 205.2GB で、うち 109GB は回収できる (`docker system df`)。glm53v が使っていないイメージは `sglang-mimo:20260930`、`lmsysorg/sglang:dev-cu13`、`lmsysorg/sglang:latest` である。
- 量子化版は、experts を含まない shard と Engram (shard 47/48、203GB) を **hardlink** で共有する。このため 1 版あたりの増分は、experts の約 137GB と、同じ shard に入っている非 expert テンソル (数 GB〜十数 GB) だけになる【資料、r2_quant_tools §1.2】。

### 5.2 作成と削除の順序

| 順 | 作るもの / 消すもの | 増減 | 作業後の空き (見込み) | 備考 |
|---|---|---|---|---|
| 0 | (現状) | — | 561GB | |
| 1 | X1 / X3 / X5 / T1 の作業ファイル (教師の top-k logits、routing の集計) | +5GB 未満 | 約 556GB | 教師は top-64 で保存する。全語彙の logits は保存しない |
| 2 | 量子化イメージ (LIL からの派生) | +2〜5GB【推測】 | 約 552GB | diffbot の第 1 案を取る場合は、vLLM day-0 image の分 (数十 GB) が増える |
| 3 | ds41rt coordinator v15 | +20〜25GB (展開後の推測) | 約 530GB | |
| 4 | **K2 v0** (E3-0) | +140〜150GB | 約 385GB | hardlink が効いていることを `stat -c %h` (リンク数 ≥2) で確かめる |
| 5 | **K2 v1** (E3-1、Viterbi + refit) | +140〜150GB | 約 240GB | v0 と比べ終わるまで両方を残す |
| 6 | v0 と v1 の負けた方を消す | −140GB | 約 380GB | 原本はここでも残す |
| 7 (任意) | E3-2 / E3-3 の版 | +150GB | 約 230GB | 同時に置くのは 2 版までとする |
| 8 (条件付き) | 4KiB に揃えた Engram 表 (X4) | +210GB | — | E3-E か案 A を選んだときだけ作る。作る前に手順 6 で 1 版に減らしておく |
| 9 (任意) | LibertAIDAI 2.9bpw (E2 の比較用) | +211GB | — | ユーザーの許可が要る。量子化版を 1 版にしてから取得する |
| 予備 | 使っていない sglang イメージ 3 本を消す | 最大 −109GB | — | ユーザーの許可が要る。GLM の再構築に要らないか確かめてから消す |

**消してはいけないもの**
- 1017 の原本 (510GB)。T1 の教師と再量子化の元であり、Engram の hardlink 先でもある。
- GLM の重み、イメージ (`ghcr.io/local-inference-lab/vllm:release-10152026`)、要求ログ (`glm53/dump/reqlog`)。

**採らないもの**: GGUF の中間ファイル (約 508GB)。

---

## 6. 課題 5: ユーザーの判断が要る点

### 6.1 運用

1. **glm53v を止めるか、いつ、どれだけの時間止めるか。**
   - T1 だけなら 2〜3 時間、T1 から T2 の初回起動までなら 8〜12 時間、E3-1 まで含めた評価なら 1〜2 日 (分けて止めてもよい)。
   - 戻すときは `sudo docker start glm53v` で、6〜9 分かかる。
   - ds41rt と GLM は同時に動かせない (VRAM と RAM の両方が足りない)。**切り替えて運用する**ことを前提にするか。
2. **`/dev/shm` に LMCache のファイルが残っていた場合に消すか。** 消すと GLM の再起動後に KV のキャッシュが冷えた状態から始まる。
3. **使っていない docker イメージ (最大 109GB) を消すか。**

### 6.2 品質と速度のトレードオフ

4. **2.0bpw の量子化をどこまで受け入れるか。合格線をどう置くか。**
   - E3 で全 experts を VRAM に置けるのは、ほぼ 2.0bpw (最大でも約 2.2bpw) である。VRAM が 96GB×2 の限りでは、3bpw 級の品質 (KL 0.055) は全 experts を VRAM に置く構成では得られない。
   - 品質を優先して E2 (2.9bpw + host experts) を選ぶと、c16 は 120〜320 tok/s に下がり、GLM を下回る (PLAN §2.3)。
   - 判断の材料: G1 の案 (§4.2)。PLAN X7 の KL ≤0.10 は届かない見込みである。
5. **c1 の目標 (≥150) を、どの負荷で測った値で判定するか。** code 系なら満たし、雑談系では 0.5〜0.6 倍になる (r2_ds41rt §10)。

### 6.3 ダウンロード

6. **ds41rt coordinator v15 のイメージ** (`ghcr.io/tpurtell/ds41rt-coordinator:v15`、圧縮 10.46GB)。v15 で EXL3 が動かない場合は v7 も取得する。
7. **vLLM day-0 image** (diffbot の量子化イメージを正規の手順で作る場合だけ)。LIL からの派生で作れれば要らない。
8. (任意) **LibertAIDAI 1017 2.9bpw** (211GB)。品質の比較用である。
9. exllamav3 を build するときに、不足する依存を pip で小さく取得すること。

### 6.4 ライセンス

| 部品 | ライセンス | 影響 |
|---|---|---|
| ds41rt | MIT【実測: `ext/ds41rt/LICENSE`】 | 本命の経路は MIT と Apache-2.0 だけで組める |
| sparkinfer (ds41rt の submodule) | Apache-2.0【資料、r2_ds41rt §1.1】 | 同上 |
| exllamav3 | MIT【実測】 | 量子化器 |
| sfxnz recipe | MIT | E3-0 と E3-1 の量子化 |
| diffbot recipe | MIT (第三者のコードは THIRD_PARTY_NOTICES に従う)【実測: `recipe/README.md`】 | 量子化スクリプト |
| **vcruz305/vllm-exl3** | **AGPL-3.0**【実測: `ext/vcruz305_vllm-exl3/LICENSE`】 | **予備の案 A だけに関係する。** 改変版をネットワーク越しに他人へ提供すると、ソースを開示する義務 (AGPL §13) が生じる。案 A に移るかどうかを決めるときに判断が要る |
| sudoingX | Apache-2.0 | E3-2 で校正手順を流用する場合 |
| DeepSeek-V4.1-Flash-1017 の重み | MIT【実測: モデルの `LICENSE`、README の `license: mit`】 | 量子化版の利用に制約は無い |

### 6.5 API と挙動の違い (GLM の利用者側で調整が要る点)

10. ds41rt にはいくつかの機能が無い (r2_ds41rt §7)。**前段に proxy (認証、`/v1/responses` の変換など) を置くかどうか**を決める。
    - 無いもの: `/v1/completions`、`/v1/responses`、API キー認証、logprobs、`n`、penalty 系。
    - 並列は 16 が上限。
11. **既定の挙動が GLM と違う。** クライアントで明示的に指定するか、proxy で既定値を書き換えるかを決める。
    - thinking が既定で high になる。
    - temperature を省くと greedy になる。
    - 返る model 名が `deepseek-ai/DeepSeek-V4.1-Flash` に固定されている。

### 6.6 開発への投資

12. E3-0 / E3-1 (3〜5 日) より先に進めるか。
    - E3-2 / E3-3 / E3-E は、それぞれ 1〜2 週かかる。
    - 予備の案 A は 1.5〜2.5 週、LIL lowbit は 4〜6 週かかる。
    - **G1〜G3 の結果を見てから決める**ことを勧める。
