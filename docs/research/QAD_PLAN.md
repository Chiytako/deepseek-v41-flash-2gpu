# QAD (量子化認識蒸留) の実行計画 — GPU が空いたら上から順に流す

作成: 2026-10-18 (v1 量子化の実行中、GPU 不使用で作成)。同日 18 時頃に画像の節 (§1V) を追加し、§0、§1.0、§2〜§9 を画像に合わせて直した (GPU 不使用)。
対象: DeepSeek-V4.1-Flash-1017 の EXL3 2.0bpw パック v1 (Viterbi + refit、relerr 0.262) の routed experts の suh/svh。
凡例: 【実測】手元で測った値、【計算】式から出した値、【推測】根拠が間接的な値。

**進み具合 (2026-10-18 17:50 頃)【実測】**: §1.1〜§1.3 は済んだ (v1 の KL は agent24 0.0593 / ja16 0.0739 / jakl32 0.1393、`work/teacher/kl_v1_*.log`。段 A の検査は `results/QUALITY.md` 末尾)。§1.5 の pilot capture 2 本 (`dsv41-qad-capstudent` / `capteacher`) が実行中。画像の手順 (§1V) はまだ GPU で 1 本も流していない。

---

## 0. 要約

- やること: L2 局所 QAD (層ごとに suh/svh を MXFP4 教師の expert 出力へ蒸留) を v1 に掛け、ハーネスの KL と ds41rt の生成で v1 と比べる。L3 (全体 KL) は L2 の結果を見てから決める。
- 学習するのは suh/svh だけで、パックの形式もバイト数も変わらない。推論の費用は増えない (SPEC §0.1)。
- 教師に合わせる蒸留なので、日本語性能の上限は原本 1017 と同じである (r3_qad §4.3)。狙うのは「v1 の劣化をどれだけ戻せるか」で、日本語は校正データの 40% を日本語にして守る (損失の日本語重み 2 倍の効果は小さい、§3.4)。
- **画像の性能も落とさない (ユーザーの要件。日本語・中国語混入抑制と並ぶ)。** 画像トークンは gate の `bias_vl` で選ばれ、テキストとほぼ別の専門家を通る (`bias` との相関は 40 層平均 −0.05【実測】、`research/r6_vision_qad.md` §2)。テキストだけの校正で L2 を掛けると、画像とテキストが共用する専門家の scale がテキストに合わせて動き、テキストの検証だけで採用される。そこで、(1) 最初に v1 / v0 の画像の基準 (画像 KL と ds41rt の画像正答率) を取り、(2) capture に画像系列を混ぜ、(3) L2 に画像の重みと「画像の守り」を掛け、(4) 合格基準に画像の関門を足す (§1V、§5)。
- 段取り (GPU 2 枚の壁時計、【計算 + 推測】):

| 段 | 中身 | 所要 | ディスク |
|---|---|---|---|
| A | 最初の検証: v1 の基準値、jakl32 の教師、分解と教師の一致、1〜2 層の試し学習、pilot (calib64k で 40 層) | 約 4 h | 一時 +52GB |
| V | 画像: ViT の smoke、vl_kl の教師と v1 / v0 の画像 KL、画像入り capture の smoke、ds41rt の画像正答率の基準 (§1V) | GPU 2 枚で約 1.2 h + ds41rt 約 20 分 | 数十 MB |
| A' | ds41rt の svh の位置 (カナリア) の検査 | 約 1 h | 一時 +3.6GB |
| B | capture 本番 (calib256k、192k preset) | 30〜50 分 | +81〜103GB |
| C | L2 全 40 層 (expert mode、必要なら layer mode) | 20〜90 分 | +0.7GB/版 |
| D | 評価 (ハーネス 1 版 約 45〜50 分 (画像 KL を含む)、ds41rt 生成 2.5〜3.5 h (画像正答率を含む)) | 3.5〜4.5 h | 0 (overlay) / +143.5GB (shards) |
| E (条件付き) | L3 | 5〜7 h | +143.5GB |

- 採否の基準 (§5): 日本語・エージェント・中国語混入・画像の 4 関門をすべて通ること。日本語 (ja16 + jakl32) の KL が v1 比で **5% 以上**下がり (系列 bootstrap の 95% 区間が 0 をまたがない)、agent24 の KL が +1% を超えて悪化せず、zh_only の質量が v1 より増えず、画像 KL が +1% 以内で有意な悪化が無く、ds41rt の生成ゲート (画像正答率を含む) を落とさなければ採用。10% 以上は「強い採用」。2% 未満なら不採用。
- MiMo-V2.6-Flash-RL の削除は、L2 の shards 版を作る時点で capture を残したいとき (§7)。GLM-5.3-Flash-W4A16 は L3 の export か K3 混合 (v2) に進む時点で検討する。

---

## 1. 段 A: GPU での最初の検証 (小さく)

### 1.0 共通の準備

以下のコマンドは bash 用である (ログインシェルは fish なので、`bash` を起動してから貼る)。

```bash
D=$DSV41; M=$MODELS
ORIG=$M/DeepSeek-V4.1-Flash-1017; V1=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1
Q=$M/qad            # QAD の作業場所。overlay は pack と同じファイルシステムに hardlink を張るので models の下に置く
mkdir -p $Q/cap $Q/l2 $Q/packs $D/work/qad
# GPU 1 枚のジョブ。models は同じパスでマウントする (差分の metadata に書く pack のパスが内外で一致する)
qrun() { local n=$1 g=$2; shift 2
  sudo docker run --rm --name dsv41-qad-$n --gpus "\"device=$g\"" --network none --ipc host --shm-size 16g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v $D/ref:/ref:ro -v $ORIG:/model:ro -v $D/harness:/h:ro -v $D/tools:/tools:ro \
    -v $D/work/teacher:/t -v $D/work/calib:/c:ro -v $D/work/vl:/vl:ro -v $M:$M \
    --entrypoint bash dsv41-quant:151 -c "cd /h && $*" ; }
# CPU だけのジョブ (apply、比較、CPU テスト)
qcpu() { sudo docker run --rm --network none -e CUDA_VISIBLE_DEVICES= \
    -v $D/ref:/ref:ro -v $D/harness:/h:ro -v $D/tools:/tools:ro -v $D/work/teacher:/t -v $D/work/calib:/c:ro \
    -v $D/work/vl:/vl:ro -v $ORIG:/model:ro -v $M:$M --entrypoint bash dsv41-quant:151 -c "cd /h && $*" ; }
```

- マウントと image は既存の教師ジョブ (`build/chain_g1.sh:5-7`) と L3 の手順 (`harness/qad_l3_design.md` §9) に合わせた。
- `-v $D/work/vl:/vl:ro` は画像の集合 (jsonl と `img/`) のために足した (§1V)。jsonl の画像は `img/...` の相対パスで、`stream_ref_vl.py` と `capture_moe.py` は jsonl のディレクトリを基準に読む。読み取り専用のマウントを 1 本足すだけなので、テキストだけのジョブの動作は変わらない。
- コンテナは root で書くので、出力 (capture、.pt) の所有者は root になる。消すときは `sudo rm`。

**開始前の確認 (5 分)**

1. v1 の量子化が終わっている: `sudo docker ps` に `dsv41-requant-g0/g1` が無く、`ls $M/dsv41-v1-work/shards | wc -l` が 40。
2. v1 を組み立てる: `$D/build/assemble_v1.sh` (hardlink だけで、追加のディスクは 0。`build/assemble_v1.sh:7-11`)。
3. GPU 2 枚が空いている: `nvidia-smi`。GLM の運用を止める時間帯であることを確かめる (SPEC §6.2-4)。
4. CPU テストを今の版で回し直す (並行編集があったため):
   `qcpu 'python3 test_capture_moe.py && python3 test_qad_l2.py && python3 test_qad_integration.py && python3 qad_l3.py selftest --ref /ref --teacher-dir /t'`
   いずれも全件通過 (41 / 49 / 24〜26 / 48) であること。
   画像の分も回す: `qcpu 'python3 test_stream_ref_vl.py /tmp/srvl && python3 test_vl_capture_qad.py /tmp/vlcq'` (81 / 50) と、ホストで `tools/.venv/bin/python tools/vl/test_vl_eval.py` (49 ケース)。
5. `df -h /` で空きが 180GB 以上あること (v1 完成後の見込みは約 185GB、§7)。

### 1.1 v1 の基準値と jakl32 の教師 (GPU 2 枚、約 55 分)

v1 の KL はまだ測っていない (`results/QUALITY.md` は v0 だけ)。
QAD の比較の基準は v1 なので、最初に取る。

```bash
# GPU0: v1 を agent24 と ja16 で採点 (40 系列 × 約 44 s = 約 30 分。v0 の採点は 93 tok/s【実測、work/teacher/kl_v0_ja16.log】)
qrun v1kl 0 "for s in agent24 ja16; do python3 stream_ref.py --teacher-in /t/\$s.jsonl --teacher-ref /t/\${s}_teacher.pt \
   --teacher-out /t/\${s}_v1.pt --exl3-pack $V1 --max-seq-len 8192 --expert-budget-gb 20 > /t/kl_v1_\$s.log 2>&1; done" &
# GPU1: jakl32 の教師 (32 系列 × 約 55 s = 約 30 分。教師は 68〜74 tok/s【実測、work/teacher/teacher_ja.log】)
qrun jaklT 1 "python3 stream_ref.py --teacher-in /c/jakl32.jsonl --teacher-out /t/jakl32_teacher.pt \
   --max-seq-len 8192 --expert-budget-gb 20 > /t/teacher_jakl32.log 2>&1"
# GPU1 の教師が終わったら v1 を jakl32 で採点 (約 25 分)
qrun v1jakl 1 "python3 stream_ref.py --teacher-in /c/jakl32.jsonl --teacher-ref /t/jakl32_teacher.pt \
   --teacher-out /t/jakl32_v1.pt --exl3-pack $V1 --max-seq-len 8192 --expert-budget-gb 20 > /t/kl_v1_jakl32.log 2>&1"
```

- 確認: v1 の agent24 / ja16 の KL が v0 (0.158 / 0.206) より下がっていること。下がっていなければ QAD より先に v1 を点検する (SPEC §5.4)。
- 版を比べる道具として `tools/kl_compare.py` を足した (§4.1)。v0 と v1 の比較もこれで出す:
  `qcpu 'python3 /tools/kl_compare.py --set agent24 /t/agent24_v0.pt /t/agent24_v1.pt --set ja16 /t/ja16_v0.pt /t/ja16_v1.pt'`
- jakl32 の v0 採点は任意 (v0 の値は比較に使わない)。

### 1.2 分解と教師の一致、capture の smoke (GPU 1 枚、約 30 分。1.1 の GPU1 の空きに入れてよい)

```bash
# (a) 分解: Q̃·外側 su/sv が get_weight_tensor と一致するか (qad_l2.py:940-953)。層 0 と 39、冷たい expert と熱い expert
qrun dec 1 "python3 qad_l2.py check-decomp --pack $V1 --layer 0 --experts 0,7,200 && \
            python3 qad_l2.py check-decomp --pack $V1 --layer 39 --experts 0,7,200"
# (b) capture の smoke: calib64k の 3 系列、1 層 8,192 行、学生 stream (方式 A)。約 3.4GB
qrun capS 1 "python3 capture_moe.py run --input /c/calib64k.jsonl --exclude /t/agent24.jsonl --exclude /t/ja16.jsonl \
   --exclude /c/jakl32.jsonl --out $Q/cap/smoke --preset 64k --rows-per-layer 8192 --limit-seqs 3 --exl3-pack $V1"
qcpu "python3 capture_moe.py verify $Q/cap/smoke && python3 capture_moe.py info $Q/cap/smoke"
# (c) 教師の一致: torch で再現した MXFP4 教師と fp4_gemm (stream_ref.expert_forward) の差 (qad_l2.py:956-977)
qrun tch 1 "python3 qad_l2.py check-teacher --orig-ckpt /model --capture $Q/cap/smoke --layer 0 --experts 0,7 && \
            python3 qad_l2.py check-teacher --orig-ckpt /model --capture $Q/cap/smoke --layer 39 --experts 0,7"
```

合格の目安と、落ちたときの対処:

| 項目 | 合格 | 落ちたら |
|---|---|---|
| check-decomp | 相対差 1e-3 以下 (CPU の fp16 経路で 4.1e-4【実測、test_qad_l2】) | 先に進まない。`qtilde_exl3` (qad_l2.py:254) と exllamav3 の呼び方を照合する |
| verify | 全層で idx が一致、w の差が fp32 の丸め程度 | capture のパッチ (capture_moe.py:454 `install`) を点検。stream_ref の MoE 差し替えとの順序 |
| check-teacher | 相対差 1e-2 以下 (bf16 の丸め程度) | 3e-2 を超えたら、活性 FP8 量子化の再現 (qad_l2.py:154、有無で 4.5% 変わる【実測】) と FP4 の nibble 順を疑う。直るまで学習しない |

### 1.3 1〜2 層の試し学習 (GPU 1 枚、約 10 分)

```bash
qrun l2S 1 "python3 qad_l2.py train --capture $Q/cap/smoke --layers 0,39 --orig-ckpt /model --pack $V1 \
   --out $Q/l2/smoke --min-tokens 64 --verbose > $Q/l2/smoke.log 2>&1"
for f in $Q/l2/smoke/qad_l2_L*.json; do jq -c '{layer, accepted, rejected, cold, seconds, layer_val}' $f; done
```

- smoke の行数は少ない (1 expert 平均約 128 トークン) ので、`--min-tokens` を 64 に下げて学習を走らせる。損失が下がる仕組みの確認だけが目的で、この差分は評価に使わない。
- 合格: NaN が無い、`accepted` が `rejected` より多い、`layer_val.weighted.qad / layer_val.weighted.v1` が 0.95 以下 (報告の作りは qad_l2.py:640-653)、1 層の時間と `torch.cuda.max_memory_allocated` が見積もり (`qad_l2.py estimate`、64k 行で計算 16〜30 s/層、GPU 約 2GiB【計算】) の 3 倍以内。
- 時間が見積もりの 3 倍を超えたら、§3 の全層の所要をその倍率で読み替える。

### 1.4 段 A': ds41rt で svh が clamp の前に掛かっているか (GPU 2 枚、約 1 h)

ハーネスは exllamav3 の `get_weight_tensor` で密な W を作るので、svh は必ず clamp の前に入る。
ds41rt のネイティブカーネルでもそうなっているかは未確認である (SPEC §7.3、r3_qad §7-2)。
ここが違うと、ハーネスでの L2 の改善が ds41rt に出ない。
L2 の学習結果を待たずに確かめられるので、段 A の途中 (GPU 2 枚が空いた時間) に入れる。

**方法 (カナリア差分)**: `harness/svh_canary.py` (新規、`make_canary` は :27) で、層 0 の全 expert について up (w3) の svh を c=64 倍、down (w2) の suh を 1/64 倍にした差分を作る。

- 正しい順 (up = clamp((x⊙su)·Q̃·sv, ±10)) では、|up| > 10/64 の位置で clamp が効き、層 0 の出力がはっきり変わる。
- svh を clamp の後で掛ける誤った順では、64 と 1/64 が打ち消し合い、出力は v1 とほぼビット一致になる (64 は 2 冪なので fp16 の値は丸めなしで移る)。
- 合成データでの確認: 正しい順で相対変化 0.719、誤った順で 0【実測、CPU】。v0 パックの層 0 で差分を作れることも確かめた (768 tensors、base_sha256 一致)【実測】。

```bash
qcpu "python3 svh_canary.py --pack $V1 --layer 0 --c 64 --out $Q/l2/canary_L00 && \
      python3 qad_l2.py apply --pack $V1 --diff $Q/l2/canary_L00 --out $Q/packs/v1-canary --mode shards"
$D/build/stage_hf.sh $Q/packs/v1-canary local/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1-canary 1017e3c2000000000000000000000000000000c1
```

- shards mode は shard 00003 (層 0) だけを実体コピーする (+3.6GB)。残りは hardlink (qad_l2.py:883-911)。
- ds41rt の活性トレースを使う。`v41_target_pass.rs:348-356` は `RUST_LOG` に `ds41rt::activation_trace=debug` があり、`DS41RT_ACTIVATION_TRACE_POSITION` と同じ位置から始まる batch で、層ごとの残差を `DS41RT_ACTIVATION_TRACE_DIR` に書く (`:482-483`)。
- ただし `run-local.sh:469-470` は coordinator に `RUST_LOG` しか渡さない。`run-local.sh` を `run-trace.sh` に複製し、coordinator の `docker run` に `-e DS41RT_ACTIVATION_TRACE_POSITION=0 -e DS41RT_ACTIVATION_TRACE_DIR=/trace -v $D/work/qad/trace:/trace` を足す (元の run-local.sh は変えない)。
- 手順: v1 で起動 → 同じ短いプロンプト (日本語 1 本、temperature 0) を 1 回投げる → 停止 → カナリアで同じことをする → `python3 ext/ds41rt/scripts/compare-ds41-layer-residuals.py` で 2 つのトレースを比べる。

| 結果 | 判定 |
|---|---|
| 層 0 で `rms / reference_rms` が 1e-2 を大きく超える (first_differing_layer=0) | 正しい順。L2 / L3 の結果は ds41rt にそのまま出る |
| 層 0 がほぼビット一致 | 誤った順か、clamp が効いていない。ハーネスでカナリアの ja16 を採点し (約 12 分)、KL が v1 より大きく悪化すれば clamp は効いているので、ds41rt の順が誤り。その場合 L2 の成果は vLLM-exl3 (`build/serve_vllm_exl3.sh`、exllamav3 のカーネル) でしか使えない。ds41rt のカーネル修正はユーザー判断 |

- トレースが取れない場合の代わり: 同じプロンプトを temperature 0 で 256 トークン生成し、v1 と文字列が完全一致するかを見る (一致なら誤った順を強く疑う)。
- 検査が済んだら `$Q/packs/v1-canary` と hf の canary snapshot を消す。**カナリアは運用に使わない。**
- gate の svh は相殺の形が作れないので検査できない。up と gate は同じ経路 (w13) で読むと見込む【推測】。

### 1.5 pilot: calib64k で 40 層 (GPU 2 枚、約 2.5 h)

ここで「L2 が実データで KL を下げるか」と「capture を学生 stream と教師 stream のどちらで取るか」を決める。

```bash
# capture を 2 本並べて取る (各 22 系列、63,811 トークン、64k preset はほぼ全トークン保持で 26GB 以下【計算】)
qrun capPs 0 "python3 capture_moe.py run --input /c/calib64k.jsonl --exclude /t/agent24.jsonl --exclude /t/ja16.jsonl \
   --exclude /c/jakl32.jsonl --out $Q/cap/p64_student --preset 64k --exl3-pack $V1 > $Q/cap/p64_student.log 2>&1" &
qrun capPt 1 "python3 capture_moe.py run --input /c/calib64k.jsonl --exclude /t/agent24.jsonl --exclude /t/ja16.jsonl \
   --exclude /c/jakl32.jsonl --out $Q/cap/p64_teacher --preset 64k > $Q/cap/p64_teacher.log 2>&1"; wait
# 学習: stream ごとに GPU 1 枚で 40 層 (64k 行で計算 16〜30 s/層 + 起動の手間。1 本 25〜40 分の見込み)
qrun l2Ps 0 "python3 qad_l2.py train --capture $Q/cap/p64_student --layers 0-39 --orig-ckpt /model --pack $V1 \
   --out $Q/l2/p64_student > $Q/l2/p64_student.log 2>&1" &
qrun l2Pt 1 "python3 qad_l2.py train --capture $Q/cap/p64_teacher --layers 0-39 --orig-ckpt /model --pack $V1 \
   --out $Q/l2/p64_teacher > $Q/l2/p64_teacher.log 2>&1"; wait
# ハーネス用の overlay (hardlink + 差分 40 本、+0.7GB/版)
for s in student teacher; do qcpu "python3 qad_l2.py apply --pack $V1 --diff $Q/l2/p64_$s --out $Q/packs/v1-l2p64$s --mode overlay"; done
```

- 時間: capture は学生 stream が約 12 分 (93 tok/s)、教師 stream が約 16 分 (70 tok/s) に、build と書き込みが加わる【計算】。2 本を同時に回すと NVMe の取り合いで 1.0〜1.5 倍遅くなりうる【推測】。
- 評価は §4.1 の手順で、2 版 × (agent24 + ja16 + jakl32 + vl_kl_ja + vl_kl_en) を採点する (1 版あたり GPU 2 枚で約 45〜50 分、2 版で約 1.5 h)。vl_kl は §1V.2 の教師が要る。
- pilot の capture はテキストだけ (calib64k) なので、pilot の画像 KL は §1V の「リスク 2 (テキストだけの L2 が画像を悪くする)」をそのまま測る。画像 KL が v1 比 +1% を超えて悪化したら、本番 (段 B) で画像系列を混ぜる必要が実データで裏付けられたことになる。悪化しなくても本番では画像を混ぜる (守りを掛けるため)。
- 層ごとの報告は `jq -c '{layer, accepted, cold, layer_val}' $Q/l2/p64_*/qad_l2_L*.json` で見る。冷たい expert (`cold`) が層の半分を超えるなら、本番は `--expert-floor` を上げる。

**pilot の判定 (v1 比)**

| 結果 | 次の段 |
|---|---|
| どちらかの stream で、日本語 (ja16 + jakl32) の KL が −2% 以上下がり、95% 区間が 0 をまたがず、agent24 が +1% を超えて悪化しない | 段 B へ。stream は日本語の KL が低い方を採る。差が 1% 未満なら学生 stream (SPEC §2.5 の方式 A) |
| 両方とも −2% に届かない | 本番の行数を増やしても伸びしろは小さいと見る。`--loss layer --init-diff $Q/l2/p64_student/qad_l2_L{L:02d}.safetensors` を 1 回だけ試し (約 1 h)、それでも届かなければ L2 を打ち切って §6 の L3 判定へ |
| どちらかの区分が +3% を超えて悪化 | 本番に進まない。`--l2` を 0.3 に上げる、`--lr` を 2.5e-4 に下げる、のどちらかで pilot をやり直す |

- pilot の数字は本番より小さく出る見込み (校正が 1/4、冷たい expert が多い)。
- 判定が済んだら capture 2 本 (約 52GB) を消す。L2 の差分 (各約 0.7GB) と overlay は残してよい。

---

## 1V. 画像 (段 V): 基準取りと、QAD に画像を入れる手順

**なぜ要るか** (詳細は `research/r6_vision_qad.md`)

- 画像の経路で量子化が効くのは routed experts だけである。ViT 32 層と aligner、区切り埋め込みは全版で BF16 原重みのまま、Engram は画像位置で DEAD (引かない) である (r6 §1、`ref/model.py:1222-1276`)。
- 画像トークン (START / IMAGE / NEW_LINE / END の全位置) は、gate で `bias_vl` を使って専門家を選ぶ (`model.py:811-830`)。ds41rt の router カーネルも同じである (`ext/ds41rt/native/cuda/kernels/v41_router.cu:83`)。
- `bias_vl` は `bias` とほぼ無相関で (40 層平均 Pearson −0.05、Spearman −0.32)、画像で押し上げられる層ごと上位 19 個は、テキストの routing の 3.1% しか持たない (一様なら 4.9%)【実測、r6 §2.1-2.2】。層 0 の CPU 近似では、画像の routing の 27% がテキストで冷たい専門家 (下位 115) に行く (テキスト自身は 7.6%)【実測、近似。近似はテキストで cos 0.43 なので向きだけを見る】。
- リスク 1 (L2): テキストだけの capture と検証では、画像とテキストが共用する専門家の suh/svh が、画像を悪くしても採用される。L2 は入力の各次元の scale (suh) を動かすので、テキストで小さい次元を画像が使っていればそこが崩れうる【推測】。テキストで冷たく画像で熱い専門家は cold (行数不足) で v1 のままになるので、こちらは安全側である。
- リスク 2 (v1 自体): v1 の画像性能はまだ一度も測っていない。v0→v1 でテキストの KL は約 −62% だったが、画像も同じだけ戻ったかは分からない。
- そこで「最初に v1 / v0 の画像の基準を取る」→「capture に画像を混ぜる」→「L2 に画像の守りを掛ける」→「合格基準に画像の関門を足す」の順で入れる。

**使うもの** (どれも CPU テスト済み。GPU の経路は未実行)

| 物 | 中身 | 出典 |
|---|---|---|
| `harness/stream_ref_vl.py` | 画像入り系列の教師 logits / 学生採点 / 教師の greedy 生成。`stream_ref.py` は変えていない。.pt は既存と同じキー (`recs`、`counts`、`kl`、`agree`) に `counts_text`、`counts_image` (画像位置だけの routing 回数 [40, 384]。r6 の言う `counts_vl` はこれ)、`recs[i].spans / q_spans / lang / cat / sub` を足したもの | test_stream_ref_vl 81/81 |
| `work/vl/vl_kl.jsonl` | 教師 KL 用 50 系列 (日本語 28 / 英語 22)、28,734 トークン (画像 23,714)、採点 3,972 トークン (日本語 3,094 / 英語 878)。応答は参照回答 | vl-data |
| `work/vl/vl_kl_ja.jsonl`、`vl_kl_en.jsonl` | 上を lang で分けたもの (28 / 22 行、行の中身は同じ)。`kl_compare.py` は set の名前で日本語をまとめるので、画像の日本語 KL を出すために分けた | 本節で作成。和集合が元と一致することを確認【実測】 |
| `work/vl/vl_smoke.jsonl`、`vl_smoke_ctrl.jsonl` | smoke 用。vl_kl_ja の 2 行 (heron 写真、合成レシート) と、同じ問いと回答で画像を入れ替えた対照 4 行 (own 2 / swap 2) | 本節で作成。`--plan-only` を通過【実測】 |
| `work/vl/vl_calib.jsonl` | QAD 校正用 24 系列 (日本語 16 / 英語 8)、13,207 トークン (画像 9,758) | vl-data |
| `work/vl/vl_gen.jsonl` + `tools/vl_eval.py` | ds41rt の生成評価 245 問 (日本語 145 / 英語 100)。thinking disabled、T=0、並列 4、`--compare` で同じ問いの対の差と 95% 区間、日本語の問いの zh_only を含む回答数 | test_vl_eval 49 ケース |
| `capture_moe.py` の画像対応 | 画像付きの行で `model.forward(ids, 0, images, token_types)`、field `tt`、`--img-boost`、統計 `tok_img`、verify は bias_vl で照合 | test_vl_capture_qad 50/50 |
| `qad_l2.py` の画像対応 | `--img-weight`、`--img-guard-tol`、`--min-val-img` (32)、`--val-frac-img` (0.25)、`layer_val.img / text`、`rejected_img`、`img_guarded` | 同上 |

- 下見は済んでいる【実測、CPU コンテナ、2026-10-18】: `stream_ref_vl.py --plan-only --max-seq-len 8192` が vl_kl_ja / vl_kl_en / vl_calib / vl_smoke / vl_smoke_ctrl のすべてで通った (最長 1,601 トークン)。`capture_moe.py plan` に calib256k と vl_calib を並べると、103 系列 274,056 トークン、画像 9,758 トークン (3.6%)、`p_img` 1.0、row_bytes 10,291、part 0 の期待 40.5GB だった。

### 1V.1 smoke: ViT が SM120 で動くか、画像が効いているか (GPU 1 枚、約 10〜15 分)

```bash
# (a) 教師で smoke 2 行 → 同じ原重みを学生にして自己一致 (H-VL0)
qrun vlS 1 "python3 stream_ref_vl.py --vl-in /vl/vl_smoke.jsonl --teacher-out /t/vl_smoke_teacher.pt \
   --max-seq-len 8192 --expert-budget-gb 20 > /t/teacher_vl_smoke.log 2>&1 && \
   python3 stream_ref_vl.py --vl-in /vl/vl_smoke.jsonl --teacher-ref /t/vl_smoke_teacher.pt \
   --teacher-out /t/vl_smoke_self.pt --max-seq-len 8192 --expert-budget-gb 20 > /t/kl_vl_smoke_self.log 2>&1"
# (b) 対照: 同じ問いと回答で、正しい画像 (own) と入れ替えた画像 (swap) の教師 nll を比べる
qrun vlC 1 "python3 stream_ref_vl.py --vl-in /vl/vl_smoke_ctrl.jsonl --teacher-out /t/vl_smoke_ctrl.pt \
   --max-seq-len 8192 --expert-budget-gb 20 > /t/teacher_vl_smoke_ctrl.log 2>&1"
grep -E '^\[(route|teacher|student)\]' $D/work/teacher/teacher_vl_smoke.log $D/work/teacher/kl_vl_smoke_self.log $D/work/teacher/teacher_vl_smoke_ctrl.log
```

| 項目 | 合格 | 落ちたら |
|---|---|---|
| 起動 | ViT (SDPA、BF16) と aligner が SM120 で落ちずに走り、build のログに vision の重みの missing / unused が 0 | ViT の SDPA を math backend に落とす改修を `stream_ref_vl.py` に足す (CPU テストを足してから)。直るまで段 V の残りと画像入り capture (§1V.4) は止め、段 B はテキストだけで進める |
| `[route] image tokens` | plan の画像トークン数 (317 + 508 = 825) と一致 | image_mask が MoE に届いていない。RouteCounter と `stream_ref.moe_forward` の image_mask の受け渡しを点検 |
| H-VL0 自己一致 | KL 0.0000、top-1 100% (QUALITY.md のテキストの再現性と同じ基準) | 画像の入力 (patches、token_types) が教師と学生で違う。sha256 の照合と、prefill 1 回に画像を渡す経路を点検 |
| 対照 (own と swap) | 2 問とも own の nll が swap より明らかに低い (目安 0.3 nat/token 以上)【推測の目安】 | 自己一致は画像が無視されていても通るので、ここが本当の検査である。差が無ければ画像の埋め込みが入っていない (merge_image_embeddings の前後) |
| メモリ | `torch.cuda.max_memory_allocated` がテキストの採点 (expert-budget 20GB) より ViT の約 1GB 程度しか増えない | `--expert-budget-gb` を下げる |

- (a) と (b) は同じ GPU で続けて流す。1 系列あたり約 30〜40 s (短い系列でも 40 層の専門家の読み込みがほぼ全部起きる。agent24 の expert 読み 874 s / 24 系列 = 36 s【実測、kl_v1_agent24.log】) に、build が 3 回で約 10〜15 分【推測】。

### 1V.2 画像の基準: 教師と v1 / v0 の画像 KL (GPU 2 枚、約 1 h。**段 V で最初に流す本番**)

```bash
V0=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw
# 教師 (原重み)。日本語 28 系列を GPU0、英語 22 系列を GPU1 (各 15〜20 分)
for s in ja en; do g=$([ $s = ja ] && echo 0 || echo 1)
  qrun vlT$s $g "python3 stream_ref_vl.py --vl-in /vl/vl_kl_$s.jsonl --teacher-out /t/vlkl_${s}_teacher.pt \
     --max-seq-len 8192 --expert-budget-gb 20 > /t/teacher_vlkl_$s.log 2>&1" & done; wait
# 学生: v1、続けて v0 (各 15〜20 分 × 2)
for s in ja en; do g=$([ $s = ja ] && echo 0 || echo 1)
  qrun vlS$s $g "for v in v1:$V1 v0:$V0; do n=\${v%%:*}; p=\${v#*:}; python3 stream_ref_vl.py --vl-in /vl/vl_kl_$s.jsonl \
     --teacher-ref /t/vlkl_${s}_teacher.pt --teacher-out /t/vlkl_${s}_\$n.pt --exl3-pack \$p \
     --max-seq-len 8192 --expert-budget-gb 20 > /t/kl_\${n}_vlkl_$s.log 2>&1; done" & done; wait
# 比較 (CPU)。vlja を「日本語」にまとめ、ALL が画像全体になる。--jsonl で区分 (sub) ごとにも出る
qcpu "python3 /tools/kl_compare.py --set vlja /t/vlkl_ja_v0.pt /t/vlkl_ja_v1.pt --set vlen /t/vlkl_en_v0.pt /t/vlkl_en_v1.pt \
   --jsonl vlja=/vl/vl_kl_ja.jsonl --jsonl vlen=/vl/vl_kl_en.jsonl --ja-sets vlja --json /t/cmp_vl_v0_v1.json"
# 中国語専用トークンの質量 (日本語の画像問の回答、教師比)
qcpu "python3 /tools/zh_teacher_mass.py --teacher /t/vlkl_ja_teacher.pt --student v0=/t/vlkl_ja_v0.pt \
   --student v1=/t/vlkl_ja_v1.pt --out /t/zh_vlkl_ja.json"
```

- 見るもの:
  - v1 の画像 KL と top-1 (日本語 / 英語 / 全体、区分ごと)。テキスト (agent24 0.059、ja16 0.074) と比べて大きく悪ければ、画像の経路は v1 の量子化誤差に弱い。
  - v0→v1 の変化。テキストの −62〜−64% と同じ程度に下がっていれば、画像も量子化誤差に素直に反応している。v0→v1 で画像だけ下がり方が小さければ、画像側の専門家が冷遇されている疑いが強い。
  - `counts_image` (画像位置だけの routing) を教師の .pt から取り出し、画像の偏り (層ごとの Gini、テキスト上位 96 との重なり、テキストで冷たい専門家に行く割合) を出す (r6 §6 の G2。CPU 数分)。`tools/r6_vl_bias.py --counts` は `counts` (テキスト + 画像) しか読まないので、`counts_image` を読む数十行の集計を足す (未作成)。この値は QAD ではなく、K3 混合 (v2) の π_img (r6 §4: Gini 0.5 以上で 0.20、0.3〜0.5 で 0.10、0.3 未満で 0.05) を決めるのに使う。
- v0 の学生は基準の「感度」を見るためで、QAD の比較の基準は v1 である。v0 の expert shard は §7 のとおりまだ残っている。消した後なら v0 の行は飛ばす。
- 採点区間は行の `score_from` から末尾まで (参照回答と EOS、`</think>` の予測を含む)。英語の公開データ (ChartQA / DocVQA / TextVQA / MMBench) は 1 問約 9 トークンしかないので、英語と区分ごとの KL は誤差が大きい。判断は全体と日本語の値で行う。
- 任意 (時間があるとき): 英語の公開データの問いについて、教師に 256 トークンほど自分で答えさせ (`--gen-out`、answer を外した行)、その回答で teacher-forced KL を取る版。教師の decode は約 1.5 tok/s【実測、work/teacher/greedy3.json: 96 tok に 55〜70 s】なので、14 問で約 45 分かかる。

### 1V.3 ds41rt の画像正答率の基準 (ds41rt が v1 で動いているとき、約 20 分)

ds41rt はハーネスと同時に動かせない (§4.2)。ハーネスの GPU 作業が終わり、ds41rt を v1 で戻した時点で流す。

```bash
cd $D
# (1) 形式の確認 (1 問だけ)。usage.prompt_tokens を参照レンダの 458 (うち画像 422) と比べる
tools/.venv/bin/python tools/vl_eval.py --limit 1 --filter synth-web-login-000 --out work/vl/results/check_v1.jsonl
# (2) v1 の基準 (245 問、thinking disabled、T=0、並列 4)
tools/.venv/bin/python tools/vl_eval.py --out work/vl/results/v1.jsonl
```

- (1) の prompt_tokens が 458 と違えば、ds41rt のテンプレートか画像の縮小規則がハーネス (公式 `ref/image_processor.py`) とずれている。そのときはハーネスの画像 KL と ds41rt の画像正答率が別の入力を見ていることになるので、(2) に進む前に原因を調べる (ds41rt の `native_v41/images.rs`、`native_v41.rs:333`)。
- 所要は 245 問 × 64 トークン上限、並列 4 で 10〜20 分【推測】。運用中の ds41rt に負荷を掛けるので、利用の少ない時間に流す。
- v0 の ds41rt 正答率は任意である (ds41rt を v0 で再起動する必要があり、再起動はユーザー判断)。
- 原本 (量子化なし) の画像正答率は取れない。vLLM の原本構成は `--language-model-only` で、ハーネスの greedy 生成は 245 問で約 5 h かかる【計算: 1 問 prefill 約 30 s + 64 tok / 1.5 tok/s】。基準は v1 との対の比較だけにする。

### 1V.4 capture に画像系列を混ぜる (段 B の変更点)

まず段 A の中で、画像入りの capture が実モデルで通るかを確かめる (CPU テストは偽モデルだけ)。

```bash
# smoke: vl_calib の 3 系列、1 層 8,192 行 (GPU 1 枚、約 5〜10 分)
qrun capV 1 "python3 capture_moe.py run --input /vl/vl_calib.jsonl --img-boost 2 --out $Q/cap/smoke_vl \
   --preset 64k --rows-per-layer 8192 --limit-seqs 3 --exl3-pack $V1 > $Q/cap/smoke_vl.log 2>&1"
qcpu "python3 capture_moe.py verify $Q/cap/smoke_vl && python3 capture_moe.py info $Q/cap/smoke_vl"
```

- 合格: verify で画像行の idx が bias_vl の選択と一致する (テキスト用 bias で選び直すと一致しないことは CPU テストで確認済み)。info の `tok_img` が 0 でない。`meta.vl = true`、row_bytes 10,291。
- 落ちたら段 B はテキストだけで取り、画像の守りは掛けられないものとして §5 の画像関門で判定する。

本番 (§2) では、calib256k に `vl_calib.jsonl` を `--input` で足し、`--img-boost 2` を付け、評価用の `vl_kl.jsonl` を `--exclude` に足す (§2 のコマンドに反映済み)。

- 画像トークンは 9,758 (全体の 3.6%) で、`p_img` = 1.0 (全部残る) なので、1 層 196,608 行のうち約 5% が画像の行になる【計算、`capture_moe.py plan` の出力】。
- 行数の上限 (preset) は同じなので、ディスクはほぼ増えない (row_bytes が 1B 増えて +0.01%)。時間は画像の分のトークン (+3.7%) と ViT (1 枚 1,000 トークンで数十 ms【推測】) で、+1〜2 分。
- vl_calib は 24 系列しかない。校正の画像比率を上げたいときは、tools/vl の公開データと合成画像を増やすか、要求ログの実画像 (288 件) を使う (§9 のユーザー判断)。

### 1V.5 L2 に画像の重みと守りを掛ける (段 C の変更点)

§3 のコマンドに `--img-weight 1.5 --img-guard-tol 0` を足す (反映済み)。

- `--img-weight 1.5`: 画像トークンの損失の重み (r6 §3.2 の推奨)。合成データでは 1→4 で画像の層出力の誤差 0.0393→0.0318、テキスト 0.0550→0.0561 だった【実測、合成】。
- `--img-guard-tol 0`: 画像の検証対が 32 以上ある expert で、学習後の画像だけの検証誤差が v1 を上回れば採らない (`rejected_img`)。
- `--val-frac-img 0.25` (既定) で画像系列 24 本のうち 6 本を検証に回す。1 expert あたりの画像の検証対は平均約 38 で、守りの下限 32 を超える【計算、レビュー】。その代わり画像の学習データは 18 本になる。
- 層ごとの報告で `layer_val.img`、`rejected_img`、`img_guarded` を見る (H-VL4)。
  `for f in $Q/l2/main_e/qad_l2_L*.json; do jq -c '{layer, accepted, rejected, rejected_img, img_guarded, img: .layer_val.img, text: .layer_val.text}' $f; done`
- `rejected_img` の合計が `accepted` の 10% を超えたら、`--img-weight 2.0` で 1 回だけ学習し直すかを判断する (+15〜30 分)。

### 1V.6 所要 (追加分)

| 段 | 中身 | GPU | 所要【計算 + 推測】 |
|---|---|---|---|
| V.0 | CPU テスト、plan-only (済) | — | 10 分 |
| V.1 | ViT smoke、自己一致、own / swap の対照 | 1 枚 | 10〜15 分 |
| V.2 | vl_kl の教師 (ja / en を 2 枚に分ける) | 2 枚 | 15〜20 分 |
| V.2 | v1、v0 の画像 KL | 2 枚 | 30〜40 分 |
| V.4 | 画像入り capture の smoke + verify | 1 枚 (V.1 と同じ枠に入る) | 5〜10 分 |
| V.3 | ds41rt の形式確認と v1 の画像正答率 | ds41rt (2 枚) | 約 20 分 |
| 段 B | capture 本番に vl_calib を足す | 2 枚 | +1〜2 分 |
| 段 C | L2 の画像の重みと守り | 2 枚 | ほぼ 0 (検証の計算が少し増えるだけ) |
| 段 D | 1 版あたりの画像 KL (vl_kl_ja / en) | 2 枚 | +15〜20 分/版 |
| 段 D | 1 版あたりの ds41rt の画像正答率 | ds41rt | +15〜20 分/版 |

- 段 V の GPU は合計で約 1.2 h (V.1 + V.4 を GPU1 で流す間、GPU0 で V.2 の日本語の教師を先に始めてよい)。pilot の評価に vl_kl を足す分 (2 版で約 40 分) は §1.5 に含めた。

---

## 2. 段 B: capture 本番 (校正データの記録)

**入力**: `work/calib/calib256k.jsonl` (79 系列、260,849 トークン、日本語 40.1%) と `work/vl/vl_calib.jsonl` (画像入り 24 系列、13,207 トークン、うち画像 9,758)。評価集合 (agent24、ja16、jakl32) とは時間分割と shingle で分けてある (重なり 0.75% / 0.17%)。vl_calib と画像の評価集合 (vl_kl、vl_gen) は画像の sha256 と元データの行で互いに素である (build_sets.py の assert)。

```bash
# 例は学生 stream (pilot で教師 stream を採ったら --exl3-pack を外す)。part ごとに別ディレクトリ
# 画像入りの行 (vl_calib) は §1V.4 の smoke が通ったときだけ足す。通らなければ --input /vl/... と --img-boost を外す
for g in 0 1; do qrun cap$g $g "python3 capture_moe.py run --input /c/calib256k.jsonl --input /vl/vl_calib.jsonl \
   --img-boost 2 --exclude /t/agent24.jsonl --exclude /t/ja16.jsonl --exclude /c/jakl32.jsonl --exclude /vl/vl_kl.jsonl \
   --out $Q/cap/main_p$g --part $g/2 --preset 192k --exl3-pack $V1 > $Q/cap/main_p$g.log 2>&1" & done; wait
qcpu "python3 capture_moe.py verify $Q/cap/main_p0 && python3 capture_moe.py verify $Q/cap/main_p1"
```

- 先に CPU で見積もる: `qcpu "python3 capture_moe.py plan --input /c/calib256k.jsonl --input /vl/vl_calib.jsonl --img-boost 2 --exclude /t/agent24.jsonl --exclude /t/ja16.jsonl --exclude /c/jakl32.jsonl --exclude /vl/vl_kl.jsonl --out $Q/cap/main_plan --preset 192k --part 0/2"`。区分が ja/agent/code/en/multi に分かれていること (capture_moe.py:185 `infer_domain` の修正後、テキストは ja 36 / agent 24 / code 9 / en 6 / multi 4 系列)。画像を足した plan【実測、2026-10-18 CPU】: 全 103 系列 274,056 トークン、画像 9,758 (3.6%)、`p_base` 0.61、`p_img` 1.0、part 0 は 52 系列で画像 5,180 トークン (ja 2,700 / en 2,480)、期待 40.5GB。
- 画像入りの記録は `meta.vl = true` で、row_bytes は 10,291 (field `tt` の 1B)。verify は画像行を bias_vl で照合する。info の `tok_img` で画像の行数を見る。
- 所要: 1 枚あたり約 13 万トークンで、学生 stream なら 23〜25 分、教師 stream なら 30〜33 分。build と書き込み (fsync) を含めて 30〜50 分【計算 + 推測】。中断しても同じコマンドで再開できる (コミット後の書きかけは切り捨て。capture_moe.py の commit / repair)。
- 容量: 1 行 10,290B × 196,608 行 × 40 層で期待 80.9GB、上限 103GB、全トークンを残しても 107GB【計算、io レビューの実 plan】。
- 1 層 196,608 行は 1 expert 平均約 3,070 (token, expert) 対で、日本語の行は保持率 2 倍 (`--ja-boost 2`) で残る。
- 行数を 256k に上げても calib256k はほぼ全トークンを残すだけなので、192k で十分である。もっと増やすなら校正データ自体を増やす (`tools/build_calib.py --large`、ただし ja-assist などの出典の不足に注意)。

---

## 3. 段 C: L2 の全層

```bash
# expert mode (既定)。層を半分ずつ 2 枚に分け、同じ出力ディレクトリに層ごとのファイルを書く
# --img-weight / --img-guard-tol は画像入りの capture のときだけ効く (画像の無い記録ではビット単位で従来と同じ)
qrun l2a 0 "python3 qad_l2.py train --capture $Q/cap/main_p0 --capture $Q/cap/main_p1 --layers 0-19 \
   --img-weight 1.5 --img-guard-tol 0 --orig-ckpt /model --pack $V1 --out $Q/l2/main_e > $Q/l2/main_e_0.log 2>&1" &
qrun l2b 1 "python3 qad_l2.py train --capture $Q/cap/main_p0 --capture $Q/cap/main_p1 --layers 20-39 \
   --img-weight 1.5 --img-guard-tol 0 --orig-ckpt /model --pack $V1 --out $Q/l2/main_e > $Q/l2/main_e_1.log 2>&1"; wait
qcpu "python3 qad_l2.py apply --pack $V1 --diff $Q/l2/main_e --out $Q/packs/v1-l2e --mode overlay"
```

- 所要: 1 層 40〜90 s で、20 層 × 2 枚は 15〜30 分【推測、trainer の見積もり】。1 層の計算は 2.9 PFLOP (20〜37 s)、IO 12.6GB【計算、`qad_l2.py estimate --tokens 196608`】。
- GPU メモリは expert mode で約 4GiB/枚。止まっても `--skip-existing` で済んだ層を飛ばして再開できる。
- 既定のハイパーパラメータ (qad_l2.py:1003-1020): δ の lr 5e-4、L2 正則化 0.1、50〜400 step、patience 3、検証 10% (系列単位)、`--min-tokens 256`、日本語重み 2.0。fp16 に丸めた値で検証し直し、v1 より良い expert だけを採る。
- 層ごとの確認: `for f in $Q/l2/main_e/qad_l2_L*.json; do jq -r '[.layer,.accepted,.rejected,.cold,.layer_val.ja.v1,.layer_val.ja.qad,.layer_val.other.v1,.layer_val.other.qad,(.seconds|floor)]|@tsv' $f; done`。`layer_val.ja.qad` が `.v1` を上回る層 (日本語で悪化) があれば、その層の `es_metric` と cold の数を見る。
- 画像の確認 (H-VL4): `jq -c '{layer, accepted, rejected_img, img_guarded, img: .layer_val.img}'` を同じく層ごとに見る。`layer_val.img.qad ≤ .v1` が 40 層中 38 層以上であること。画像の重みと守りの意味は §1V.5。layer mode の 2 本目にも同じ引数を付ける。

**layer mode (任意、2 本目の版)**

```bash
qrun l2c 0 "python3 qad_l2.py train --capture $Q/cap/main_p0 --capture $Q/cap/main_p1 --layers 0-19 --loss layer \
   --init-diff '$Q/l2/main_e/qad_l2_L{L:02d}.safetensors' --orig-ckpt /model --pack $V1 --out $Q/l2/main_l" &   # 層 20-39 も同様に GPU1
```

- 層の MoE 出力 (top-6 の和) で合わせる。全 expert の Q̃ を GPU に置くので約 43GiB/枚。384 expert を Python で回すので、1 層 1〜3 分に延びうる【推測】。
- 補充行 (F_FLOOR) の kp が本来の保持確率のままで、layer mode では冷たい expert を通るトークンが多めに効く (未修正、capture レビュー)。`--no-kp-weight` 版と比べて、良い方を採る。
- expert mode の版より KL が良いときだけ採る。

### 3.4 日本語の重みについて

- 合成データでは、日本語重み 1→2 で日本語の誤差が 0.2306→0.2302 しか変わらなかった【実測、test_qad_l2】。suh/svh は expert あたり 22,272 値しかなく、損失の重み付けの効きは小さい。
- 日本語を守る主な手段は、capture に日本語を厚く入れること (校正 40%、保持率 2 倍) である。重みは 2.0 のままにし、日本語の KL が伸びない場合でも 4 以上には上げない (他の区分が悪くなるだけだった)。

---

## 4. 段 D: 評価

### 4.1 ハーネス (teacher-forced、1 版あたり GPU 2 枚で約 45〜50 分。画像 KL を含む)

```bash
P=$Q/packs/v1-l2e; T=l2e      # 評価する overlay パックと、出力の名前
qrun ev0 0 "for s in agent24 ja16; do python3 stream_ref.py --teacher-in /t/\$s.jsonl --teacher-ref /t/\${s}_teacher.pt \
   --teacher-out /t/\${s}_$T.pt --exl3-pack $P --max-seq-len 8192 --expert-budget-gb 20 > /t/kl_${T}_\$s.log 2>&1; done" &
qrun ev1 1 "python3 stream_ref.py --teacher-in /c/jakl32.jsonl --teacher-ref /t/jakl32_teacher.pt \
   --teacher-out /t/jakl32_$T.pt --exl3-pack $P --max-seq-len 8192 --expert-budget-gb 20 > /t/kl_${T}_jakl32.log 2>&1"; wait
qcpu "python3 /tools/kl_compare.py --set agent24 /t/agent24_v1.pt /t/agent24_$T.pt --set ja16 /t/ja16_v1.pt /t/ja16_$T.pt \
   --set jakl32 /t/jakl32_v1.pt /t/jakl32_$T.pt --jsonl jakl32=/c/jakl32.jsonl --json /t/cmp_v1_$T.json"
# 画像 (§1V.2 の教師が要る。2 枚で 15〜20 分)
for s in ja en; do g=$([ $s = ja ] && echo 0 || echo 1)
  qrun evV$s $g "python3 stream_ref_vl.py --vl-in /vl/vl_kl_$s.jsonl --teacher-ref /t/vlkl_${s}_teacher.pt \
     --teacher-out /t/vlkl_${s}_$T.pt --exl3-pack $P --max-seq-len 8192 --expert-budget-gb 20 > /t/kl_${T}_vlkl_$s.log 2>&1" & done; wait
qcpu "python3 /tools/kl_compare.py --set vlja /t/vlkl_ja_v1.pt /t/vlkl_ja_$T.pt --set vlen /t/vlkl_en_v1.pt /t/vlkl_en_$T.pt \
   --jsonl vlja=/vl/vl_kl_ja.jsonl --jsonl vlen=/vl/vl_kl_en.jsonl --ja-sets vlja --json /t/cmp_vl_v1_$T.json"
# 中国語専用トークンの質量 (テキストの ja16 / agent24 と、日本語の画像問。教師比を v1 と並べる)
qcpu "for s in ja16 agent24; do python3 /tools/zh_teacher_mass.py --teacher /t/\${s}_teacher.pt --student v1=/t/\${s}_v1.pt \
   --student $T=/t/\${s}_$T.pt --out /t/zh_\${s}_$T.json; done; python3 /tools/zh_teacher_mass.py --teacher /t/vlkl_ja_teacher.pt \
   --student v1=/t/vlkl_ja_v1.pt --student $T=/t/vlkl_ja_$T.pt --out /t/zh_vlkl_ja_$T.json"
```

- overlay は `TensorIndex` がファイル名順に読んで後勝ちになること (stream_ref.py:50, 58-62) に頼るので、ハーネス専用である (`QAD_OVERLAY_HARNESS_ONLY` の印がある)。
- `tools/kl_compare.py` (新規) は 2 版の学生採点 .pt を系列ごとに対にし、トークン平均の KL と top-1、相対変化、系列単位で区分ごとに層化した bootstrap の 95% 区間 (既定 10,000 回) を、set ごと、jakl32 の区分 (ja-agent、ja-assist、ja-chat、ja-wiki、ja-tech、ja-short、ja-aozora) ごと、日本語 (ja16 + jakl32) と非日本語 (agent24) と全体で出す (`pair` :41、`summarize` :54)。
- 確認済み【実測、CPU】: v0 の .pt 同士で agent24 0.1579 / 86.39%、ja16 0.2060 / 84.56% を再現 (QUALITY.md と一致)、KL を 0.9 倍した偽の版で −10.0%、別の系列集合との対は拒否。
- 系列は agent24 24 本、ja16 16 本、jakl32 32 本しかない。bootstrap の区間は広めに出るので、区間が 0 をまたぐ改善は「差なし」と扱う。
- 画像の採点は `stream_ref_vl.py` で、教師と学生の ids・プロンプト長・画像の sha256 が一致しないと止まる。vl_kl は 50 系列・採点 3,972 トークンと小さいので、画像の判定は「悪化していないこと」(§5 の画像関門) に限る。
- zh_teacher_mass の出力は教師比 (v1 は ja16 回答本文 2.20 倍、agent24 3.16 倍、`results/QUALITY.md`)。§5 の中国語混入関門 (H-ZH1) に使う。

### 4.2 ds41rt (生成、採用候補だけ。GPU 2 枚で 2〜3 h)

採用候補 (§5 の H を満たした版) だけを shards の版にして ds41rt で回す。

```bash
qcpu "python3 qad_l2.py apply --pack $V1 --diff $Q/l2/main_e --out $M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1-l2 --mode shards"
$D/build/stage_hf.sh $M/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1-l2 local/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-v1-l2 1017e3c200000000000000000000000000000013
```

- shards mode は 40 本の expert shard を新しい inode に実体コピーし (計約 143.5GB)、suh/svh のバイト範囲だけ書き換えて読み戻しで確かめる (qad_l2.py:883-911)。v1 の shard (work/shards と hardlink 共有) には書かない。
- apply は差分の `base_sha256` を v1 の値と照合し、合わなければ止まる (qad_l2.py:864-871)。overlay から shards を作ることは拒否される (:848)。
- ds41rt の設定 (`ext/ds41rt/local.config` の複製) で `MODEL_ID` と `MODEL_REVISION` を上の値にし、`--dry-run` を通してから起動する。v1 も同じ条件で回す (版どうしの比較)。

| 項目 | 道具 | 見るもの |
|---|---|---|
| 起動と速度 | `--dry-run`、既存の bench (c1 / c16) | 形式が同じなので v1 と同じ速度のはず。±3% を超えて変われば異常 |
| 日本語で答える率、簡体字の混入 (G-JA5 / G-JA6) | `tools/zh_gen_eval.py gen/score` (プロンプトは `tools/zh_promptset.py` で事前に作る。まだ `work/zh/promptset` に無い) | temperature 1.0 と 0.6 の両方 |
| ツール (G-TL1) | `ext/ds41rt/scripts/qualify-ds41-tool-eval.py` | 176 件 |
| DSpark の受理 (G-JA9) | ds41rt のログの 1 round あたりのトークン数 | 日本語負荷で v1 比 |
| 画像の正答率 (R-VL1、R-VL2) | `tools/vl_eval.py --out work/vl/results/$T.jsonl --compare work/vl/results/v1.jsonl` (245 問、15〜20 分) | 全体・日本語・区分ごとの対の差と 95% 区間、日本語の問いで zh_only を含む回答の件数 |
| 人手 | 日本語の自由記述 20 問を v1 と並べる。画像 10 問 (画面写し 6、写真 4) も並べる (R-VL4) | 最終候補のみ (SPEC §4.2 の判定規則) |

- ds41rt の API には logprobs が無いので、分布の比較 (KL) はハーネスでしかできない (SPEC §4.1)。生成は v1 との比較で見る。
- ハーネスと ds41rt は同時に動かせない。生成評価は候補が決まってからまとめて回す。

---

## 5. 合格基準 (v1 比)

KL は教師 top-32 上の近似 (stream_ref.py:452-458) で、比較は同じ教師・同じ系列の対で行う。

**4 つの関門**: ユーザーの要件 (日本語性能、中国語混入の抑制、画像の性能) とエージェント用途を、それぞれ 1 つの関門にする。**4 つすべてを通った版だけを採る。** 日本語だけが「改善」を求め、残りの 3 つは「v1 より悪化しない」を求める。

| 関門 | ハーネス (H) | ds41rt (R) | 求めるもの |
|---|---|---|---|
| 日本語 | H1、H3、H4 (日本語)、H5 | R2 (G-JA5)、R5 | v1 比で改善 (採否の段階は下の表) |
| エージェント | H2、H4 (agent24) | R3、R4 | 悪化しない |
| 中国語混入 | H-ZH1 | R2 (G-JA6)、R-VL2 | 増えない |
| 画像 | H-VL0〜H-VL4 | R-VL1、R-VL3 (v2 のみ)、R-VL4 | 悪化しない |
| 共通の前提 | — | R0、R1 | — |

- 画像の基準を「改善」にしないのは、校正の画像が 24 系列 (トークンの 3.6%) しかなく、L2 の狙いが日本語だからである (r6 §5)。v2 (K3 混合) にも同じ基準を使う。

**ハーネスの段 (H、すべて満たす)**

| ID | 指標 (kl_compare の出力) | 基準 |
|---|---|---|
| H1 | 日本語 (ja16 + jakl32) の KL | v1 比 −5% 以上、かつ bootstrap の 95% 区間の上端が 0 未満 |
| H2 | agent24 の KL | +1% を超えて悪化しない |
| H3 | ja16、jakl32 を別々に見た KL | どちらも悪化しない。jakl32 の区分ごと (各 2〜8 系列) は +3% を超えないこと (系列が少ないので、超えたら復号して中身を見る) |
| H4 | top-1 一致 | 日本語、agent24 のどちらも −0.3 点より下がらない |
| H5 | 日本語 KL / agent24 KL の比 | v1 の比より 0.05 を超えて大きくならない (G-JA1 の代わり。en-neutral の評価集合はまだ無い) |
| H-ZH1 | zh_only の質量 (zh_teacher_mass、回答本文、教師比) | ja16 と agent24 の点推定が v1 (2.20 倍 / 3.16 倍) を超えない。日本語の画像問 (vlkl_ja) も v1 の値を超えない |
| H-VL0 | (前提) 量子化なしの学生で vl_smoke を採点 | KL 0.0000、top-1 100%、own / swap の対照で画像が効いている (§1V.1) |
| H-VL1 | 画像系列 (vlja + vlen の ALL) の KL | 点推定で v1 比 +1% 以内、かつ bootstrap の 95% 区間の下端が 0 を超えない (有意な悪化が無い) |
| H-VL2 | 同じく top-1 一致 | v1 から −0.3pt より下がらない |
| H-VL3 | 日本語の画像問 (kl_compare の JA = vlja) の KL | v1 比 +2% 以内 |
| H-VL4 | L2 の層ごとの報告 `layer_val.img` | 40 層中 38 層以上で `qad ≤ v1`。`rejected_img` の合計が `accepted` の 10% を超えたら §1V.5 の再学習を検討 |

- 画像の KL は 50 系列・3,972 トークンしかなく、区間は広い。H-VL1 の「区間の下端が 0 を超えない」は、はっきりした悪化だけを止める条件である。英語 (878 トークン) と区分ごと (sub) の値は判定に使わず、+10% を超えた区分は復号して中身を見る。
- pilot の版 (テキストだけの capture) で H-VL1 を落とした場合、本番 (画像入り capture + 守り) で通るかを必ず確かめる。本番でも落ちるなら、`--img-guard-tol` を負 (例 −0.02、画像で 2% 以上良くならない expert は採らない) にするか、画像の重みを 2.0 に上げて学習し直す。それでも落ちれば不採用。

**採否の段階**

| 日本語 KL の改善 (H1 の点推定) | 判定 |
|---|---|
| 10% 以上 | 強い採用。ds41rt の段を通せば v1 を置き換える |
| 5〜10% | 採用。ds41rt の段を通せば置き換える |
| 2〜5% | 条件付き。推論の費用は 0 なので、H2〜H5 と ds41rt の段をすべて満たし、区間が 0 をまたがないときだけ採る |
| 2% 未満、または区間が 0 をまたぐ | 不採用。§6 の L3 判定に回す |

**ds41rt の段 (R、採用の確定条件)**

| ID | 基準 |
|---|---|
| R0 | §1.4 のカナリアで、svh が clamp の前に掛かっていることを確かめてある |
| R1 | `--dry-run` を通り、c1 / c16 の速度が v1 の ±3% 以内 |
| R2 | 日本語で答える率 99% 以上 (G-JA5)、簡体字専用字が 1 字以上 1.0% 以下・3 字以上 0.3% 以下 (G-JA6)、どちらも v1 より悪化しない |
| R3 | ツール評価 150/176 以上で、v1 から 3 件を超えて減らない (G-TL1) |
| R4 | 日本語負荷での DSpark の 1 round あたりのトークン数が v1 の −5% 以内 (G-JA9) |
| R5 | 人手の 20 問で、v1 より明らかに悪い回答が無い |
| R-VL1 | `vl_eval.py --compare` で v1 と対にした画像正答率 (245 問)。全体: 点推定 −1pt 以内、かつ対の 95% 区間の上端が 0 以上 (有意な悪化が無い)。日本語 (145 問): 点推定 −2pt 以内、かつ区間の上端が 0 以上 |
| R-VL2 | 日本語の画像問で zh_only を含む回答の件数が v1 より増えない |
| R-VL3 | (v2 の K3 混合のときだけ) 画像付き要求 (1024×768 を 1 枚、3000×2000 を 1 枚) の prefill 時間が v1 の ±5% 以内。L2 はパックの形式が同じなので R1 で足りる |
| R-VL4 | 人手で画像 10 問 (画面写し 6、写真 4) を v1 と並べ、明らかに悪い回答が無い |

- R-VL1 の問題数【計算】: 対にした正誤の差の標準誤差は、v1 と答えが分かれる問いの割合を d として √(d / n) である。d = 0.10、n = 245 で 2.0pt、95% 区間は約 ±4pt になる。したがって 245 問では −1pt の差は見分けられず、「点推定 −1pt 以内 + 有意な悪化なし」で読む。区分ごと (20〜50 問、1 問で 2〜5pt 動く) は判定に使わず、−5pt を超えて下がった区分は不一致の問いを目で見る。−1pt を判定したいなら、r6 §5 のとおり 1,000 問以上に増やす (§9 のユーザー判断)。
- vl_gen の自作スクショと図表は、校正 (vl_calib) と同じテンプレートを使っている (レビューの指摘、未修正)。QAD 版は自作区分で甘く出うるので、R-VL1 は公開データの区分 (日本語: mecha、jmmmu、javg、ccocr / 英語: chartqa、docvqa、textvqa、mmbench) だけの値も並べて見る。

- 期待値【推測】: QuIP# の 70B 2bit では微調整で誤差の約 24% が戻ったが、他のパラメータの学習も含む値である。suh/svh だけの局所蒸留では、日本語 KL の −5〜−15% を見込む。合成データの層出力の誤差が 0.19→0.024 まで下がった結合テストの値は、実データには当てはまらない。

---

## 6. L3 (全体 KL の suh/svh 学習) に進む条件

L3 は `harness/qad_l3.py` と設計 (`harness/qad_l3_design.md`) があり、CPU の selftest は 48/48 通過している。
GPU での G0〜G8 (設計 §9) は未実行である。

**進む条件 (いずれか 1 つを満たし、かつ前提をすべて満たす)**

1. L2 を採用したが、日本語 KL / agent24 KL の比が 1.2 を超える (設計 §8 (2))。
2. L2 が「条件付き」または「不採用」で、pilot の層ごとの報告では層出力の誤差が下がっている (局所の目的と logits の目的がずれている疑い)。
3. K3 混合 (v2) に VRAM を使い切っても日本語のゲートを満たさない (追加の VRAM が要らない最後の手段、設計 §8 末尾)。

**前提**

- §1.4 のカナリアが通っている (通らないと ds41rt に効かない)。
- G4: θ=0 の L3 の評価が、同じパックのハーネス採点を KL ±0.002 (か ±1%) と top-1 ±0.3 点で再現する。届かなければ `gemm_dtype` を fp16 にする (math レビューの未修正点)。
- 日本語の判定規則を capture_moe の `tag_tokens` (窓 16、仮名 3 字以上) に揃える。今は L3 が 32 トークンの塊で判定している (qad_l3.py:771 `classify_ja`、io レビューの未修正点)。
- 教師: calib256k を区分ごとのファイルに分け (L3 は `--teacher` のファイル名の先頭を区分名に使う)、`score_from=1` で全位置の top-32 を取る。79 系列で 2 枚約 40 分、出力は 1 位置 192B で約 50MB【計算】。設計の 1M トークンに増やすなら `build_calib.py --large` を上げる (出典の不足に注意)。
- 画像: `qad_l3.py` は画像に未対応である (教師の系列に images / token_types を通していない。r6 §6)。L3 は全体 KL で全 expert の suh/svh を動かすので、テキストだけの教師で学習すると L2 より強くリスク 1 (§1V) が出る。進む前に `microbatch_from_record` に images / token_types を通す改修を足し、vl_calib の教師 (stream_ref_vl.py で取る) を `--teacher` に入れる。改修しないで進む場合は、§5 の画像関門 (H-VL1〜3、R-VL1、R-VL2) を L3 の版にも掛け、落ちたら採らない。
- GPU 2 枚を約 5〜7 h 続けて使える (教師 1〜2 h、学習 1〜4 h、評価)。
- ディスクに export 用の 143.5GB がある (§7)。

**手順**: 設計 §9 の G0〜G8 を順に行い、G7 の pilot (64 系列、約 25 万トークン、1 epoch) で区分平均の KL が 5% 以上下がり、どの区分も 1% を超えて悪化しなければ本番に進む。2% 未満なら打ち切る。
L2 を採用した場合は、L2 の overlay (`$Q/packs/v1-l2e`) を `--pack` にして L2 の上に学習する (`SafetensorsIndex` は weight_map を優先して読むので overlay を読める)。export は L2 の shards の版か v1 に当て、`base_sha256` の照合を通す (qad_l3.py:1586 `export_main`)。

**本番の採用**: 設計 §8 (4) のとおり、L2 の版 (無ければ v1) 比で区分平均の KL が 10% 以上改善 (SPEC X-7)、bootstrap の 95% 区間が 0 をまたがない、どの区分も 3% を超えて悪化しない、§5 の R1〜R5。

---

## 7. ディスクの使い方

**現状**【実測、2026-10-18 15:00 前】: `/` の空き 210GB。v1 は 33/40 shard で、残り 7 本 (約 25GB) を書くと約 185GB になる。
v1 の組み立て (`assemble_v1.sh`) と `stage_hf.sh` は hardlink だけなので増えない。

| 時点 | 増減 | 空き (目安) |
|---|---|---|
| v1 完成 | −25GB | 185GB |
| 段 A: smoke capture、カナリア shard | −3.4、−3.6GB (済んだら消す) | 178GB → 185GB |
| 段 V: 画像の smoke capture、画像の教師・学生の .pt | 約 −1GB (smoke capture、済んだら消す)、.pt は数十 MB | ほぼ変わらない |
| 段 A: pilot capture 2 本 | −52GB | 133GB |
| pilot 判定後、pilot capture を消す | +52GB | 185GB |
| 段 B: capture 本番 | −81〜103GB | 82〜104GB |
| 段 C: overlay (版ごと) | −0.7GB | 80〜103GB |
| **分岐 1**: L2 を採用し、shards の版を作る | −143.5GB | capture を残すと足りない |

**分岐 1 の選び方**

- capture をもう使わない (ハイパーパラメータの探索も layer mode も済んだ) なら、capture を消してから shards を作る: 185 → 41GB。MiMo / GLM は消さない。
- capture を残して ds41rt の結果を見てから再学習したいなら、**このとき MiMo-V2.6-Flash-RL (166GB、hardlink 無し) を消す**: 82〜104 + 166 → shards 後 105〜127GB。

**L3 に進むとき**: export の版が +143.5GB 要る。L2 の shards の版 (41GB 残り) と並べるには、MiMo をまだ消していなければここで消す (+166GB → export 後 約 64GB)。
**GLM-5.3-Flash-W4A16** (166GB) は、L3 の版と L2 の版と capture を同時に持ちたいとき、または K3 混合 (v2: K3 候補 82GB + snapshot 150GB、SPEC §5.2 X-5) に進むときに消す。GLM-5.3-Flash-FP4-Local は削除の許可が無いので触らない。

**消すときの注意**

- MiMo と GLM-W4A16 の大きなファイルは hardlink 数 1 で、消せばそのまま空く【実測、`stat -c %h`】。
- v0 (`DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw`) の expert shard (00003〜00042、約 143GB) は v1 から参照されていないが、`dsv41/hf/hub/models--local--DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw` の snapshot と hardlink を共有している (hardlink 数 2【実測】)。空けるには両方を消す必要がある。v0 の非 expert shard (Engram を含む) は v1 と共有しているので、v0 のディレクトリを消しても v1 は壊れないが、空く量は expert shard の分だけである。v0 を消すかどうかはユーザー判断 (比較の基準は v1 に移るので、評価上は不要)。
- 負けた版は、評価が済んだら shards の版と hf の snapshot の両方を消す。
- capture と .pt は root 所有なので `sudo rm -rf`。

---

## 8. 全体の時間割 (GPU 2 枚、目安)

| 順 | 段 | GPU0 | GPU1 | 壁時計 | 状態 (10-18 17:50) |
|---|---|---|---|---|---|
| 1 | §1.0 確認、v1 組み立て | — | — | 10 分 | 済 |
| 2 | §1.1 / §1.2 / §1.3 | v1 採点 (agent24、ja16) | jakl32 教師 → check-decomp、smoke capture、check-teacher、試し学習 → v1 採点 (jakl32) | 約 1 h | 済 |
| 3 | §1.5 pilot の capture | 学生 stream | 教師 stream | 約 20〜30 分 | 実行中 |
| **4** | **§1V.0〜1V.2、1V.4 画像の smoke と基準** | vl_kl_ja の教師 → v1 / v0 の学生 (ja) | ViT smoke + 対照 → 画像入り capture の smoke → vl_kl_en の教師 → v1 / v0 の学生 (en) | **約 1.2 h** | 未 |
| 5 | §1.5 pilot の学習と評価 | 学生 stream の学習 → 評価 | 教師 stream の学習 → 評価 | 約 40 分 + 評価 1.5 h (画像を含む) | 未 |
| 6 | §1.4 カナリア + **§1V.3 ds41rt の画像の基準** | ds41rt (2 枚): v1 で形式確認・vl_eval (約 20 分) → カナリア | | 約 1.3 h | 未 |
| 7 | §2 capture 本番 (vl_calib を足す) | part 0 | part 1 | 30〜50 分 (+1〜2 分) | 未 |
| 8 | §3 L2 全層 (画像の重みと守り) | 層 0〜19 | 層 20〜39 | 15〜30 分 (layer mode を足すと +30〜90 分) | 未 |
| 9 | §4.1 ハーネス評価 | agent24、ja16、vl_kl_ja | jakl32、vl_kl_en | 約 45〜50 分/版 | 未 |
| 10 | §4.2 shards と ds41rt | CPU 5〜10 分 + ds41rt 2〜3 h (v1 と候補、画像正答率を含む) | | 2.5〜3.5 h | 未 |
| 11 | §6 L3 (条件付き、画像の改修が前提) | stage0 | stage1 | 5〜7 h | 未 |

- 画像の基準 (順 4) を pilot の学習 (順 5) より先にする理由は 3 つある。(1) ViT が SM120 で動かなければ、段 B の画像入り capture の手順が変わる。(2) pilot の評価に画像 KL を入れるには、教師と v1 の値が先に要る (pilot はテキストだけの capture なので、リスク 1 を実データで測る最初の機会である)。(3) v1 の画像性能はまだ一度も測っていない。
- 順 4 と順 5 は入れ替えてもよい (pilot の結果を早く見たいとき)。その場合も、pilot の評価の前に順 4 の V.2 (教師と v1) を済ませる。
- ds41rt の画像の基準 (順 6 の前半) は、ds41rt が運用で v1 を出している時間ならいつ流してもよい。GPU の作業と同時にはできない。
- L3 を除いて、GPU を約 11〜14 h 使う (画像の分で約 2〜2.5 h 増えた)。段 A + V (順 1〜6) と段 B〜D (順 7〜10) は別の日に分けてもよい。capture と差分はディスクに残るので、途中から再開できる。

---

## 9. 未確認の点とユーザー判断

**GPU で確かめるまで分からないこと**

- 実データで L2 が KL をどれだけ下げるか (pilot で最初に分かる)。
- torch の教師の再現が fp4_gemm と一致するか (§1.2 の check-teacher)。一致しないうちは学習の目標が正しいと言えない。
- ds41rt で svh が clamp の前に掛かっているか (§1.4)。
- 学生 stream と教師 stream のどちらが良いか (§1.5)。学生 stream も v1 の stream で、L2 で直した前段の層は反映されない (層を順に直しながら stream を進める方式は未実装)。
- 1 step あたりの Python と起動の手間 (expert mode の時間が 2 倍程度ずれうる)。
- 画像の経路が GPU で動くか: ViT 32 層 (SDPA、BF16) と aligner が SM120 + stream_ref の build 環境で走るか、capture_moe の実モデル経路 (CPU テストは偽モデルだけ) が通るか (§1V.1、§1V.4)。
- 層 1〜39 の画像の routing の偏り (r6 の近似は層 0 だけ)。テキストだけの L2 が実際に画像を悪くするか、守りがどれだけの expert を落とすか (pilot と段 C で初めて分かる)。
- ds41rt の画像の縮小規則とテンプレートがハーネスと一致するか (§1V.3 の prompt_tokens 458)。ds41rt の画像 API は HTTP では一度も確かめていない (ソースだけ)。

**ユーザーに決めてほしいこと**

1. GPU を続けて使ってよい時間帯 (段 A に約 4 h、段 V に約 1.2 h + ds41rt 約 20 分、段 B〜D に約 6〜9 h、L3 にさらに 5〜7 h)。
2. 採用の基準 (§5): 日本語 KL −5% で採用、−2〜5% は条件付き、でよいか。
3. MiMo を消すタイミング (§7 の分岐 1)。capture を残して再学習の余地を持つなら、shards を作る時点で消す。
4. v0 の expert shard と hf snapshot を消すか (143GB 空く。v0 との比較はできなくなる)。
5. jakl32 と agent24 / ja16 の内容の重なりを 30% まで許している点 (評価集合どうしなので学習への漏れではない。厳しくするなら `build_calib.py --jakl-dup-teacher`)。
6. **要求ログの実画像を校正に使うか**。GLM の要求ログに画像付き要求が 288 件 (画像 105 種、304MB、PNG の画面写しが中心) ある。中身と問いの文は見ていない (r6 §7)。使えば画面写しの分布に最も近いが、個人データを含みうるので承認が要る。使わなければ公開データと合成画像 (vl_calib 24 系列) のままにする。
7. 画像の生成評価を 245 問から 1,000 問以上に増やすか。今の 245 問では −1pt の差を見分けられない (§5 の R-VL1 の注)。増やすと ds41rt の評価が 1 版あたり +30〜60 分になる。
8. ds41rt を v0 で再起動して、v0 の画像正答率も取るか (基準の感度を見るためで、QAD の比較には不要)。
9. `--img-weight` の既定 (1.0 のまま計画のコマンドで 1.5 を指定する案にした) と、K3 混合 (v2) の π_img (r6 §4 の案 0.20、日本語 0.30 は据え置き)。

---

## 参照

- 前提: `research/SPEC_QUANT_JA.md` (§2.5 L2 の仕様、§4.2 ゲート、§7 反証レビュー)、`research/r3_qad.md` (§2.2、§4)、`research/r3_japanese.md`、`results/QUALITY.md`、`research/r3_routing_skew_measured.md`。
- ツール: `harness/capture_moe.py`、`harness/qad_l2.py`、`harness/qad_l3.py`、`harness/qad_l3_design.md`、`tools/build_calib.py`。
- この計画で足したもの: `tools/kl_compare.py` (版の対比較と bootstrap)、`harness/svh_canary.py` (ds41rt の svh の位置の検査用差分)。どちらも既存のファイルは変えていない。
- 画像: `research/r6_vision_qad.md` (routing の分析、改修、配分と合格基準の案)、`research/arch_spec.md` (vision 節)、`harness/stream_ref_vl.py`、`harness/vl_rows.py`、`tools/vl_eval.py`、`tools/vl/build_sets.py`、`work/vl/` (vl_kl / vl_gen / vl_calib、`sources.json` に出典とライセンス)。§1V で `work/vl/vl_kl_ja.jsonl`、`vl_kl_en.jsonl`、`vl_smoke.jsonl`、`vl_smoke_ctrl.jsonl` を足した (既存の行から作ったもので、元のファイルは変えていない)。
