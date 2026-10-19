# quant: 量子化と scale 校正の道具

DeepSeek-V4.1-Flash-1017 の routed experts を EXL3 2.0bpw にし、scale 校正 (expert ごとの scale だけを層ごとに教師の出力へ合わせる局所蒸留) で合わせ直したときの道具一式。
スクリプト名やメモに出てくる QAD は、ここでは scale-only の局所蒸留のことで、重みや離散コードは学習せず、モデル全体での end-to-end の蒸留もしていない。
公開している重みのパックは、ここの手順で作った (v0 → v1 → v1-l2e)。

**手元で使った形のまま入れてある。** 作業場の決め打ちのパスは環境変数に置き換えたが、それ以外は手を入れていない。
私的なデータを前提にした道具も含むので、使うときはデータを各自で用意する (下の「データ」)。

## 置き方 (作業場)

スクリプトは「作業場」のルート (`DSV41`) の下に次の形で物がある前提で書いてある。このディレクトリ (`quant/`) が作業場のルートに当たる。

```
$DSV41/                 (= quant/)
├── harness/            参照ハーネス・捕捉・scale 校正 (このリポジトリ)
├── tools/              評価・校正データ・調査の道具 (このリポジトリ)
├── build/              量子化・scale 校正・評価の手順 (このリポジトリ)
│   └── qrecipe/tools/  EXL3 の量子化器 (sfxnz / diffbot のレシピ由来)
├── ref/                公式の参照実装 (各自で用意: 公式 HF repo の inference/ の中身)
├── work/               データと中間生成物 (各自で用意。下の「データ」)
└── models/             重み (MODELS。既定は $DSV41/models)
```

| 環境変数 | 既定 | 内容 |
|---|---|---|
| `DSV41` | Python: カレントディレクトリ。シェル: スクリプトの 1 つ上 (`quant/`) | 作業場のルート |
| `MODELS` | `$DSV41/models` | 重みの置き場。公式 1017 は `$MODELS/DeepSeek-V4.1-Flash-1017` |
| `GLM_REQLOG` | `reqlog` | 要求ログの解析の道具 (`x2_*`、`r3ja/`、`zh_glm_eval.py`、`zh_promptset.py`) が読む要求ログ。公開していない |
| `R3JA_RULES` | (空) | `tools/r3ja/r3ja_join.py` の `cat()` (要求の区分) の判別規則の JSON。公開版はクライアント固有の規則を含まないので、自分の要求ログに合わせて書く (`cat()` の説明を参照) |
| `EXTRA_TECH_GLOBS` | 空 | `build_calib.py` の技術文書に足す glob (`:` 区切り) |
| `DSV41_REF` / `DSV41_CKPT` | `/ref` / `/model` | ハーネスがコンテナの中で読む参照実装と原重み |

## コンテナ

GPU の仕事はすべて docker の中で動かす (手順では `sudo docker`)。

- `dsv41-quant:151`: LIL の vLLM イメージ `ghcr.io/local-inference-lab/vllm:release-10152026` に exllamav3 1.5.1 (`958ec933`) を入れたもの。
  手元では次のように作って `docker commit` した。

  ```sh
  git clone https://github.com/turboderp-org/exllamav3 && git -C exllamav3 checkout v1.5.1
  sudo docker run --name q151 --gpus all -v $PWD/exllamav3:/src/exllamav3:ro --entrypoint bash \
    ghcr.io/local-inference-lab/vllm:release-10152026 -c 'cp -r /src/exllamav3 /opt/exllamav3 && cd /opt/exllamav3 &&
      MAX_JOBS=16 TORCH_CUDA_ARCH_LIST=12.0a pip install --no-build-isolation --no-deps . &&
      pip install --no-deps rich marisa_trie pydantic pyyaml'
  sudo docker commit q151 dsv41-quant:151
  ```
- `dsv41-quant:5be8`: 同じ作り方で exllamav3 `5be88657` (1.4.9) を入れたもの。v0 の量子化にだけ使った。

## 手順 (公開パックができるまで)

| 段 | スクリプト | 内容 |
|---|---|---|
| v0 | `build/quantize_1017.sh` | routed experts を EXL3 K2 MCG にする (無校正、greedy beam 16、exllamav3 1.4.9)。`build/qrecipe/tools/quantize_experts_exl3_sm120.py` を当機向けに呼ぶ。比較の基準 |
| v1 | `build/requant_v1.sh`、`build/assemble_v1.sh` | 原本から Viterbi (exllamav3 1.5.1) で再符号化し、suh/svh を閉形式で合わせ直す (H=I)。v0 を基準に shard 単位で検査する |
| HF キャッシュに並べる | `build/stage_hf.sh` | パックを ds41rt が読む HF キャッシュの形に hardlink で並べる |
| 教師 | `harness/stream_ref.py`、`harness/stream_ref_vl.py` | 原重み (MXFP4 experts、FP8 Engram) を層ごとに流す参照ハーネス。1 枚の GPU で教師の logits と routing を出す。学生 (EXL3 パック) の採点もする |
| 段 A | `build/qad_stageA.sh` | 分解の検査 (`qad_l2.py check-decomp`)、捕捉の検査、教師の検査、試し学習 |
| 段 V | `build/stageV.sh` | 画像入り系列の教師と学生の採点、画像入りの捕捉の確認 |
| pilot | `build/qad_pilot.sh` | 64K トークンで L2 を 40 層回す |
| 本番 | `build/qad_main.sh` | `capture_moe.py run` (校正 256K + 画像 24 系列、2 GPU で分割) → `qad_l2.py train` (層 0-19 と 20-39 を 2 GPU で) → `qad_l2.py apply` → 評価 → `kl_compare.py` と `zh_teacher_mass.py` で比べる |
| ds41rt で確認 | `build/ds41rt_eval_l2e.sh`、`build/bench_conc.py`、`build/bench_prefill.py`、`tools/vl_eval.py` | 付属ベンチ、同時 1〜16 本の速度、prefill、画像の正答率、生成の中国語混入 |

`qad_l2.py apply --mode shards` は、差分のある層の shard だけを実体にコピーして suh/svh のバイト範囲を書き換え、他の shard は hardlink にする。公開パックはこの形 (`v1-l2e-shards`) である。
`--mode overlay` はハーネス専用で、ds41rt には読ませない。

### L2 scale 校正 (QAD L2) の中身

- 学生の各 expert は `W = diag(su) · H_in · Q · H_out · diag(sv)` (Q は EXL3 の trellis を su=sv=1 で復元した内側の行列、H は 128 次のブロック Hadamard)。
- 学習するのは各 expert の w1/w3/w2 の `suh` / `svh` (22,272 値/expert) だけで、`su = su_v1 · (1 + δ)` の δ を学ぶ。離散コード (trellis) は固定する。
- 教師は原重み (MXFP4) の routed expert の出力。入力は `capture_moe.py` で記録した MoE の入力と routing で、量子化済みの前段を通したもの (`--exl3-pack` で学生の stream)。
- トークンの重みは `重要度 × (日本語なら ja_weight) × (画像なら img_weight)`。本番は `--img-weight 1.5`。
- 画像の検証誤差が v1 より悪くなる expert は採らない (`--img-guard-tol 0`)。本番では 15,360 本のうち 14,582 本を採った (不採用 51、画像ガードで不採用 154、学習用のトークンが 256 未満で v1 のままにしたもの (cold) 573)。
- 設計の詳細は [`docs/research/QAD_PLAN.md`](../docs/research/QAD_PLAN.md)、全体 KL の L3 の設計は [`harness/qad_l3_design.md`](harness/qad_l3_design.md) (L3 は骨格だけで、公開パックには使っていない)。

## データ (公開していないもの)

| 置き場 | 中身 | 用意のしかた |
|---|---|---|
| `ref/` | 公式の参照実装 (`model.py`、`kernel.py`、`engram.py`、`generate.py` など) と `headers.json` | 公式 HF repo の `inference/` をそのまま置く。`headers.json` は各 shard の safetensors のヘッダを並べたもの |
| `work/calib/` | 校正と蒸留のデータ (`calib256k.jsonl`、`jakl32.jsonl` など) | `tools/fetch_ja_corpus.py` で公開コーパス (Wikipedia ja、青空文庫、oasst2-ja、dolly-ja) を取り、`tools/build_calib.py` で組む。**手元の版はエージェント部分に私的な要求ログ (`work/x2/req`) を使った** |
| `work/teacher/` | 教師の出力 (`*_teacher.pt`) と評価用の系列 (`agent24.jsonl`、`ja16.jsonl`) | `stream_ref.py --teacher-out` で作る。`agent24` と `ja16` は私的な要求ログ由来なので公開しない |
| `work/vl/` | 画像の評価セットと校正用の画像系列 | `tools/vl/fetch_raw.py`・`fetch_textvqa.py` で取り、`tools/vl/build_sets.py` で組む。出典ごとのライセンスは `work/vl/sources.json` に書き出される。DocVQA (研究目的に限る)、ChartQA (GPL-3.0) などは再配布しない |
| `work/x2/`、`work/r3ja/`、`work/zh/` | 要求ログの解析の中間生成物 | 私的な要求ログ (`GLM_REQLOG`) から作る。公開しない |

## 道具の一覧

| ファイル | 内容 |
|---|---|
| `harness/stream_ref.py` | 層ストリーミングの参照ハーネス。experts を必要な分だけ GPU に読み、Engram は pread する。教師の logits / 学生の採点 / greedy 生成 |
| `harness/stream_ref_vl.py`、`harness/vl_rows.py` | 画像入り系列の版と、画像入りの行の形式 |
| `harness/capture_moe.py` | 教師の MoE の入力と routing を層ごとに記録する (`run` / `verify` / `info` / `plan` / `pack`) |
| `harness/qad_l2.py` | L2 scale 校正 (`check-decomp` / `check-teacher` / `estimate` / `train` / `apply`) |
| `harness/qad_l3.py`、`qad_l3_design.md` | 全体 KL での scale-only QAD の骨格と設計 (未使用) |
| `harness/svh_canary.py` | ds41rt が svh を clamp の前に掛けているかを確かめる差分を作る |
| `harness/steer_*.py` | 言語方向の推定と、量子化のどこが中国語を増やすかの切り分け (調査用) |
| `harness/test_*.py` | CPU だけで回る試験 (GPU・CUDA 拡張・実重みは使わない。`ref/` と原重みの config・tokenizer は使う) |
| `tools/kl_compare.py` | 2 版の学生採点を系列ごとに対にして、KL と top-1 の変化をブートストラップの区間付きで出す |
| `tools/zh_detect.py`、`zh_teacher_mass.py`、`zh_gen_eval.py`、`zh_serve_mask.py` ほか `zh_*` | 中国語の混入の判定器、teacher-forced の中国語専用トークンの質量、生成の採点、配信時のマスク |
| `tools/vl_eval.py`、`tools/vl/` | 画像の評価セットの取得・組み立て・採点 |
| `tools/build_calib.py`、`fetch_ja_corpus.py` | 日本語を厚くした校正セットの組み立て |
| `tools/r2_*`、`r3_*`、`r6_*`、`x2_*`、`x5_*`、`engram_sim.py`、`r3ja/` | 調査で使った解析 (routing の偏り、Engram の当たり率、ビット配分、日本語の比率など)。多くは私的な要求ログか中間生成物を読む |
| `build/qrecipe/tools/` | EXL3 の量子化器と周辺。sfxnz のレシピの `tools/` (`3de9146`) をそのまま写し、`pack_meta.py` を手直しした。`quantize_experts_exl3_sm120.py` は diffbot のレシピの `recipe/quantize/` のもの |
| `build/serve_vllm_exl3.sh`、`build_vllm_exl3_prestage.sh` | 比較に使った vLLM + vllm-exl3 の起動とイメージ作り (外部の diffbot / nktlabs のレシピを読む。ここには含めていない) |

## 試験

```sh
# CPU だけ (GPU を割り当てない)
sudo docker run --rm --network none -e CUDA_VISIBLE_DEVICES= \
  -v $DSV41/ref:/ref:ro -v $DSV41/harness:/h:ro -v $MODELS/DeepSeek-V4.1-Flash-1017:/model:ro \
  --entrypoint bash dsv41-quant:151 -c 'cd /h && python3 test_qad_l2.py && python3 test_qad_integration.py'
python3 tools/test_build_calib.py
```

各試験の先頭の docstring に、必要なマウントが書いてある。
