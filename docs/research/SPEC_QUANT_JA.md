# 日本語性能を守る 1017 量子化の仕様書

作成日: 2026-10-18
対象: `deepseek-ai/DeepSeek-V4.1-Flash-1017` の routed experts (40 層 × 384、1 expert = 3 × 5120 × 2304 = 35.39M 重み、計 543.6B 重み、MXFP4 288.8GB)
実行先: ds41rt (MIT) の 0-spark 構成 (2x RTX PRO 6000 Max-Q、experts を全層 VRAM 常駐、TP2)
入力資料: `PLAN.md`、`DECISION.md`、`r2_quant_tools.md` (以下 QT)、`r2_ds41rt.md` (DR)、`r2_lil_lowbit.md`、`r2_engram_hit.md` (EH)、`arch_spec.md`、`r3_qad.md` (QAD)、`r3_dynamic.md` (DYN)、`r3_tensorfold.md` (TF)、`r3_japanese.md` (JA)、`r3_routing_skew_measured.md`
凡例: 【実測】は手元で確かめた値、【資料】は出典 (ファイル:行か URL) 付きの値、【計算】は式から出した値、【推測】は根拠が間接的な値。
この文書の作成では GPU を使っていない。docker コンテナには触れていない。重みは取得していない。

版の呼び方 (以下この名前で通す)

- **v0**：いま作成中の一律 K2 MCG、校正なし (identity Hessian)、greedy beam16。出力は `$MODELS/DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw`。DECISION の E3-0 に当たる。基準版とする。
- **v1**：v0 と同じ形式で、tail-biting Viterbi + scale refit (H=I) で符号化し直した版 (L0、DECISION の E3-1)。
- **v2**：v1 を土台に、expert 別に K2/K3 を混ぜた ds41rt staged 版 (静的混合精度、DECISION の E3-3)。
- **v3**：日本語を厚くした校正データで Hessian 校正 (L1) と層ごとの局所蒸留 (L2) をかけた混合版 (DECISION の E3-2 と E3-3 の合成)。

---

## 0. 結論 (ユーザーへの回答)

### 0.1 QAD は可能か

本格的な QAD (全重みを学習する end-to-end 蒸留、数億〜十億トークン) は、この機材では不可能である。
GLM-5.3-Flash-FP4-Local の QAD 規模 (500M トークン × 3 epoch) を当機で回すと 80 日を超え【計算、QAD §2.1】、V4.1-Flash の LoRA でさえ NeMo のレシピは H100 64 枚を前提にしている【資料、QAD §1.1】。
一方、EXL3 の離散コードを固定して連続成分だけを合わせる「軽い QAD」は数時間で回せる。
具体的には L0 (Viterbi 再符号化 + refit、データ不要、GPU 3〜6 h)、L1 (日本語を含む実データの Hessian 校正)、L2 (層ごとに suh/svh を教師の MoE 出力へ局所蒸留) の 3 段を採り、L1 と L2 は同じ pass で GPU 4〜8 h かかる。
学習した suh/svh (計 0.68GB) は ds41rt が checkpoint の値のまま読むので、推論の追加費用は無い【資料、`v41_exl3_residency.rs:233-290`】。

### 0.2 動的量子化は可能か

「expert ごとにビット数を変える静的な混合精度」(Unsloth Dynamic と同じ発想) は可能で、これを本線にする。
ds41rt の staged manifest は projection ごとに K を持ち、隣り合う 2 段 (K2/K3) を 1 回の起動で混ぜられる【資料、`v41_exl3.rs:440-466`、`v41_exl3_residency.rs:197-209`】。
平均 2.19bpw (K3 は約 2,960 本) で代理損失は一律 K2 の 0.40 倍程度になる見込みである【計算 + 推測、DYN §2.4】。
実行時に精度を切り替える方式 (高ビット版の差し替え、残差の追加) は、PCIe x8 と RAM 128GB の制約で静的配分に負けるので採らない (同じ 17GB の枠で静的 0.33 に対して動的 0.41〜0.58【計算、DYN §3.1】)。

### 0.3 Tensorfold をどう扱うか

"Tensorfold" という名前の圧縮手法は見つからず、実在するのは推論エンジン TensorFold (ashhart/TensorFold) とその V4.1 向け fork である【資料、TF §1】。
名前から連想される「折りたたみ」型の圧縮 (expert 間の共有基底、共通成分 + 差分、テンソル分解、統合) は 1017 の実データで測ったところ得が 0.001〜0.04 bit/重みしかなく、共有基底を持つ費用 (0.01〜0.06bpw) と速度低下 (c1 最大 −39%) に見合わないので採らない【実測 + 計算、TF §2〜3】。
TensorFold エンジンも RoCE 専用の通信と同時 4 系列の制約で ds41rt の代わりにならない。
流用するのは expert ごとの混合幅 EXL3 という表現と Krishando 方式の校正手順で、本仕様の manifest (§2.3、§2.9) はそれと相互に変換できる形にする。

### 0.4 日本語をどう守るか

守るのは「日本語の指示を読んでコードとツールを動かす力」と「最後に返す日本語の回答」の 2 つである (要求の 92.2% が日本語の発話、出力トークンでは日本語の回答本文が 14.0%【実測、JA §2.2】)。
手段は 4 つあり、(a) 校正データの 40% を日本語が主の系列にする、(b) 日本語系列で数えた routing 頻度をビット配分に入れ (π_ja = 0.30)、日本語で熱い experts を K3 にする、(c) 局所蒸留の損失で日本語トークンの重みを 2 倍にする、(d) 日本語と英語を分けて評価し、日本語の KL が英語の 1.3 倍以内などの基準 (G-JA1〜10) で合否を決める。
K1 は使わず、Engram は FP8 のまま変えない。
教師に合わせる量子化と蒸留で元の 1017 を超える日本語性能は原理的に得られないので、「向上」は一律量子化と比べた劣化の縮小と、運用設定 (回答言語の指示、reasoning effort の写像) の範囲で狙う。

### 0.5 最初にやる実験 3 つ

1. **X-1 日本語の教師と区分別 routing の取得**：GPU が 1 枚空き次第、ハーネスで ja16 と日本語評価セット (JA-KL 70 本、MCQ 123 本) の教師を取り、区分別の routing 回数を保存する (GPU 1 枚で約 3 h、ja16 の routing だけなら 15〜25 分)。配分、校正、評価のすべてがこれに依存する。
2. **X-2 v0 の区分別ベースライン評価**：v0 の完成後、ハーネスの 1 段目 (約 3 h、1 枚) と ds41rt の 2 段目 (約 2〜3 h、2 枚) を回し、日本語と英語の劣化の比を初めて測る。
3. **X-3 L0 (v1) の作成と v0 との比較**：`--greedy` を外した Viterbi + refit で v1 を作り (2 枚で 6.2 h、exllamav3 1.5.1 なら 3.1 h【計算、QAD §3.1】)、同じ評価で v0 と比べる。形式もバイト数も変わらない最大の改善候補である (0909 で NLL 0.243→0.139【資料】)。

---

## 1. 全体パイプライン

```
 [S1 校正データ]           calib_set.v1 (60 万トークン、日本語が主 40%)   ← 要求ログ前半 + 公開コーパス
 [S1' 評価データ]          eval_set.v1 (JA-KL 70 本 + MCQ 123 本 + 生成タスク)  ← 要求ログ後半 + 公開ベンチ
        │
        ▼
 [S2 教師と感度測定]       harness 教師 (原重み)  → teacher_*.pt (top-32)
                           区分別 routing          → routing_stats.v1
                           層感度 (層を K2 に差替)  → alpha.v1
                           super expert の候補      → superexpert.v1 (CPU で重みから + 出力エネルギー)
        │
        ▼
 [S3 ビット配分]           mixk_plan.v1 (層 × expert の K、k23、π_ja 0.30、安全項 m 0.3)
        │
        ▼
 [S4 量子化]               L0: K2 Viterbi+refit 全体 (v1) + K3 候補 (各層上位 40%) + DSpark K3
                           L1: 校正 pass (plan を入力に、層ごとに LDLQ)  ← v3 のみ
        │
        ▼
 [S5 局所蒸留 / refit]     L0 refit (H=I) / L1 H 重み付き refit / L2 suh/svh の局所蒸留 (日本語 ×2)
        │
        ▼
 [S6 組み立て]             assemble_staged → ds41rt staged snapshot (quantize_config.json、shard 連番)
        │
        ▼
 [S7 評価]                 1 段目: harness teacher-forced (KL / top-1 / MCQ、区分別)
                           2 段目: ds41rt 生成 (日本語 + 日英の対 + ツール + 速度)
                           → eval_report.v1、合否 (§4)
        │
        ▼
 [S8 サーバー]             ds41rt 0-spark exl3-k23/rtx-tp2、KV を明示、前段 proxy (effort の写像)
```

版ごとに通る段は次のとおりである。

| 版 | S1 | S2 | S3 | S4 | S5 | S6 | 形式 | ds41rt の経路 |
|---|---|---|---|---|---|---|---|---|
| v0 | 不要 | 評価用の教師だけ | 不要 | greedy beam16 (作成中) | なし | 不要 (raw) | raw publication、一律 K2 | `exl3` raw |
| v1 | 不要 | 同上 | 不要 | Viterbi | L0 refit | 不要 (raw) | 同上 | 同上 |
| v2 | 不要 | routing + α + super expert | 要 | v1 + K3 候補 + DSpark K3 | L0 refit | 要 | staged、k23 | `exl3-k23/rtx-tp2` |
| v3 | 要 | 同上 | 要 (v2 の plan を流用可) | L1 pass (K2 と候補の K3) | L1 refit + L2 | 要 | staged、k23 | 同上 |

設計上の注意が 1 つある。
H=I (L0) の符号化では experts が互いに独立に符号化されるので、K2 パックと K3 候補パックを残しておけば、配分を変えた再組み立てが GPU なしで 5 分以内にできる【資料、DYN §2.1, §5.4】。
L1 では層 l の入力が量子化済みの層 0..l−1 を通した stream になるので、前段の K の選び方が後段の Hessian に影響する。
そのため v3 は plan を先に決めてから pass を回し、pass の中で候補 experts の K2 版と K3 版を同じ H で両方作る。
v3 の後で配分を少し変える再組み立ては、stream が plan と食い違う分だけ近似になる (影響は小さいと見込むが未確認【推測】)。

---

## 2. 各段の仕様

### 2.1 S1: 校正データと評価データ

**校正データ (calib_set.v1)**

総量は 60 万トークンとする。
sudoingX の pass1 は 85.3 万トークンで 1 層あたり約 49GiB を使った【資料、QT §1.3】ので、60 万なら約 35GiB で GPU 1 枚に収まる【計算】。
区分は採点区間の主な文字種で決め、CJK と仮名が 25% 以上の窓を「日本語が主」とする【JA §7】。

| 区分 | 比率 | トークン | 出典 |
|---|---:|---:|---|
| 日本語が主: エージェント (A0〜A2、F) | 14% | 8.4 万 | `work/x2/req` の前半 |
| 日本語が主: アシスタント (B、D) | 12% | 7.2 万 | 同上 |
| 日本語が主: 短い出力 (C1、C2、G) | 3% | 1.8 万 | 同上 |
| 日本語が主: on-policy (v0/v1 の ds41rt に、評価に使わない日本語の質問を答えさせた推論と回答) | 5% | 3 万 | X-2 の副産物 |
| 日本語が主: 一般コーパス (Wikipedia 3%、技術文書 2%、青空文庫 1%) | 6% | 3.6 万 | `wikimedia/wikipedia` 20231101.ja の 1 shard (176MB)、`research/*.md` の校正用分割、`globis-university/aozorabunko-clean` |
| コードとエージェント (英語の推論、ツール JSON) | 32% | 19.2 万 | `work/x2/req` の code と other の窓、sudoingX `conversations` |
| コード単体 | 13% | 7.8 万 | `work/x2/req` の code 窓、sudoingX code |
| 英語の一般文と数学 | 8% | 4.8 万 | sudoingX `conversations` / `instruct_v3` |
| 中国語ほかの多言語 | 5% | 3 万 | sudoingX の多言語部分、MGSM zh (評価に使わない分割) |
| 画像を含む系列 | 2% | 1.2 万 | 要求ログの画像付き要求 (259 件) |

- 日本語を 40% にするのは、トラフィックのトークン比 (10.9%) より多いが、校正の薄い領域ほど KLD が広がり、データを足すと閉じるという sudoingX の観察に従うためである【資料、`ext/sudoingx_dsv41/FINDINGS.md:56-62`】。公開の校正データには仮名が全文字の 0.06% しかない【実測、QAD §4.1】。
- 日本語だけに寄せないのは、狭い校正が活性の範囲を狭めて別の言語を悪化させた例があるためである【資料、Chimoto et al. EACL 2026、https://arxiv.org/abs/2601.18306 】。
- 冷たい experts の対策: 60 万トークンでは下位 30% の experts が平均約 1,150 回、最も冷たいものは数十回以下しか routing されない【計算、JA §7】。sudoingX の「全トークンの事前分布 τ·mean(g²)·H_all/N を Hessian に混ぜる」処理を使う【資料、`scripts/quantize/p4_pass1.py:1-30`】。
- 保留の分離: 校正は要求ログの前半、評価は後半から取り、同じタスク (最初の発話のハッシュ) が両方に入らないようにする。技術文書はファイル単位で分ける。

形式 (jsonl、1 行 1 系列)

```json
{"schema":"dsv41.calib-set.v1","id":"ja-agent-000123","domain":"ja","sub":"agent",
 "source":"work/x2/req/<file>#<ts>","task_hash":"…","tokens":[…],"score_from":2048,
 "split":"calib"}
```

- 系列長は 4,096 トークン以下。`domain` は {ja, code, agent-en, en, zh-multi, vl} のいずれか。
- 付帯ファイル `calib_set.v1.meta.json` に区分別のトークン数、保留データのハッシュ集合、作成スクリプトの sha256 を記録する。

**評価データ (eval_set.v1)**

JA §3 の 2 段構成をそのまま使う。

- 1 段目: JA-KL 70 本 (J-agent 12、J-assist 12、J-short 4、J-onpolicy 8、J-wiki 4、J-tech 4、J-aozora 2、対照 en-neutral 8、code 8、agent-en 8、計約 28.7 万トークン) と、詰め込み型 MCQ 123 本 (JCommonsenseQA 1,119 問、JamC-QA 2,309 問、MMLU-ProX ja 1,400 問)。
- 2 段目: A0 型エージェントの再生 30 タスク、JHumanEval 164×3、MGSM ja 250、ELYZA-tasks-100、Japanese MT-Bench 60×2 ターン×3、M-IFEval-Ja 172、JSQuAD 1,000、C1 型の状況要約 200、B 型の質問 60、ツール評価 176 (英語版と日本語訳版)。
- 日英の対として、HumanEval、MGSM en、IFEval、MT-Bench en、GSM8K-200 を同じサーバーで回す。
- 取得済みの評価データは計 6.6MB で、`work/r3ja/data/` にある【実測、JA 冒頭】。

| 項目 | 値 |
|---|---|
| ツール | 新規 `tools/calib_build.py` (要求ログの窓切り出し、文字種分類、保留分離)。分類は `tools/r3ja/r3ja_join.py` の `cat()` を流用。新規 `tools/eval_ja/build_sets.py` (MCQ の詰め込み、JA-KL の窓) |
| 所要 | 実装 1 日、実行は CPU で 10〜20 分【推測】 |
| GPU | 不要 (on-policy 5% だけ X-2 の ds41rt 実行に相乗り) |
| RAM / ディスク | 数 GB / 数百 MB |

### 2.2 S2: 教師と感度測定

**(1) 教師 (teacher_*.pt)**

- 実行: `harness/stream_ref.py --teacher-in <set>.jsonl --teacher-out <set>_teacher.pt` (原重み、GPU 1 枚)。
- 保存: 採点位置ごとに top-32 の logprob (fp16) と id (int32)、1 位置 192B。10 万位置で約 19MB【計算、JA §3.2】。全語彙の logits は保存しない。
- 速度: 4,096 トークンの系列 1 本 52〜61 秒 (律速は experts の NVMe 読み込み)【実測、`work/teacher/teacher.log`】。JA-KL 70 本で約 66 分、MCQ 123 本で約 2 h、計約 3 h (1 枚)。
- 区分別の routing を取るため、区分ごとに別の実行に分ける。または系列ごとの counts の差分を保存するよう `stream_ref.py:472` 付近を約 10 行直す【推測】。

**(2) routing_stats.v1**

DYN §5.2 の形式に従う。

```json
{"schema":"dsv41.routing-stats.v1","model":"DeepSeek-V4.1-Flash-1017","source_sha256":"…",
 "domains":{"ja":{"tokens":61402,"files":["ja16.jsonl", "…"]},"agent":{…},"code":{…},"en":{…}},
 "per_domain":{"ja":{"count":[[384 int]×40],"wsum":[[float]×40],"energy":[[float]×40],"xsq":[[float]×40]}, …},
 "mtp":{"count":[[128]×3]}}
```

- `count` は選択回数、`wsum` は routing 重みの和、`energy` は Σ‖w·E(x)‖²、`xsq` は Σ w‖x‖² (Hessian trace 型) である。
- 改修箇所は `stream_ref.py:295-311` の `moe_forward` で、約 30〜50 行【推測、DYN §5.3】。
- 現状: agent24 の教師は完了しており (`work/teacher/agent24_teacher.pt`、`chain.log` に 10:33 完了)、counts は 17 本 68,308 トークン分で集計済み (上位 25% が 69.3%、Gini 0.60)【実測、DYN §1.2】。ja16 の routing は未測定である。

**(3) 日本語との差の指標**

- 層ごとの Jensen-Shannon 距離、上位 25% 集合の Jaccard 係数を ja と agent で比べる (`tools/r3_skew.py` を 2 ファイル比較に拡張)。
- 先頭と末尾の層で言語ごとに分かれ中間層で揃う形 (Bandarkar et al., https://arxiv.org/abs/2510.04694 ) が V4.1 でも出るかを見る。
- 「日本語 expert」の定義: 層内で share_ja が上位 5% かつ share_ja / share_all ≥ 2 の expert【JA §4.2 の案】。

**(4) alpha.v1 (層感度)**

- 方法: 原重みのうち 5 層ずつ 8 群の experts だけを v1 の K2 に差し替え、ja / code / agent の 3 区分 × 各 4 本 (約 1.6 万トークン) で ΔKL を測る。
- 改修: `stream_ref.py:179-192` の `_is_exl3` に層の条件を足す `--exl3-layers` (約 30 行)【推測、DYN §5.3】。
- 所要: 2 枚で約 1.5〜3 h【推測】。
- 測るまでは 0909 の値 (sudoingX `data/alpha_sensitivity.json`、層 0 で 4.71、層 39 で 0.019、約 250 倍の差) を使う【資料】。

```json
{"schema":"dsv41.alpha.v1","method":"swap-layer-group-to-K2","pack":"…-vit","teacher":"…pt",
 "groups":[[0,4],[5,9],…],"alpha":{"ja":[40],"agent":[40],"code":[40],"mix":[40]}}
```

**(5) superexpert.v1**

- CPU で、各層の w1/w3/w2 のノルムを expert ごとに出し、中央値の何倍かを記録する。層 37 の e290 は w2 のノルムが中央値の 2.92 倍だった【実測、TF §2.2】。E8M0 scale (約 9GB) だけを読めば近似でき、量子化と並走できる【推測、TF §6】。
- GPU が空いたら、(2) の `energy` で出力エネルギー最大の expert を確定する。0909 では層 1 で 1 本がエネルギーの 92.3% を持っていた【資料、sudoingX `FINDINGS.md:50-54`】。
- 新規 `tools/superexpert_scan.py`。所要は CPU で 1〜2 h【推測】。

### 2.3 S3: ビット配分 (静的混合精度、「動的量子化」の本体)

**重要度と目的関数**

```
I[l,e] = α_l × Σ_d π_d × S̃_d[l,e]
  S_d[l,e]  = 区分 d での xsq (当面は count で代用) を層内で正規化した share
  S̃_d       = (1−m)·S_d + m/384               (一様分布との混合。m = 0.3)
  π         = {ja 0.30, code 0.35, other 0.35}  (other = 英語の推論、ツール JSON、エージェント)
目的: Σ I[l,e] × ε(K[l,e]) を、bytes_per_rank ≤ 予算 のもとで最小化
  ε(K2) = 0.026, ε(K3) = 0.0042             (sudoingX 実測の相対 MSE、plans/plan_v3_270.json)
  ε が凸なので、1 byte あたりの改善が大きい順に K2→K3 へ上げる貪欲法で最適になる
```

- π_ja = 0.30 は実トラフィックの日本語 (9〜11%) の約 3 倍である。JA §4.2 の案 (ja 0.40) と DYN §4.3 の案 (ja 0.30) のうち、agent 側の代償が +3〜5% にとどまると計算された 0.30 を既定にし【計算、DYN §4.3】、0.15 と 0.45 の plan も作って比べる (§5 X-5)。
- 安全項 m = 0.3 は、計画の分布から外れたときの劣化を 2〜7% 減らす代わりに、手元の分布での損失を 0.5〜5% 増やす【計算、DYN §2.4】。

**制約と保護**

| 項目 | 規則 | 根拠 |
|---|---|---|
| K の段 | {2, 3} のみ。K1 は使わない | ds41rt と sparkinfer に K1 が無い (K2〜K6)。agent で冷たい experts が日本語 expert である可能性がある【資料 + 推測、DYN §4.3】 |
| super expert | 各層で出力エネルギー最大の 1〜2 本を K3 に固定 (計 40〜80 本) | k23 の上限が K3 なので、sudoingX の K5 保護は使えない |
| 日本語 expert | §2.2 (3) の定義に当たる expert を K3 に固定。ただし K3 枠全体の 25% を上限とし、超えた分は重要度順に切る | 日本語の保護で他の枠を潰さないため【推測】 |
| 層 37〜39 | 下限は置かずに始め、DSpark の受理長 τ を v2 で測ってから決める。全部 K3 にすると 1,152 本 (5.1GB、0.075bpw 相当) | DSpark は層 37〜39 の入力を使う (`model.py:1268-1269`)【資料、DYN §2.4】 |
| 速度の上限 | routing で重み付けした K の平均 (decode_weighted_K) ≤ 2.45 | c16 の低下を約 10% に抑えるため【推測、DYN §2.4】 |
| projection | w1/w3/w2 で同じ K にする | ds41rt は projection ごとに表せるが、感度のデータが無い |

**mixk_plan.v1 (配分器の出力)**

```json
{"schema":"dsv41.mixk-plan.v1","family":[2,3],"budget_bytes_per_rank":…,"kv_gib_per_rank":4.0,
 "pi":{"ja":0.30,"code":0.35,"other":0.35},"uniform_mix":0.3,
 "protect":{"super_expert_per_layer":1,"ja_expert":{"top_frac":0.05,"ratio":2.0,"cap_frac_of_k3":0.25},
            "floor_layers":{}},
 "decode_bpw_cap":2.45,
 "layers":{"0":"2332…(384 文字)", …, "39":"…"},
 "mtp":{"0":"333…(128 文字)","1":"…","2":"…"},
 "predicted":{"proxy_cost":…,"proxy_cost_by_domain":{"ja":…,"code":…,"other":…},
              "avg_bpw":2.19,"decode_weighted_K":…,"n_k3":2963},
 "inputs":{"routing_stats_sha256":"…","alpha_sha256":"…","superexpert_sha256":"…"}}
```

| 項目 | 値 |
|---|---|
| ツール | `tools/r3_alloc.py` を拡張した新規 `tools/mixk_plan.py` |
| 所要 | 実装 1 日、実行は CPU で数秒 |
| 出力の置き場 | `work/plans/mixk_<名前>.json` |

### 2.4 S4: 量子化

**(1) L0 の K2 全体パック (v1)**

- 量子化器: `build/qrecipe/tools/quantize_experts_exl3_sm120.py`。`--greedy` を付けなければ tail-biting Viterbi (`quantize_tiles`) を使う (`:256-330`)【資料、QAD §3.1】。
- 起動: `build/quantize_1017.sh` の `gpu` 分岐から `--greedy --beam $BEAM` を外した版を作り、`DST=DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-vit` とする。
- refit: 手元の exllamav3 5be88657 には `refit_scales` が無いので、`build/qrecipe/tools/requant_probe.py:49` の `refit_identity` (純 torch) を `_quantize_group` の後段に入れる。trellis から復元し、行と列の scale r, c を交互最小二乗で求め、`suh *= r`、`svh *= c` とし、relerr が下がったときだけ採用する【資料、`requant_full.py:11-16`】。
- 所要: 当機の greedy は 4.60〜4.77 tensor/s/GPU【実測、`build/quant-logs/quant-g1-20-39.log`】。GB10 比 4.7 倍で換算すると Viterbi は 1.4.9 (手元) で約 0.97 s/tensor、全 46,080 tensor を 2 枚で **約 6.2 h**、1.5.1 なら約 3.1 h【計算、QAD §3.1】。1.4.9 と 1.5.1 で relerr は同じ (0.2619 / 0.2617)【資料】。
- ディスク: +約 140GB (非 expert shard と Engram は hardlink)。

**(2) K3 候補パック**

- 対象: いずれかの区分 (ja / code / other) で層内上位 40% に入る experts の和集合と、superexpert.v1 の候補。
- 方法: 同じ量子化器を `--bits 3` で、Viterbi + refit で回す。expert を絞る `--expert-filter <plan_or_list.json>` を足す (約 20 行、`convert_shards` `:376-470`)【推測、DYN §5.3】。
- 所要: Viterbi の K3 は K2 の約 2 倍の計算と見て【推測】、候補 40% で 2 枚 2.5〜5 h。greedy なら約 1.1 h【推測、DYN §5.4】。
- ディスク: 約 82GB (候補 40% の場合)【計算、DYN §5.4】。

**(3) DSpark の experts を K3 に**

- staged 経路では DSpark の experts も EXL3 にする必要がある (`v41_exl3.rs:141-151, 432-438`)【資料、DYN §2.1】。
- diffbot の `quantize_mtp_exl3.py` を使う。1 枚で約 9 分、5.1GB【資料、QT §1.2】。
- sudoingX は drafter の experts を K3 にしても受理率が保たれたと報告している【資料、`FINDINGS.md:63-65`】。

**(4) L1 校正 pass (v3 のみ)**

- 土台: sudoingX の層ストリーミング pass1 (`ext/sudoingx_dsv41/scripts/quantize/p4_pass1.py`、`dsv41q/stream.py`)。
- 当機向けの改修 (QAD §3.3):
  1. codebook を `mcg` にする (`dsv41q/exl3.py:36` の `mul1` を置き換える)。ds41rt は MCG を必須にしている。
  2. mixk_plan.v1 を入力にし、plan が K3 の expert は K3、候補の expert は K2 と K3 の両方を同じ H で符号化する。stream は plan どおりの K で進める。
  3. `DSV41_CUDA_MEM_FRACTION` を 0.95 程度にする (`stream.py:39`)。
  4. Engram の reader を、ハーネスの `HostEngramEmbedding` (`stream_ref.py:106-153`) に替える。
  5. `--part 0/2` と `1/2` で GPU ごとに 2 プロセスにし、part の交換はローカルの FS で行う。
  6. 出力のテンソル名を `layers.L.ffn.experts.E.wN.{trellis,suh,svh,mcg}` に揃える。
- Hessian の扱い: expert 別の完全な H は 40 層で 1.94TB になり保存できないので、層ごとに GPU 上で `X_eᵀ diag(g²) X_e` を作って捨てる。保存するのは層共有の H_x (計 4.2GB) と expert 別の対角 (計 0.46GB) だけにする【計算、QAD §3.4】。
- 区分の重み付けは H に入れない (データを含めるだけにする)。Hessian 内の重み付け変更は追加効果が無かった【資料、sudoingX `FINDINGS.md:58-62`】。
- 所要: 実装 3〜5 日、GPU は 2 枚で 3〜5 h (pass0 と感度で +0.5 h)【推測、QAD §3.3】。
- メモリ: 60 万トークンで stream + x + y_T が 1 枚 40GB 前後【計算、QAD §2.2 の表から内挿】。

### 2.5 S5: 局所蒸留と refit

| 段 | 学習するもの | 損失 | 推論コスト | 所要 |
|---|---|---|---|---|
| L0 refit | expert ごとの行と列の scale (suh, svh に掛け込む) | 重みの Frobenius 誤差 (閉形式) | なし | S4 (1) に含む |
| L1 refit | 同上 | H 重み付きの誤差 | なし | S4 (4) に含む |
| **L2 局所蒸留** | suh (gate/up 5,120、down 2,304)、sv (gate/up 2,304、down 5,120)、計 22,272 param/expert | `Σ_t c_t g_t² ‖E_T(x_t) − E_S(x_t)‖²`。c_t は日本語トークン 2.0、それ以外 1.0 | なし (計 342M param、0.68GB) | L1 に +0.5〜1.5 h |
| L2' (任意) | shared expert (FP8、STE)、`ffn_norm` | 層単位の `‖MoE_T − MoE_S‖²` | なし (公式と同じ FP8 形式なら ds41rt の raw 経路で受け付ける) | +1〜2 h |
| L3 (任意) | suh/svh、norm、shared | 最終 logits の KL (教師 top-32) | なし | 実装 1〜2 週、学習 7〜10 h + 教師 0.5〜2 h |

**L2 の仕様** (QAD §2.2)

- 学生の重みは `W = diag(su) · Q̃ · diag(sv)` で、`Q̃` は trellis を su=sv=1 で復元したもの (`exllamav3_5be8/.../exl3_lib/quantize.py:1397-1401`)。1 層 384 experts で 27.2GB になるので 64 experts ずつ処理する (4.5GB)。
- SiLU と clamp(±10) を通した 3 射影を一緒に学習する。gate/up の svh は clamp の前に掛かる。
- 入力: 学生の stream (量子化済みの前段を通したもの) を入力にし、同じ入力に対する MXFP4 教師の出力を目標にする (QAD の方式 A)。
- 最適化: Adam、su/sv の lr 5e-4 (QuIP# の設定、https://arxiv.org/abs/2402.04396 §F.6)、4 epoch、検証用 10% で早期終了、refit 後の値からの L2 正則化。
- 日本語 expert (§2.2 (3)) は、早期終了の基準を日本語トークンの誤差に置く。
- 確認事項: ds41rt の native kernel で gate/up の svh が clamp の前に掛かっているかを、`--dry-run` の後に層 0 の参照比較で確かめる【QAD §7-2】。

**採らないもの**

- LoRA / 低ランク補正: ds41rt に経路が無く、MoE の後ろに GEMM を足すカーネル改修が要る (QAD §2.3)。
- 全重みの学習 (L4): 543.6B param の勾配と Adam 状態が数 TB になり、どの規模でも不可 (QAD §2.1)。
- router の学習: routing が変わり DSpark の受理率と負荷分布にも影響するので後回しにする (QAD §2.3)。

### 2.6 S6: staged snapshot の組み立て

- 新規 `tools/assemble_ds41rt_staged.py`。入力は K2 パック、K3 候補パック、DSpark K3、mixk_plan.v1。
- 出力 (ds41rt の契約、`v41_exl3.rs:344-438`)
  - `config.json`: `quantization_config` は compact 版で、`quantize_config.json` のキーの部分集合でなければならない (`:376-385`)。
  - `quantize_config.json`: `quant_method / method / format / checkpoint_format = "exl3"`、`codebook: "mcg"`、`out_scales: "never"`、`group_size: -1`、`desc_act: false`、`pack_dtype: "int32"`、`bits: 2`、`meta.ds41rt = {schema: "ds41rt.v41-routed-exl3.v1", tensor_naming: "checkpoint-native", native_quantization_config: {公式の FP8 ブロック}}`、`tensor_storage[<proj>] = {quant_format: "exl3", bits_per_weight: K, stored_tensors: {trellis (int16, [in/16, out/16, 16K]), suh, svh, mcg}}`。対象は 47,232 projection (40×384×3 + DSpark 3×128×3)。
  - `model.safetensors.index.json` と、**1..N の連番の shard** (`v41_catalog.rs:542-559`)。2 つのパックから hardlink で組み合わせることはできないので、experts を含む shard は書き直す。experts を含まない shard (Engram の 47/48 を含む) は hardlink のままにする。
- 本仕様の追加 (Tensorfold 互換と追跡のため、§2.9): `quantize_config.json` の `meta.dsv41` に、mixk_plan の sha256、calib_set の sha256、区分別の routing share の要約、評価結果の要約を書く。ds41rt が未知のキーを拒否しないかは `--dry-run` で確かめ、拒否される場合は別ファイル `dsv41_provenance.json` に置く。
- 検査: projection 数、shape、byte 長、compact のキーの一致を ds41rt のローダと同じ規則で検査し、その後 ds41rt の `--dry-run` で確かめる。
- 所要: 実装 1〜1.5 日 + 検査 0.5 日。実行は CPU で、読み 150GB と書き 150GB で 2〜5 分【推測、DYN §5.3】。
- ディスク: +約 150GB。

### 2.7 S7: 評価

§4 にまとめる。
出力は版ごとの `eval_report.v1.json` とし、区分別の KL、top-1 一致、ΔNLL、100 文字あたりの KL、MCQ の差と McNemar 検定、生成タスクの得点、日本語固有の自動チェック (§4.4) を 1 表にまとめる。

```json
{"schema":"dsv41.eval-report.v1","version":"v2-pi030-kv4","snapshot_sha256":"…",
 "stage1":{"by_domain":{"J-agent":{"kl":…,"top1":…,"dnll":…,"kl_per100char":…}, …},
           "mcq":{"jcqa":{"acc":…,"teacher_acc":…,"disagree":…,"mcnemar_p":…}, …}},
 "stage2":{"jhumaneval":…,"humaneval":…,"mgsm_ja":…,"mgsm_en":…,"elyza":…,"mtbench_ja":…,
           "mifeval_ja":…,"ifeval":…,"tool_en":…,"tool_ja":…,"c1_rule":…,"a0_replay":…},
 "ja_checks":{"answer_in_ja":…,"zh_simplified_ge1":…,"zh_simplified_ge3":…,"reasoning_lang":…},
 "speed":{"c1":{"code":…,"ja":…},"c16":{"mixed":…},"dspark_tokens_per_round":{"ja":…,"code":…}},
 "gates":{"G-JA1":"pass", …}}
```

### 2.8 S8: サーバー

- image: `ghcr.io/tpurtell/ds41rt-coordinator:v15` (取得済み【実測、`build/pull_ds41rt.log`】)。v15 で EXL3 経路が動かなければ v7 に切り替える (DR §1.2)。
- v0 / v1 は raw publication (一律 K2、`mtp_experts=source`) で起動する。v2 / v3 は staged で `exl3-k23/rtx-tp2` を使う。
- KV を明示する (§3 の案ごとの値)。auto のままだと残りをすべて KV が埋め、K3 の枠と競合する。
- 前段 proxy で reasoning_effort を写す。`medium` (419 件来ている) は V4.1 の `encoding.py:448-453` も ds41rt も受けないので、整数 (例 62) か high に写す【実測 + 資料、JA §2.3】。
- 運用の既定値: V4.1 の推奨は temperature 1.0 / top_p 1.0、GLM の運用は 0.6 が 65%。ds41rt は temperature を省くと greedy になる (DECISION §6.5)。
- 日本語の回答言語は system prompt で指定する (費用 0、JA §4.7)。

### 2.9 Tensorfold の扱い (仕様)

**折りたたみ型の圧縮は採らない。**
1017 の層 3 / 20 / 37 で測った結果は次のとおりで、どの方式でも共有できる成分がほぼ無い【実測、TF §2.2】。

| 測定 | 結果 | 独立なガウスの場合 |
|---|---|---|
| 全 expert の平均成分のエネルギー比 | 0.26〜0.43% | 0.26% (1/384) |
| 平坦化した expert 間 cos の最大値 | 0.008 | 0 付近 |
| hidden 側共有共分散の有効ランク | 4,805〜5,018 / 5,120 | 5,120 |
| 共有 KLT5120 の符号化利得 | 0.013〜0.044 bit/重み | 0 |
| ニューロン単位の最近傍 \|cos\| (中央値 / 最大) | 0.05 / 0.50 | 0.043 (中央値の基準) |
| expert 単体 rank 1024 のエネルギー | 75% (パラメータ 64%) | 72.6% |
| 2bit シミュレーションでの等価ビット得 | +0.001〜+0.010 bit | なし |

- 共有基底を bf16 で持つと 4.19GB (0.062bpw)、FP8 でも 2.1GB かかり、c1 decode の読み出しが +20〜39% 増える【計算、TF §3.2】。正味は −0.06〜−0.02 bit の損である。
- EXL3 の Viterbi + refit (相対誤差 0.262) はガウスの R(D) 限界 (0.25) から約 0.065 bit の位置にあり、重みの空間で残る大きな改善は greedy (0.377) から Viterbi への再符号化 (0.53 bit 相当) だけである【計算、TF §2.3】。
- 未確認の可能性として、Hessian で白色化した空間では expert 間で共有される構造が見える可能性が残る (TF §7-1)。その場合も取り込む手段は L1 の Hessian 校正であり、折りたたみではない【推測】。

**TensorFold エンジンは採らない。**
通信が GB10 の RoCE 専用で PCIe/NCCL への移植が要り、同時処理が 4 系列で c16 の目標に届かず、既製の 2.9bpw パックは experts だけで 182.6GiB あって VRAM に入らない【資料、TF §4.1】。

**流用するもの (本仕様の manifest に入れる要件)**

1. expert ごとの K (ragged per-expert) を持てること。mixk_plan.v1 の `layers` 文字列がそれに当たり、TensorFold fork の mixed-width EXL3 (Krishando 2.86bpw、K2〜K5) と相互に変換できる。
2. suh / svh を fp16 の任意の値として持てること (L2 の学習結果を載せるため)。
3. codebook は MCG に固定すること (ds41rt の kernel と量子化器を共用するため)。
4. 学習した shared expert と norm を非 expert の shard に書き戻せること (hardlink ではなく実体のコピー)。
5. 追跡情報として、expert ごとの区分別 routing share、校正データの構成 (区分、比率、トークン数、保留データの識別子)、区分別の評価結果を持つこと (JA §4.8)。§2.6 の `meta.dsv41` か `dsv41_provenance.json` に置く。
6. 校正手順は Krishando 方式 (量子化済みモデル自身の活性で Hessian を取り直す逐次校正) を L1 として採る。これは sudoingX の pass1 と同じ考え方である。

### 2.10 実行時に動的な方式 (採らない。再検討の条件)

| 方式 | 同じ 17GB 枠での代理損失 | 工数 | 判断 |
|---|---|---|---|
| 静的 K2→K3 (本線) | 0.331 (3,840 本) | §2.3 | 採る |
| K3 を重複して持つ動的プール | 0.584 (1,280 本) | 2〜3 週 | 採らない |
| K2 の残差を加える動的プール | 0.414 (1,920 本) | 3〜4 週 (FC1 で活性化前に足すカーネル改修) | 採らない |
| 活性化の per-token FP8 | 品質の利点なし (参照実装が 1×32 MXFP8、ds41rt の EXL3 は A16) | prefill の速度目的なら別途 | 品質目的では採らない |
| 言語別の router bias (`bias_ja`) | 文献で中間層への介入が多言語 +1〜2% (Qwen3 ほか) | 約 1 週 + GPU 数時間 | 実験扱い (§5 X-8) |

数値は DYN §3 の計算である。
PCIe x8 (約 20GB/s/枚) では 1 step 内の差し替えはできず (c1 で 1 step に約 0.9GB 必要)、秒〜分単位の追従しかできない【計算、DYN §3.1】。

**再検討の条件**: v2 の運用中に routing を時間窓ごとに集計し、静的 plan をその分布で評価した代理損失が 0.50 を超える状態 (plan 時 0.38〜0.40 に対して) が続く場合に限り、残差方式を検討する。
「agent と一様の半々」程度のずれでは静的が 0.53 で残差方式 (0.41) とほぼ互角になるので、閾値はその手前に置いた【計算、DYN §2.4、§3.1】。

---

## 3. VRAM 150GB の予算表

### 3.1 前提

- 1 枚の容量 95.59GiB、余裕として 0.8GiB を残す。
- 0-spark の起動直後の内訳【資料、DR §2.1、`docs/release-v7-configurations.svg`】
  - RTX0: experts 64.38、KV 7.34、実行 2.44、その他 16.79GiB
  - RTX1: experts 64.38、KV 4.90、実行 2.68、その他 19.59GiB (DSpark の MXFP4 experts 7.22GB を含む)
- TP2 ではどの expert も 2 枚に半分ずつ載るので、追加できる量は空きの少ない RTX1 で決まる。
- K2→K3 の増分は 1 本 4.42MB (1 rank 2.21MB)、1 rank あたり 1GiB で約 485 本【計算、DYN §1.1】。
- RTX1 の追加枠 (DSpark を K3 にした場合) = 95.59 − 0.8 − (64.38 + 2.68 + 19.59 − 1.97 + KV) = 10.11 − KV (GiB)【計算】。1.97GiB は DSpark の MXFP4 (7.22GB) と K3 (5.1GB) の差である。
- 1M 文脈の KV は VRAM 上で約 12GiB という見込みがある【推測、DR §10】。現行の KV は 2 枚で計 12.24GiB である。

### 3.2 案の比較

| 項目 | v0 / v1 (参考) | **案 A: 1M 維持** | **案 B: 推奨基準** | **案 C: 品質最大** |
|---|---|---|---|---|
| 形式 | raw、一律 K2 | staged k23 | staged k23 | staged k23 |
| KV / 枚 | auto (7.34 / 4.90) | 6.0 | 4.0 | 2.0 |
| DSpark experts | MXFP4 7.22GB | K3 5.1GB | K3 5.1GB | K3 5.1GB |
| RTX1 の追加枠 | なし | 4.11GiB | 6.11GiB | 8.11GiB |
| K3 の本数 (全 15,360 中) | 0 | 約 1,990 (13.0%) | 約 2,960 (19.3%) | 約 3,930 (25.6%) |
| 平均 bpw (名目) | 2.00 (実効 2.035) | 2.13 | 2.19 | 2.26 |
| experts 合計 (2 rank) | 138.2GB | 147.0GB | 151.3GB | 155.6GB |
| 代理損失 (一律 K2 = 1、agent 分布)【計算 + 内挿】 | 1.00 | 約 0.49 | 約 0.40 | 約 0.32 |
| 換算 KL の目安 (×0.36、0909 の一律 K2 から)【推測】 | 0.36 (0909) | 約 0.18 | 約 0.14 | 約 0.12 |
| 速度 (v1 比)【推測】 | 基準 | c1 −3%、c16 −6% | c1 −5%、c16 −10% | c1 −7%、c16 −13% |
| 長文脈 | 1M | 1M を 1 本保持できる見込み | 2 枚で 8GiB、1M の 2/3 程度【推測】 | 2 枚で 4GiB、1M の 1/3 程度【推測】 |

- 代理損失は DYN §2.4 の表 (2.10bpw 0.545、2.15bpw 0.452、2.20bpw 0.383、2.25bpw 0.331) を線形に内挿した値で、m = 0.3 の安全項でさらに数 % 増える。L0 (v1) の改善 (MSE 比 0.48【資料、`RESULTS.md:1004`】) はこの表に入っていない。L0 と配分の効果が掛け算で効くかは未確認である【推測】。
- 換算 KL は「一律 K3 の代理損失 0.16 × 0.36 = 0.058 が校正あり 3.0bpw の公開値 0.0545 とほぼ合う」ことに基づく目安で、偶然の一致の可能性もある (DYN §2.4)。
- 速度は、ホットな experts を K3 にするので読み出し量が平均 bpw より大きく増える (2.20bpw で c1 ×1.22、c16 ×1.17) ことから見積もった【計算 + 推測、DYN §2.4】。案 A と案 C は本数比で按分した。
- 長文脈の列は「1M で約 12GiB」が推測なので、v2 の起動時に `/v1/stats` と `qualify-ds41-native-needle.py` で実測する。

### 3.3 案 B の K3 枠の内訳 (目安)

| 枠 | 本数 | 規則 |
|---|---:|---|
| super expert | 40〜80 | 各層の出力エネルギー最大 1〜2 本 |
| 日本語 expert | 最大 740 (枠の 25%) | §2.2 (3) の定義に当たるもの。ja16 の測定前は不明 |
| 重要度順 | 残り (約 2,140〜2,880) | §2.3 の貪欲法。層 0〜19 に集まる (2.20bpw で層 0 は 316 本、層 30〜39 は計 11 本)【計算、DYN §2.4 `dec_out.txt`】 |
| 層 37〜39 の下限 | 0 (初版) | τ を測ってから決める。全部 K3 にすると 1,152 本で枠の 39% を使う |

---

## 4. 評価プロトコルと合格基準

### 4.1 2 段構成

| 段 | 実行先 | 中身 | 所要 / 版 | 基準の取り方 |
|---|---|---|---|---|
| 1 段目 | harness (teacher-forced、原重みの教師と学生に同じ列を通す) | JA-KL 70 本 (日本語 7 区分 + 対照 en-neutral / code / agent-en)、MCQ 123 本 (4,828 問) | 約 3 h (1 枚)、2 枚で約 1.5 h【計算、JA §3.3】 | 教師との直接比較 (対の比較) |
| 2 段目 | ds41rt (生成、実運用と同じ API) | JA §3.4 の日本語生成タスク、日英の対、ツール評価、速度 | 約 1.5〜2 h + 日英の対 約 1 h【推測、JA §3.4】 | 版どうしの比較、日英の比、小さな原重み基準、on-policy 採点 |

- ds41rt の API には logprobs が無いので、分布の比較はハーネスでしかできない (DR §7)。
- ハーネスの生成は 1.4〜1.75 tok/s しか出ない【実測、`work/teacher/greedy3.json`】ので、原重みの生成基準は ELYZA 30 問と JHumanEval 30 問を thinking off で夜間に 1 回だけ取る (3.8〜4.8 h、1 枚)【計算、JA §3.5】。
- 評価条件は実運用に合わせる。A 群と F 群は thinking + effort high、C 群と G 群は effort low、回帰比較は greedy、最終確認は temperature 1.0 と 0.6 の両方で回す (JA §4.5)。
- GPU の順序: ハーネスと ds41rt は同時に動かせないので、2 段目を版ごとにまとめて回し、その後に 1 段目をまとめて回す (JA §6.1)。

### 4.2 区分ごとの合格基準 (案)

KL の絶対値の基準は置かない。
ハーネスの KL は教師の top-32 上の近似で、diffbot (top-512) の値とは比べられないからである。
v0 で初めて測った値を基準にし、以降の版はその比で見る。

**日本語 (JA §6.2 の G-JA1〜10)**

| ID | 指標 | 基準 |
|---|---|---|
| G-JA1 | 日本語各区分の KL / 同じ版の en-neutral の KL | 1.3 以下 |
| G-JA2 | 日本語区分の top-1 一致 | en-neutral の top-1 − 2 点以上 |
| G-JA3 | MCQ の教師との正答率差 (対の比較) | JamC-QA 3.0 点、JCommonsenseQA 1.5 点、MMLU-ProX ja 2.5 点以内 |
| G-JA4 | JHumanEval、MGSM ja、ELYZA、MT-Bench ja | 日英の比が v0 より悪化しない。原重みの小基準と比べられる項目は低下 3 点以内 |
| G-JA5 | 日本語の指示に日本語で答える率 | 99% 以上 |
| G-JA6 | 簡体字専用字を含む日本語回答 | 1 字以上 1.0% 以下、3 字以上 0.3% 以下 (GLM の実出力 0.47% / 0.10%) |
| G-JA7 | C1 型の字数と文体、M-IFEval-Ja (strict) | C1 95% 以上、M-IFEval-Ja と IFEval の差が v0 より広がらない |
| G-JA8 | 日本語の推論トークンの中央値、上限到達率 | 最良版の 1.2 倍以内、上限到達 1% 以下 |
| G-JA9 (参考) | 日本語負荷の c1、DSpark の 1 round のトークン数 | c1 80 tok/s 以上、2.3 トークン/round 以上 |
| G-JA10 | 日本語に寄せた版での en-neutral と code の KL | v1 より 10% 以上悪化しない |

**英語、コード、エージェント、ツール (DECISION §4.2 の G1 を区分別に分けたもの)**

| ID | 指標 | 基準 | 根拠 |
|---|---|---|---|
| G-EN1 | en-neutral の KL と top-1 | v0 比で改善 (v1 以降)。v2 / v3 で v1 より悪化しない | 版どうしの比較 |
| G-CD1 | code と agent-en の top-1 一致 | 0909 の 3.0bpw 校正版 (96.0 / 98.2%) から 3 点以内 | 【資料、QT §5】 |
| G-CD2 | HumanEval、JHumanEval | HumanEval の低下 3 点以内 (基準は 1017 の原重み小基準、無ければ 0909 の native 95.7) | DECISION G1、QT §5 |
| G-CD3 | GSM8K-200、MGSM en | GSM8K 97% 以上 (0909 一律 2.0 で 97.5%) | 【資料、QT §5】 |
| G-AG1 | A0 型エージェント再生 30 タスクの受け入れテスト合格率 | v0 以上、v2 / v3 で v1 以上 | 自前。元タスクの置き場は要確認 (§6) |
| G-TL1 | ツール評価 176 件 (英語版) | 150/176 以上 | DECISION G1 |
| G-TL2 | ツール評価の日本語訳版 | 英語版との差が v0 の差より広がらない | 日本語の指示でツールを動かす力 |
| G-LC1 | 1M の needle (`qualify-ds41-native-needle.py`) | 合格 (案 A のみ必須) | §3.2 |
| G-SP1 | 同じ再生負荷での c16 合計と c1 | c16 が GLM 以上、c1 150 以上 (DECISION G2)。v2 / v3 の v1 比の低下は案の見込み (§3.2) の 1.5 倍以内 | DECISION §4.2 |

**判定の規則**

- 合否は個別の基準で決める。JA §5 の重み付き総合点 (日本語エージェント 40%、アシスタント 25%、知識 10%、短い出力 10%、ツール 10%、対照 5%) は版の順位付けにだけ使う。総合点で判定すると、重みの小さい区分での大きな劣化 (言語の切り替わりなど) を見落とすからである。
- MCQ と top-1 の差は McNemar 検定で p < 0.05 のときだけ「差あり」とする。JCommonsenseQA 1,119 問で食い違い率 8% なら 95% 区間は ±1.7 点である【計算、JA §3.3】。
- 自動指標で −1.7% の日本語が人手評価では −16% だった例がある【資料、Marchisio et al. 2024、https://arxiv.org/abs/2407.03211 】ので、最終版の候補は日本語の自由記述 20 問程度を人手で見比べる。

### 4.3 judge

- 外部 judge (Swallow は `gpt-5.2-2025-12-11` を使う) が最も信頼できる。送るのは公開ベンチの設問と量子化版の回答だけにし、要求ログは送らない。使うかどうかはユーザーの判断である (§6)。
- 外部に出さない場合は、ハーネスの原重みを judge にし、「評価: 」の直後の 1〜10 のトークン確率から期待値を読む。ELYZA 100 件と MT-Bench 360 件で約 3.2 h【計算、JA §3.5】。

### 4.4 日本語固有の自動チェック (すべての生成出力に掛ける)

応答言語の一貫性、簡体字の混入 (`tools/r3ja/r3ja_zhmix.py`)、推論の言語 (`</think>` 前)、濁点の誤り (辞書にない語で濁点の付け外しで辞書語になるもの)、文体の揺れ (です・ます調と である調の混在)、推論長と上限到達率を見る (JA §3.6)。
濁点を見るのは、Engram の圧縮語彙が濁点と半濁点を区別しない (が=か、で=て など 131 トークン) ためである【実測、JA §1.2】。
これは学習時からの仕様なので、実装を作り直すときに「直して」はいけない。

---

## 5. 実験計画

### 5.1 時刻の見込み (2026-10-18 11:10 時点)

- 量子化 v0 の進捗: GPU1 (層 20〜39) は 17,096 / 23,040 tensor、GPU0 (層 0〜19) は 9,384 / 23,040 tensor で、どちらも 4.75 tensor/s 前後【実測、`build/quant-logs/quant-g{0,1}-*.log`】。
- GPU1 は約 21 分後 (11:30 頃)、GPU0 は約 48 分後 (12:00 頃) に空く見込みである【計算】。その後 `quantize_1017.sh finish` で index を作る。
- ディスクの空きは 414GB【実測、`df -h /`】。v0 の残り (約 59GB) を書くと約 355GB になる【計算: 138.2GB × (1 − 26,480/46,080)】。

### 5.2 実験の一覧 (優先順)

| ID | 中身 | 前提 | GPU | CPU 実装 | ディスク | 判定 |
|---|---|---|---|---|---|---|
| **X-0** | CPU 作業: calib_set.v1 と eval_set.v1 の作成、harness 改修 (routing_stats、`--exl3-layers`、系列ごとの counts)、量子化器への refit と `--expert-filter`、mixk_plan.py、assemble_ds41rt_staged.py、superexpert_scan.py | なし (今すぐ) | 不要 | 3〜4 日 (並行して進める) | 数 GB | 単体テスト。assemble は v0 を入力にした「全部 K2」の staged が ds41rt の `--dry-run` を通ること |
| **X-1** | 日本語の教師と区分別 routing: ja16 → JA-KL 70 本 → MCQ 123 本の教師を取り、区分別の counts を保存。あわせて V4.1 が日本語の指示に何語で推論するかを 3 件 × 300 トークンで見る | GPU1 が空く (11:30 頃)。eval_set.v1 の JA-KL / MCQ 部分 (X-0 の一部、半日) | 1 枚 約 3 h (ja16 だけなら 15〜25 分) + 推論言語 約 10 分 | X-0 の harness 改修の一部 | 約 50MB | 日本語と agent の routing の JS 距離と上位集合の Jaccard。日本語 expert の本数。推論言語が中国語なら校正の zh 比率と G-JA6 を見直す |
| **X-2** | v0 のベースライン: 1 段目 (学生採点) と 2 段目 (ds41rt raw、日本語 + 日英の対 + ツール + 速度) | v0 完成、X-1 の教師 | 1 段目 1 枚 3 h、2 段目 2 枚 2〜3 h | 評価スクリプト (X-0) | 数百 MB | 全ゲートの基準値を記録。G-TL1 と G-SP1 で ds41rt の動作を確認。G-JA1 が 1.3 を超えるかで日本語の劣化の大きさが分かる |
| **X-3** | L0: v1 (Viterbi + refit) を作り、X-2 と同じ評価 | v0 の評価が済んでいる (GPU の取り合いを避けるため。X-2 の 2 段目と順番を入れ替えてもよい) | 2 枚 6.2 h (1.5.1 なら 3.1 h) + 評価 5 h | refit の組み込み (0.5 日) | +140GB (空き約 215GB) | v1 が v0 より区分別 KL で改善し、G-JA10 以外の悪化が無ければ v1 を土台にする。その後 v0 を消す (+140GB) |
| X-4 | 層感度 α (5 層 × 8 群 × ja / code / agent) と super expert の確定 | v1、X-1 | 2 枚 1.5〜3 h | `--exl3-layers` (X-0) | 数 MB | 1017 の α と 0909 の α の相関。日本語の α が端の層で大きいか |
| X-5 | v2: K3 候補パック + DSpark K3 + plan (π_ja 0.30 / 0.15 / 0.45、agent のみ) + assemble。案 B (KV 4GiB) で評価 | v1、X-1、X-4、v0 削除済み | K3 候補 2 枚 2.5〜5 h、DSpark 9 分、評価 5 h × 2〜3 plan | mixk_plan、assemble (X-0) | +82GB (K3) +150GB (snapshot)。空き約 123GB | π_ja 0.30 が agent のみの plan より日本語区分で良く、code / agent の悪化が G-JA10 内なら採用。τ と速度で層 37〜39 の下限と案 A / B / C を決める。再配分は GPU なしで 5 分 |
| X-6 | v3: L1 + L2 (日本語 40% の校正、日本語 ×2 の局所蒸留)、plan は X-5 の採用版 | X-5、calib_set.v1 (on-policy 5% は X-2 / X-3 の副産物) | 2 枚 4〜8 h + 評価 5 h | pass1 の改修と L2 で 3〜5 日 | v3 snapshot 150GB。v2 の snapshot と K3 候補を消す必要がある (§6) | v3 が v2 より日本語区分で改善し、G-JA10 と G-EN1 を満たせば本採用 |
| X-7 (任意) | L2' (shared expert の FP8 STE)、L3 (全体 KL の scale-only KD) | X-6 | +1〜2 h / 7〜10 h + 教師 0.5〜2 h | 2 日 / 1〜2 週 | なし | L2 からの上積みが KL で 10% を超えるか |
| X-8 (任意) | 言語別 router bias (`bias_ja`、中間層のみ)、DSpark の日本語追加学習 (速度目的) | X-6 | 数 h | 1 週 / 数日 | 隠れ状態 30GB (DSpark) | 日本語区分の KL と G-JA9。V4.1 で効く保証は無い |
| X-9 (任意) | Engram を 0 にした原重みで日本語と英語の落ち幅を比べる | X-1 | 1 枚 約 3 h | なし | なし | 日本語が Engram にも敏感かどうか (Engram の表を変える案を判断する材料) |

### 5.3 GPU 時間の合計 (X-1〜X-6)【計算 + 推測】

| 段 | GPU 時間 |
|---|---|
| X-1 教師 | 1 枚 3 h |
| X-2 v0 評価 | 1 枚 3 h + 2 枚 2〜3 h |
| X-3 v1 作成 + 評価 | 2 枚 6.2 h + 5 h |
| X-4 α | 2 枚 1.5〜3 h |
| X-5 v2 (3 plan 評価) | 2 枚 2.5〜5 h + 10〜15 h |
| X-6 v3 + 評価 | 2 枚 4〜8 h + 5 h |
| 計 | 2 枚換算で約 40〜55 h (GLM 停止時間の確保が要る) |

X-5 の評価を 3 plan すべて 2 段目まで回さず、1 段目で上位 1〜2 plan に絞れば 5〜8 h 減らせる。

### 5.4 判定の流れ

```
X-2 (v0) ─ G-SP1 で ds41rt が動かない ──→ v7 image を試す。だめなら DECISION §3.4 の案 A を検討
   │
X-3 (v1) ─ v1 ≤ v0 (改善なし) ──→ refit の組み込みを点検 (relerr のログ)。exllamav3 1.5.1 で再実行
   │ 改善
X-1 の結果 ─ 日本語 expert が実在し、agent plan で K2 に落ちる率が高い ──→ X-5 の π_ja を 0.30 以上に
   │
X-5 (v2) ─ 全ゲート合格 ──→ 運用候補。X-6 は日本語の上積みを狙う任意の段に格下げ
   │ G-JA1 / G-JA3 / G-JA4 で不合格
X-6 (v3) ─ 不合格 ──→ 案 C (KV 2GiB) を試す。それでも不合格なら、全 VRAM 常駐の 2.2bpw 級では日本語の合格線に届かないと判断し、DECISION §6.2 の E2 (2.9bpw + host experts) と比べる
```

---

## 6. リスクとユーザー判断事項

### 6.1 リスク

| リスク | 根拠 | 対処 |
|---|---|---|
| 代理損失から実際の KL への換算が外れる | 代理損失は「誤差が足し算で効き MSE が KL に比例する」仮定 (DYN §2.4) | X-5 の第 1 版で係数を当てはめ直す |
| 日本語の routing が agent と大きく違い、配分が日本語に効かない | routing が一様に近い分布では expert 別配分 0.67〜0.69 で層別 (0.59) より悪い【計算、DYN §2.4】 | X-1 を最初に回す。安全項 m を 0.5 まで上げる案も持つ |
| routing の標本が少ない | agent は 17〜24 本で同系統のセッションが多く、偏りが強めに出ている可能性 (DYN §1.2) | X-1 と X-2 の on-policy で標本を足す |
| 1017 で L0 の効果が 0909 ほど出ない | NLL −42% は 0909 での sfxnz の測定 | X-3 で確かめる。1017 と 0909 の量子化耐性はほぼ同じという公開値はある (Lucebox、QT §5) |
| ds41rt の EXL3 経路が v15 で動かない、または staged の混合経路が未検証 | EXL3 経路は v7 以降に測られていない (DR §1.2) | v0 で raw を、X-0 の「全部 K2 の staged」で staged を先に確かめる |
| ds41rt で svh が clamp の前に掛かっていない | sfxnz の p2b kernel では検査済み、ds41rt は未確認 (QAD §7-2) | 層 0 の参照比較 |
| 日本語に寄せすぎてコードとエージェントが悪化する | Chimoto の狭い校正の失敗例 | G-JA10、G-EN1、G-CD1 で監視。π と校正比率は 1 行の設定で戻せる |
| L2 の過学習 (冷たい experts) | 下位の experts はトークンが平均の数分の 1 (QAD §2.2) | 検証 10%、早期終了、refit 値からの L2 正則化 |
| DSpark の受理率が日本語で下がる | multilingual は 56.3% (code 88.1%)、量子化でさらに下がる見込み (JA §4.6) | G-JA9 で監視。層 37〜39 の下限を τ で決める |
| KV の縮小で長文脈が保持できない | 1M で約 12GiB の見込み (DR §10、推測) | 案 A を残す。G-LC1 で実測 |
| ディスク不足 | v1 と v2 の snapshot と K3 候補を並べると空きが約 123GB、v3 の snapshot (150GB) は置けない | §5.2 の削除順 (v0 → v2 の snapshot と K3 候補) を守る。docker イメージ最大 109GB の削除はユーザー判断 |
| 生成タスクの原重み基準が取れない | ハーネスの生成が 1.4〜1.75 tok/s | 版どうしの比較、日英の比、on-policy 採点、夜間の小基準で補う |
| A0 型エージェント再生の元タスクの所在が不明 | 要求ログには作業ツリーが無い (JA §8) | 見つからなければ G-AG1 を ds41rt のツール評価と JHumanEval で代える |

### 6.2 ユーザー判断事項

1. **KV と品質の交換 (§3.2)**：案 A (KV 6GiB/枚、1M 維持、2.13bpw)、案 B (4GiB、2.19bpw、推奨基準)、案 C (2GiB、2.26bpw) のどれにするか。1M 文脈を何本同時に保持する必要があるかで決まる。
2. **速度の許容**：案 B で c1 約 −5%、c16 約 −10% (v1 比、推測) を許すか。許さない場合は decode_weighted_K の上限を 2.35 前後に下げる。
3. **日本語の重み**：π_ja = 0.30 (配分)、校正の日本語 40%、L2 の日本語トークン重み 2.0 でよいか。日本語に寄せるほどコードとエージェントの余裕が減る。
4. **GPU の占有時間**：X-1〜X-6 で 2 枚換算約 40〜55 h。GLM (glm53v) を止める時間帯と、ds41rt との切り替え運用 (DECISION §6.1)。
5. **exllamav3 1.5.1 への更新**：Viterbi の時間が 6.2 h から 3.1 h になる。image の作り直しと pip の小さな取得が要る。
6. **外部 judge と公式 API の利用**：公開ベンチの設問と量子化版の回答だけを送る。要求ログは送らない。使わない場合はハーネスの原重みを judge にする (約 3.2 h / 版)。
7. **ディスク**：v1 の評価後に v0 を消すこと。X-6 の前に v2 の snapshot と K3 候補を消すか、使っていない sglang イメージ 3 本 (最大 109GB) を消すか。
8. **X-6 以降への投資**：L1 + L2 は実装 3〜5 日、L3 は 1〜2 週。X-5 の結果を見てから決めることを勧める。
9. **「向上」の範囲**：量子化と蒸留では元の 1017 を超えない。日本語を 1017 より良くしたい場合は別の SFT 課題になり、この機材と ds41rt には学習経路も LoRA の経路も無い (QAD §4.3、JA §4.7)。DSpark の日本語追加学習は品質ではなく速度の向上として、X-6 の後に検討できる。
10. **reasoning_effort の `medium` の写し方**：前段 proxy で整数 (例 62) にするか high にするか。

---

## 参考 (主な出典)

- 手元の調査: `research/r3_qad.md`、`r3_dynamic.md`、`r3_tensorfold.md`、`r3_japanese.md`、`r3_routing_skew_measured.md`、`DECISION.md`、`r2_ds41rt.md`、`r2_quant_tools.md`
- 手元のコード: `ext/ds41rt/rust/crates/ds41rt-loader/src/{v41_exl3,v41_exl3_residency,v41_exl3_staging,v41_catalog}.rs`、`ext/ds41rt/native/include/ds41rt_native.h`、`harness/stream_ref.py`、`build/qrecipe/tools/{quantize_experts_exl3_sm120,requant_probe,requant_full}.py`、`build/exllamav3_5be8/exllamav3/modules/quant/exl3_lib/quantize.py`、`ext/sudoingx_dsv41/{RECIPE.md,FINDINGS.md,scripts/quantize/p4_pass1.py,scripts/quantize/dsv41q/}`、`ext/sfxnz_recipe/results/RESULTS.md`
- 手元の測定: `research/tmp/r3/*_out.txt`、`work/r3fold/layer{3,20,37}.json`、`work/r3ja/*.out`、`work/teacher/teacher.log`、`build/quant-logs/*.log`
- QuIP#: https://arxiv.org/abs/2402.04396 / QTIP: https://arxiv.org/abs/2406.11235 / PV-Tuning: https://arxiv.org/abs/2405.14852
- NVIDIA ModelOpt QAD (Qwen3.6 W4A4): https://nvidia.github.io/Model-Optimizer/announcements/qwen36-w4a4-qad.html
- Unsloth Dynamic 2.0: https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- Marchisio et al. 2024: https://arxiv.org/abs/2407.03211 / Chimoto et al. 2026: https://arxiv.org/abs/2601.18306 / Bandarkar et al.: https://arxiv.org/abs/2510.04694
- TensorFold: https://github.com/ashhart/TensorFold 、https://github.com/bertholomus/TensorFold

---

## 7. 反証レビュー (2026-10-18 11:50、独立検算)

この節は、本仕様と r3_*.md の数値とコードの参照を、手元のコードとログで検算した結果である。
GPU は使っていない。docker コンテナには触れていない。重みは取得していない。
致命的な誤りを §7.1 に、計画の修正が要る誤りを §7.2 に、検算で正しいと確かめた項目を §7.3 に書く。

### 7.1 致命的な誤り (このまま実行すると失敗するもの)

**R-1. X-6 (v3) のディスク計画が成り立たない。**

- §2.4 (4) の 2 は、L1 pass で「候補 experts は K2 版と K3 版の両方を同じ H で作る」としている。
- その出力は K2 全体 136GB (15,360 × 8.85MB) と K3 候補 81.5GB (6,144 × 13.27MB) で、計約 218GB になる【計算】。
- これを組み立てた v3 snapshot が約 151GB 要るので、X-6 で新たに要る量は約 369GB である【計算】。
- 一方、§6.1 の削除順 (v2 snapshot と K3 候補を消す) を守っても、X-6 の開始時の空きは 123 + 150 + 82 = 約 355GB しかない【計算、§5.2 の X-5 の値から】。
- 不足は約 14GB で、数時間の GPU 実行の終盤 (組み立て) でディスクが埋まる。
- 対処の案は 3 つある。(a) 組み立てを層ごとに pass の出力から直接 shard に書き、採らなかった版を層ごとに消す。(b) 候補の二重符号化を、配分の境目にある experts (例: 重要度順で 2,960±1,000 本) に絞る。(c) docker イメージの削除 (ユーザー判断)。
- あわせて、v2 snapshot と K3 候補を消すと、v3 が不合格だったときに v2 へ戻すには K3 候補の作り直し (GPU 2 枚で数時間) が要る。この戻しの費用は §5.3 に入っていない。

**R-2. L1 (+L2) の GPU 時間「2 枚で 3〜5 h」は、同じ仕様の L0 の値 (6.2 h) と矛盾する。**

- L1 も L0 と同じ Viterbi カーネル (`quantize_tiles`) で同じ 46,080 tensor を符号化する。LDLQ は帯ごとの補償 GEMM が加わるだけで、符号化の量は減らない (`exllamav3_5be8/.../quantize.py:1507-1530` の `ldlq` / `ldlq_batched`)。
- そのうえ候補 40% は K2 と K3 の両方を作るので、符号化する tensor は約 64,500 本 (×1.4) になる【計算】。
- 仕様自身の L0 の値 (0.97 s/tensor、1.4.9) を使っても、符号化だけで 64,512 × 0.97 / 2 = 約 8.7 h になる【計算】。stream の前進と Hessian の作成がこれに加わる。
- 3〜5 h という値は、sudoingX の 2x GB10 で 13 h を 4.7 倍で割ったもので、sudoingX が使った exllamav3 は 1.6 系の tree である (`ext/sudoingx_dsv41/RECIPE.md:80`)【資料】。手元の 5be88657 は 1.4.9 (`exllamav3/version.py:1`)【実測】で、Viterbi は 1.5.1 以降の約 2 倍遅い (QAD §3.1 の表)。
- 修正値は、1.4.9 のままなら 2 枚で約 10〜16 h、1.5.1 以降なら約 5〜8 h である【計算 + 推測、R-3 の速度比を使用】。§0.1 の「L1 と L2 は同じ pass で GPU 4〜8 h」と §5.3 の合計 (40〜55 h) もこれに合わせて直す必要がある (合計は約 50〜65 h)。

### 7.2 修正が要る誤りと見落とし

**R-3. Viterbi の速度換算 (GB10 比 4.7 倍) の根拠が弱い。L0 は 6.2 h ではなく 8〜11 h かかる可能性がある。**

- 4.7 倍は greedy beam16 の比である。greedy は全 tile を一度に処理する torch の演算 (`quantize_experts_exl3_sm120.py:189-245`) で、メモリ帯域 (GB10 273GB/s に対し 1.79TB/s) が効く。
- Viterbi カーネルは 1 tile 1 block で、K≤3 では同時に走る tile が 256 に固定されている (`quantize.py:53-65` の `max_batch_size = 256`、`exllamav3_ext/quant/quantize.cu:91`)【資料】。
- GB10 (48 SM × 2 block = 96 並列) に対し、当機 (188 SM) でも 256 並列で頭打ちになり、並列度の比は約 2.7 倍である【計算】。block あたりの速度差を 1.0〜1.3 倍と見ると、Viterbi は 1.3〜1.7 s/tensor、全 46,080 tensor を 2 枚で 8.3〜10.9 h になる【推測】。
- 対処: GPU が空いたら最初に 1 shard 分 (数分) の Viterbi を測る。K2 で `max_batch_size` を 2×SM 数 (376) に上げる 1 行の変更で並列度は 1.47 倍になる (temp_edges は 376 × 256 × 16,384 × 2B = 3.2GB で収まる)【計算】。

**R-4. 「Viterbi の K3 は K2 の約 2 倍の計算」(§2.4 (2)) は逆向きである可能性が高い。**

- カーネルの 1 step の仕事量は `(32768 >> K) × 2^K = 32,768` で K によらない (`quantize_tiles_kernel.cuh:21-23`)。K3 は temp_edges の書き込みが K2 の半分 (`edges = 65536 >> K`、`:61`) で済む【資料】。
- sudoingX も「K2 が多い層が最も遅い」と書いている (`RECIPE.md:99` の pass 1 の行)【資料】。
- K3 候補パックの時間は、K2 全体の 0.4 倍以下 (R-3 の速度で 2 枚 3.3〜4.4 h、変更後 2.3〜3.0 h) と見るのがよい【推測】。greedy の K3 は候補が 2 倍になるので、diffbot の 0.47 s/tensor (K2 greedy の約 2.2 倍) はその場合の値である。

**R-5. DSpark の EXL3 化に使う量子化器が手元に無く、X-0 の「全部 K2 の staged が `--dry-run` を通る」も GPU なしでは作れない。**

- diffbot の `quantize_mtp_exl3.py` は手元に無い (`ext/diffbot_exl3_2bpw_recipe/` にも無い。README の記載だけ、`research/tmp/r2q/diffbot/quantize_README.md:23-31`)【実測】。出力名は「vLLM が layers 40-42 として読む」形で、ds41rt の `mtp.{stage}.ffn.experts.{e}.wN.*` (`v41_exl3.rs:398-417`) とは名前の変換が要る可能性がある【推測】。
- 手元の量子化器は `mtp.` を明示的に除外している (`build/qrecipe/tools/pack_meta.py:58-59`)。除外を外して名前を合わせる改修 (数行) か、diffbot のスクリプトの取得 (小さなファイル) が要る。
- staged の manifest は DSpark の 3 × 128 × 3 projection を必須にしている (`v41_exl3.rs:395-436`) ので、X-0 の staged 検査の前に DSpark の EXL3 化 (GPU 1 枚 数分〜十数分) が要る。X-0 を「GPU 不要」とした §5.2 は誤りで、X-3 の前後に組み込む必要がある。

**R-6. L2' で `ffn_norm` を学習すると routing が変わる。**

- `ffn_norm` の出力は experts だけでなく router (`gate`) にも入る (`ref/model.py:992-994`)【資料】。
- 学習で top-6 の選択が変わる (微分できない) ので、層単位の損失が不連続になり、DSpark の受理率と負荷分布にも影響する。§2.5 の「router の学習は後回し」と同じ理由で、L2' では `ffn_norm` を固定し、shared expert だけにするのがよい。

**R-7. SwiGLU の clamp は対称ではない。**

- gate は上側だけ (`clamp(max=10)`)、up は ±10 である (`ref/model.py:847-848`、`arch_spec.md:506`)【資料】。§2.5 の「SiLU と clamp(±10)」をそのまま実装すると、gate が −10 未満の位置で教師とずれる (差は小さいが、局所蒸留の目標が教師と一致しなくなる)。

**R-8. L2 の目標 `y_T` の持ち方が損失の形と合っていない。**

- §2.5 の損失 `Σ_t c_t g_t² ‖E_T(x_t) − E_S(x_t)‖²` は expert ごとの出力を目標にする。QAD §2.2 のメモリ表の y_T は [N, 5120] (MoE 合成後) で、expert ごとの目標にはならない。
- expert ごとの目標を持つと 60 万トークンで 600K × 6 × 5,120 × 2B = 36.9GB になる【計算】。教師の MXFP4 experts (層あたり 7.2GB) を GPU に置き、64 本ずつの chunk で教師出力をその場で計算する方式にすべきである。その場合、chunk 内の活性と勾配で 1 枚 70〜80GB 前後になるので、chunk を 16〜32 本にする【計算 + 推測】。

**R-9. 現状の記述が古い (11:50 時点)。**

- ja16 の教師は 11:42 に完了しており、routing の counts を含む `work/teacher/ja16_teacher.pt` (2.1MB) がある (`build/quant-logs/chain.log`)【実測】。§2.2 (2) の「ja16 の routing は未測定」と X-1 の最初の段は、CPU の集計だけで済む。
- GPU1 の量子化は 11:27 に完了し、GPU0 は 21,728 / 23,040 tensor (4.69/s) で 12:00 頃に終わる見込みである【実測】。ディスクの空きは 361GB【実測、`df -h /`】。
- 教師の速度は、仕様の 52〜61 s/系列に対し、課題文では 60〜100 s である。X-1 と 1 段目の評価 (約 3 h) は 3〜5.5 h を見込むのが安全である【計算】。

**R-10. 小さな点**

- §2.6 の manifest の `stored_tensors` は、キーが `<projection>.<suffix>` の完全名で、各値に `torch_dtype` と `shape` を持つ (`v41_exl3.rs:466-487`)。仕様の書き方は省略形なので、組み立てツールはこの形に合わせる。manifest の上限は 64MiB (`v41_exl3.rs:15`) で、47,232 projection では約 20MB と見込まれ収まる【計算】。
- 量子化器は `apply_out_scales: None` (自動) で、svh に出力側の scale が入りうる (`quantize_experts_exl3_sm120.py:38-46`、`quantize.py:1197-1199`)。ds41rt は宣言 `out_scales: "never"` だけを検査し、値は有限かどうかしか見ない (`v41_exl3_residency.rs:109-117`) ので動作には影響しない。ただし L0 refit と L2 の後も宣言は "never" のままにする必要がある。
- §2.3 の「凸なので貪欲法で最適」は、制約が予算だけのときに正しい (全 expert の増分が同じ 4.42MB なので、重要度順に選べば最適)。`decode_weighted_K ≤ 2.45` を加えると 2 制約のナップサックになり、貪欲法の最適性は保証されない。

### 7.3 検算で正しいと確かめた項目

- 大きさ: 1 expert 35.39M、計 543.6B、MXFP4 288.8GB、K2→K3 の増分 4.42MB (1 rank 2.21MB、1GiB あたり 485 本)、案 A/B/C の本数と bpw と合計 (147.0 / 151.3 / 155.6GB)、RTX1 の追加枠 `10.11 − KV`【計算】。
- suh/svh の 342M param (0.68GB)、expert 別の完全な H 1.94TB、層共有 H_x 4.2GB、expert 別の対角 0.46GB【計算】。
- ds41rt の k23 混合: projection ごとの `bits_per_weight` (`v41_exl3.rs:451-455`)、checkpoint 全体の tier 集合 (`:170-183`)、tier ごとの payload が実数ぶんだけ確保されること (`v41_exl3_residency.rs:222-231`)、1 層 1 tier の本数が 512 未満という制約 (`:206`、384 本なので満たす)、起動スクリプトが `bits` から `k23` を選ぶこと (`run.sh:169-172, 200-202`)【資料】。DSpark の EXL3 は TP1 で 1 枚に載る (`ds41rt-daemon/src/v41_experts/exl3.rs:82`)【資料】。
- 学習した suh/svh を ds41rt がそのまま使えること: 6 本のベクトルを fp16 のまま別バッファに読み (`v41_exl3_residency.rs:233-290`)、非有限値だけを拒否する (`:109-117`)。v0 の時点で suh は符号だけでなく `in_channel_scales / codebook_scale` の大きさを持っている (`quantize.py:1217`) ので、kernel が大きさを使っていることは v0 が正答する時点で確かめられる【資料 + 推測】。
- 局所蒸留の勾配: 復元は `W = diag(su) · H Q H · diag(sv)` の順 (`quantize.py:1397-1401`) で、su と sv について線形であり、勾配が流れる。trellis (離散) には流れない。gate/up の svh は clamp の前、down の suh は活性化の後に掛かる。up の sv と down の su は clamp の外では積しか効かない冗長な自由度なので、refit 値からの L2 正則化は必須である【計算】。
- ハーネスは trellis の形から K を読むので (`harness/stream_ref.py:223-227`)、expert ごとに K が混ざった pack でもそのまま採点できる【資料】。
- Tensorfold の数値 (共有基底 4.19GB = 0.062bpw、R(D) との差 0.067 bit、greedy→Viterbi 0.53 bit)【計算】。

### 7.4 結論への影響

- 本線の判断 (静的 K2/K3 混合、L0 最優先、本格 QAD は不可、Tensorfold 型の圧縮は不採用、日本語の評価を分ける) は変わらない。
- 変わるのは時間とディスクで、L0 と L1 の GPU 時間は 1.5〜2 倍、X-6 は組み立て方式を変えないと実行できない。
- 最初の GPU 作業として、Viterbi の 1 shard 測定 (数分) と DSpark の EXL3 化を X-3 の前に入れることを勧める。
