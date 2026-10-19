# r3: QAD (quantization-aware distillation) と蒸留系の量子化を、この機材で 1017 に適用できるか

作成日: 2026-10-18
対象: `deepseek-ai/DeepSeek-V4.1-Flash-1017` の routed experts (40 層 × 384、1 expert 35.39M param、合計 543.6B param)
前提資料: `PLAN.md`、`DECISION.md`、`r2_quant_tools.md`、`r2_ds41rt.md`、`r2_lil_lowbit.md`、`r2_engram_hit.md`、`arch_spec.md`
見積もりスクリプト: `<scratchpad (非公開)>/r3/qad_calc.py` (`python3 -I` で表の数値を再現できる)
方針: GPU は使っていない (量子化と教師取得で使用中)。docker コンテナには触れていない。重みはダウンロードしていない。
凡例: 【実測】は手元で確かめた値、【資料】は出典付きの値、【計算】は式から出した値、【推測】は根拠が間接的な値。

---

## 0. 結論

**「QAD は可能か」への答えは、レベルによって分かれる。**
NVIDIA 型の本格的な QAD (重みそのものを学習する end-to-end 蒸留) は、この機材では不可能である。
一方、EXL3 の形式を変えずに連続成分だけを学習する「層単位の局所蒸留」は可能で、計算時間も数時間で済む。

| レベル | 中身 | この機材で | GPU 時間 (2 枚)【計算 / 推測】 | 期待効果 | 推論コスト |
|---|---|---|---|---|---|
| L0 | **Viterbi 再符号化 + scale refit (H=I)**。データを使わない | **可。最優先** | 3〜6 h | relerr 0.377→0.262、NLL 0.243→0.139 (0909 での実績)【資料】 | 0 (形式もバイト数も同じ) |
| L1 | **Hessian 校正** (LDLQ + H 重み付き refit)。日本語とエージェントの校正データを使う | **可** | 4〜8 h | KL がさらに 2〜4 割下がる【推測】 | 0 |
| L2 | **層ごとの局所蒸留**。suh/svh (と shared expert) を教師の MoE 出力に合わせて学習する | **可** (実装 2〜3 日) | L1 に +0.5〜1.5 h | 2bit 誤差の 1〜2.5 割を追加で回復【推測、QuIP# の FT 効果から】 | 0 |
| L3 | **end-to-end の scale-only KD** (QuIP# の第 2 段に相当。suh/svh、norm、shared を最終 logits の KL で学習する) | 条件付きで可 (実装 1〜2 週) | 1M tok × 5 epoch で約 7〜10 h と教師 0.5〜2 h | L2 の上に小幅【推測】 | 0 |
| L4 | **本格 QAD** (全重みの学習、PV-Tuning の離散コード更新、数億〜十億トークン) | **不可** | GLM 型 (1.5B トークン) で 80 日超【計算】 | — | — |

1. **最初の一手は L0 である。** 現在走っている量子化は `--greedy --beam 16` で、relerr は約 0.377 になる。
   - 同じスクリプトから `--greedy` を外すだけで、tail-biting Viterbi (`quantize_tiles`) に切り替わる。
   - sfxnz の `refit_identity` (`build/qrecipe/tools/requant_probe.py:49`) を後段に足せば L0 になる。
   - 0909 ではこれだけで NLL が −42%、GSM8K は 95→97、MMLU は 196→206 になった (§3.1)。
   - 出力は ds41rt が要求する形式 (K2、MCG、同じテンソル名) のままである。
2. **ds41rt の推論カーネルに追加コストなしで載る学習対象は、suh / svh (expert ごとの fp16 ベクトル、合計 342M param、0.68GB)、shared expert (FP8)、RMSNorm、router、mHC の係数である。** (§2.3)
   - ds41rt は suh / svh を checkpoint の値のまま expert ごとに読み込む (`ext/ds41rt/rust/crates/ds41rt-loader/src/v41_exl3_residency.rs:233-290`)。
   - LoRA のような低ランク補正は、カーネルの追加が必要になるので「追加コストなし」には入らない。
3. **end-to-end QAD が非現実的な理由は、教師の速度よりも学生の学習経路にある。**
   - 教師の forward は、現行ハーネスの 40〜70 tok/s から、層ストリーミング (sudoingX 方式) にすれば 1M トークンあたり 15〜30 分程度まで速くできる見込みである【推測】。
   - 律速は学生側にある。2bit の experts だけで 137GB あり、GPU 1 枚に入らない。学習可能な EXL3 の backward カーネルも無い。
   - NeMo AutoModel の V4.1-Flash の LoRA レシピですら H100 64 枚を前提にしている【資料】。
4. **日本語性能は、校正データに日本語を入れるかどうかでほぼ決まる。**
   - sudoingX の公開校正データ (892 会話) のうち、日本語を含む会話は 10 件、仮名は全文字の約 0.06% しかない【実測】。
   - sudoingX は「薄い領域ほど KLD の差が広がる (multilingual 1.34 倍)」こと、そして「データを足せば差は閉じ、Hessian 内での重み付け変更は追加効果なし」だったことを報告している【資料】。
   - L1〜L2 では、GLM の要求ログ (日本語 9%、エージェント型) と日本語の補充データを 30% 程度混ぜる。L2 の損失では日本語トークンの重みを上げる (§4)。
   - **元の 1017 を上回る日本語性能**は、教師に合わせる蒸留では原理的に得られない。そのためには日本語 SFT などの別の学習が要る (§4.3)。

---

## 1. QAD と蒸留系量子化の現状 (公開事例)

### 1.1 手法と規模の一覧

| 手法 | 学習するもの | 損失 | 規模と所要 | 2bit 級の効果 | 出典 |
|---|---|---|---|---|---|
| **NVIDIA ModelOpt QAD** (GLM-5.3-Flash-FP4-Local) | routed experts を NVFP4 で量子化し、量子化を戻した値で forward する。勾配は BF16 の潜在重みを更新する。router、shared experts、dense MLP、注意 (MXFP8) も学習する。norm は固定 | 教師の次トークン確率と最終 hidden state | **500M トークン × 3 epoch**。教師 (BF16) が応答を生成する | 4bit: GPQA 92.1→91.3、ToolEvalBench 91.9→92.4 | `$MODELS/GLM-5.3-Flash-FP4-Local/README.md:83-128`【実測: 手元のカード】 |
| ModelOpt QAD (Qwen3.6-35B-A3B W4A4) | 同上 | BF16 教師に対する蒸留 | 500 iter × 32K 文脈、GB200 128 枚で 5.7 h (**約 735 GPU 時間**) | W4A4 で劣化した IFBench 57.9→59.6 (BF16 60.0)。劣化が無いベンチでは変化なし | [ModelOpt announcement](https://nvidia.github.io/Model-Optimizer/announcements/qwen36-w4a4-qad.html) |
| NVIDIA DeepSeek-V4.1-Flash-NVFP4 | MXFP4→NVFP4 の変換 (routed のみ) | — | QAD の記載は見つからない (PTQ と見られる) | — | [datanorth 記事](https://datanorth.ai/news/nvidia-releases-deepseek-v4-1-flash-nvfp4)、[V4-Flash-0731-NVFP4 カード](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4) |
| Kimi K2 Thinking | MoE 部分の INT4 weight-only QAT (後学習の段階) | 不明 | 1T param / 384 experts。開発元の学習基盤で実施 | 報告値はすべて INT4 で測定 | [HF カード](https://huggingface.co/moonshotai/Kimi-K2-Thinking) |
| **QuIP# FT** (EXL3/QTIP の系譜) | 符号は固定。**SU/SV を実数に緩和**し、未量子化の層、layernorm、lm head を学習する。第 1 段: ブロック内 (ブロック出力の MSE)。第 2 段: 全体 (CE) | 層出力 MSE → 全体 CE | 256 系列 (RedPajama) × 5 epoch。70B で約 50〜100 GPU 時間 (A100) | Llama-2-70B 2bit W2 ppl: NoFT 4.16 → FT 3.91 (FP16 3.12)。7B: 8.22 → 6.19 | [arXiv 2402.04396](https://arxiv.org/abs/2402.04396) §5、Table 4、App. D、F.6-F.7 |
| QTIP | 純計算符号 (EXL3 の MCG と同系統) は FT なし。ハイブリッド符号は符号表と未量子化重みをブロック単位で FT | 同上 | 同上 | 70B 2bit: NoFT 3.90 (3INST) → FT 3.70 | [arXiv 2406.11235](https://arxiv.org/abs/2406.11235) Table 3, 5 |
| AQLM + PV-Tuning | 連続パラメータ (P ステップ) と**離散コード** (V ステップ、変化の大きい一部だけ再割り当て) を交互に更新 | 全体 KD | 70B 2bit で W2 3.78 (1x16g8、18.8GB) | 2bit で Pareto 最適 | [arXiv 2405.14852](https://arxiv.org/abs/2405.14852)、[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/091166620a04a289c555f411d8899049-Abstract.html) |
| EfficientQAT | Block-AP (ブロックごとに全重みと量子化パラメータを学習) → E2E-QP (step size だけを全体で学習) | ブロック MSE → 全体 CE | 2bit Llama-2-70B を A100 1 枚で 41 h | 精度 69.48 vs FP 72.41 | [arXiv 2407.11062](https://arxiv.org/abs/2407.11062) |
| sudoingX (V4.1、0909) | 蒸留はしない。**順次の層ストリーミング校正** (量子化済みの前段を通した入力で、routing 重み付き Hessian を取る) と expert 別 K | 層ごとの重み空間誤差 | 853K トークン、2x GB10 で pass1 約 13 h | 2.70bpw で KLD 0.0243 (Mia 2.9 は 0.0295) | `ext/sudoingx_dsv41/RECIPE.md:89-104`、`scripts/quantize/p4_pass1.py:1-30` |
| sfxnz (V4.1、0909) | 蒸留も校正もしない。Viterbi で再符号化し、suh/svh を閉形式で refit (H=I) | 重みの Frobenius 誤差 | 46,080 テンソル、2x GB10 で 12.61 h | NLL 0.243→0.139 | `ext/sfxnz_recipe/results/RESULTS.md:866, 1001-1006`、`model-card.md:24` |

**500B 級 MoE で QAD をした公開事例と所要**
- 確かめられた中で最も近いのは NVIDIA の GLM-5.3-Flash (320B MoE) の QAD で、500M トークン × 3 epoch である。GPU 時間は公表されていない。
- Qwen3.6-35B-A3B の 735 GPU 時間 (GB200) を、活性パラメータ比 (GLM 18B / Qwen 3B) と同じトークン数で単純に伸ばすと、数千 GPU 時間級になる【推測】。
- V4.1-Flash については、NeMo AutoModel の LoRA レシピが H100 64 枚 (8 ノード × 8)、全体 fine-tune が 128〜512 枚だった【資料: [NeMo AutoModel の V4.1-Flash ページ](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/deepseek-ai/DeepSeek-V4.1-Flash)】。QAT / QAD のレシピは載っていない。

### 1.2 公開事例から読み取れること【推測】

1. **符号 (離散部分) を固定したまま連続成分を学習するだけで、2bit の誤差はかなり戻る。**
   - QuIP# の 70B 2bit では、FP16 との差 1.04 のうち約 24% (0.25 ppl) が FT で戻った。7B では約 65% 戻った。
   - 大きなモデルほど、FT なしでも差が小さく、FT の伸びしろも小さい。
2. **符号の質 (Viterbi か greedy か) の方が、FT より効く場合がある。**
   - QTIP は「FT なしの QTIP が、FT ありの QuIP# / AQLM をほぼ全サイズで上回る」としている (Table 3)。
   - sfxnz の Viterbi 化 (NLL −42%) も、データを使わずに大きく効いた例である。
3. **QAD が大きく効くのは、PTQ で実害が出ている項目だけである。** Qwen3.6 の例では、劣化の無い 4 ベンチは QAD で変わらなかった。
   - このことから、評価は日本語・エージェント・コードの区分別に行い、劣化した区分を狙って重みを付けるべきである。

---

## 2. この機材での可否

### 2.1 律速の所在

**教師の forward**【実測 + 計算】
- 参照ハーネス `harness/stream_ref.py` は、系列ごとに 40 層すべての forward を行う。そのたびに、触れた experts を NVMe から LRU に読み込む (`stream_ref.py:156-263`)。
- 4,096 トークンの系列は、全 384 experts × 40 層にほぼ全部触れる。つまり 1 系列ごとに experts 288.8GB を読む計算になり、NVMe 6GB/s で約 48 秒かかる。
- 実測は 1 系列 51〜96 秒 (43〜79 tok/s) だった (`work/teacher/teacher.log`)。律速は expert の IO である。
- このままの方式で 2 GPU を使ったときの教師取得時間は、1M トークンで 2.0 h、5M で 9.9 h、100M で 198 h になる【計算】。

**層ストリーミングにした場合の教師**【推測】
- sudoingX の Streamer (`ext/sudoingx_dsv41/scripts/quantize/dsv41q/stream.py:35-160`) は、全系列の residual stream (`[L, hc=4, 5120]` bf16、40KB/token) を GPU に置く。そして 1 層ずつ、全トークンをまとめて処理する。
- この方式なら experts の読み込みは 1 パスあたり 288.8GB を 1 回だけで済む (約 50 秒)。
- 計算量は 34 GFLOP/token なので、1M トークンでは 3.4×10^16 FLOP になる。実効 50 TFLOPS で約 11 分。
- したがって **1M トークンの教師 top-k は、GPU 1 枚で 15〜30 分程度**になる見込みである。
- 上限は GPU メモリで決まる。stream は 1M トークンで 41GB を占めるので、1 枚あたり 1〜1.5M トークンが上限になる。

**学生の学習 (end-to-end)**【計算 + 推測】
- 学生の experts は 2bit でも 137GB あり、1 枚に入らない。層を 2 枚に分けるパイプラインになる。
- EXL3 には backward の GEMM (重みの転置で掛ける経路) が無い。そのため、触れる expert をそのつど bf16 に戻して計算する。
  - 全 543.6B 重みを戻すと、1 パスあたり数秒 (300G 重み/s と仮定して 1.8 s) かかる【推測】。
- 4K トークンの 1 step は、forward・再計算・入力側の backward で 34 GFLOP × 4,096 × 3 = 4.2×10^14 FLOP。実効 50〜100 TFLOPS で 4〜8 s。復元と Engram を足して **1 step 約 15〜25 s、約 200 tok/s** になる。
- 規模ごとの所要
  - QuIP# 規模 (1M トークン × 5 epoch) なら約 7 h で、**現実的な範囲に入る**。
  - GLM の QAD 規模 (1.5B トークン) なら約 87 日で、**不可**。
- 全重みを学習する場合は、543.6B param の勾配と Adam 状態 (fp32 で数 TB) が要る。**この機材では、どの規模でも不可**である。

### 2.2 層単位の局所蒸留 (本命の L2)

**設計**
- sudoingX の pass1 と同じく、層 l の MoE 入力 `x` (`hc_pre` → `ffn_norm` の後、[N, 5120]) と routing (top-6 の番号と重み) を、全校正トークンについて GPU に置く。
- 教師の出力 `y_T` は、ネイティブの MXFP4 experts (層あたり 7.2GB) で同じ `x` から計算する。
- 学生の重みは `W = diag(su) · Q̃ · diag(sv)` と書ける (`exllamav3_5be8/exllamav3/modules/quant/exl3_lib/quantize.py:1397-1401`)。
  - `Q̃ = H_k Q H_n` は、trellis を su=1、sv=1 で復元したものである。
  - Q̃ を 1 回だけ bf16 で作っておけば、学生の forward は「要素ごとの積 → bf16 GEMM → 要素ごとの積」になる。
  - 1 層 384 experts の Q̃ は 27.2GB【計算】。64 experts ずつ処理すれば 4.5GB で足りる。
- 学習するのは su (gate/up 5,120、down 2,304) と sv (gate/up 2,304、down 5,120) である。SiLU と clamp(±10) を通した 3 射影を一緒に学習する。
- 損失は、expert ごとに routing 重みを付けた出力誤差 `Σ_t g_t² ‖E_T(x_t) − E_S(x_t)‖²` とする。
  - これは expert 間で分離できるので、expert 単位で並列に学習できる。
  - 層単位の損失 (`‖MoE_T − MoE_S‖²`、shared expert も込み) にすれば、shared expert も学習対象に加えられる (§2.3)。
- 入力の取り方は 2 通りある。
  - (A) sudoingX と同じく、学生の stream を入力にし、同じ入力に対する教師出力を目標にする。局所誤差だけを消す方式。
  - (B) 教師の stream も並べて持ち、目標を教師の層出力にする。上流の誤差も吸収できるが、stream が 2 本要り (1M トークンで 82GB)、過学習の危険も大きい。
  - **(A) から始めることを勧める。**
- backward が要るのは活性側の勾配だけである。重みの勾配 GEMM は不要なので、1 epoch は forward の約 2 倍の計算量になる。

**計算量** (`qad_calc.py`)【計算】
- routed の forward は 425 MFLOP/token/層 (6 experts × 3 射影 × 2×5120×2304)。

| 校正トークン N | 1 expert あたりの平均トークン | 1 層の時間 (教師 1 回 + 4 epoch、実効 80〜150 TFLOPS) | 40 層 | GPU メモリ (stream + x + y_T) |
|---|---:|---:|---:|---:|
| 256K | 4,000 | 6.5〜12 s | 4〜8 分 | 10.5 + 2.6 + 5.2 = 約 18GB |
| 512K | 8,000 | 13〜25 s | 9〜16 分 | 約 37GB |
| 1M | 15,625 | 26〜48 s | 17〜32 分 | 約 72GB (1 枚ではぎりぎり) |

- これに、層ストリーミングで注意側を流す時間 (1M トークンで 40 層合計 3〜6 分【推測】) と、Engram の gather が加わる。
- **L1 の再符号化 (§3) の時間と比べると小さい。** L1 の pass の中に、層ごとの「符号化 → 局所蒸留 → stream を進める」として組み込むのが効率がよい。
- 過学習への対策
  - 下位の experts は平均の数分の 1 しかトークンを持たない (下位 30% の experts に入る routing は約 7%、`r2_quant_tools.md §4.2`)。
  - 学習するのは 22,272 param/expert で、データ (1 トークンあたり 5,120 次元の出力) に比べて十分小さい。
  - それでも、検証用の 10% を取り分けて早期終了を入れる。refit の初期値からの L2 正則化も入れる。
  - QuIP# の設定 (Adam、su/sv の lr 5e-4、他 5e-5、検証集合で早期終了、§F.6) に従う。

### 2.3 学習できる連続成分と、ds41rt の推論での追加コスト

| 成分 | 形 (1 層あたり) | 学習方法 | ds41rt / EXL3 の推論 | 判定 |
|---|---|---|---|---|
| **suh** (expert ごとの入力側 scale と符号) | gate/up [5120]、down [2304]、fp16 | L0 / L1 は閉形式の refit、L2 / L3 は勾配 | ds41rt は `gate_suh` / `up_suh` を別バッファで、down の suh を intermediate 側のバッファで読む (`v41_exl3_residency.rs:233-290`)。TP2 では intermediate 方向を切り出す (`v41_exl3_staging.rs:71-74`) | **追加コストなし** |
| **svh** (expert ごとの出力側 scale) | gate/up [2304]、down [5120]、fp16 | 同上。gate/up の svh は clamp の前に掛かる | 同上。sfxnz の refit で「svh が実際の列ごとの scale になった」pack が、同じ kernel で動いている (`RESULTS.md:1005`) | **追加コストなし** |
| 「Hadamard 後の scale」 (g_scale) | スカラー | — | 量子化時に su に畳み込まれ、別の値として保存されない (`quantize.py:1227-1228`) | su に含まれる |
| expert ごとの大域 scale | — | — | ds41rt の `unit_scales` は 1.0 固定で、checkpoint からは読まない (`v41_exl3_residency.rs:248`) | svh で代用 |
| **shared expert** (FP8、32×32 ブロック、UE8M0 scale) | 35.4M param | 層単位の損失で、FP8 丸めを STE で通す QAT | 公式と同じ FP8 形式なら、ds41rt の raw 経路で受け付ける。ds41rt が検査するのは形式 (`non_routed_quantization`) で、値の hash は照合しない (`r2_ds41rt.md §2.1, §2.2`) | **追加コストなし**。ただし e4m3 の丸め幅は相対 6〜12% あり、小さな更新は消える【推測】 |
| RMSNorm (`ffn_norm`) | [5120] bf16 | 勾配 | 同上 | 追加コストなし。ただし router の入力も変わる |
| router (`gate.weight` bf16 [384,5120]、bias) | — | 勾配 | 同上 | 追加コストなし。routing が変わり、DSpark の受理率や負荷分布にも影響するので後回しにする |
| mHC 係数 (`hc_*` fp32) | — | 勾配 | 同上 | 追加コストなし。効果は小さい見込み【推測】 |
| trellis (離散コード) | 2bit × 35.4M | Viterbi / LDLQ の再実行 (PV の V ステップ) | 形式は不変 | 学習ではなく再符号化になる。1 回 3〜6 h |
| LoRA / QLoRA 型の補正 (expert ごと、または層で共有) | rank r | 勾配 | **ds41rt には経路が無い**。MoE の後ろに低ランク GEMM を足す改修が要る。decode では層ごとに r×(5120+5120) の読み出しが増える | **追加コストあり**。採らない |
| expert の bias | — | — | V4.1 の experts には bias が無い (`arch_spec.md:28`) | 対象外 |

**まとめ**
- ゼロコストで載るのは、suh / svh (合計 342M param、0.68GB【計算】) と、非 expert の FP8 / bf16 / fp32 テンソルである。
- QuIP# が学習した集合 (SU/SV、layernorm、未量子化の層) とほぼ同じになる。
- 確認が要る点: ds41rt の native kernel で、gate/up の svh が clamp の前に掛かっているか。数学的にはそうでなければならず、sfxnz の refit pack が同系統の kernel で一致検査に通っている (`RESULTS.md:991`、kernel の参照比較 24/24)。ds41rt では、`--dry-run` の後に層 0 の参照比較で確かめる。

---

## 3. 既存の校正手順をこの機材で回す手順と所要時間

### 3.1 sfxnz の Viterbi + scale refit (L0)

**中身【資料】**
- exllamav3 の `quantize_tiles` (tail-biting Viterbi) で符号化する。
- その後、`refit_scales` を H=I に特化した閉形式 (行と列の scale を交互に最小二乗で解く) で refit する。refit は relerr が悪化しない場合だけ採用する (`build/qrecipe/tools/requant_full.py:11-16`、`requant_probe.py:49-71`)。
- 46,080 テンソルすべてが改善した。relerr は 0.3775 → 0.2616、MSE 比は 0.480 (`RESULTS.md:1004`)。
- 0909 での結果: NLL 0.2434 → 0.1386 (40 本中 39 本で低下、Wilcoxon p=5.5e-12)。GSM8K 95/94 → 97/97、MMLU 196/198 → 206/204、tools exact args 84 → 88/88。decode の ms/step は変わらない (`model-card.md:24`、`RESULTS.md:994`)。

**速度 (GB10 で実測、`results/2026-09-24-review/campaign/s1-microbench/summary.json:37-38`)**

| 符号化 | exllamav3 1.4.9 | 1.5.1 |
|---|---:|---:|
| greedy beam16 | 1.06 s/tensor | 1.03 s/tensor |
| Viterbi | 4.54 s/tensor | 2.24 s/tensor |
| Viterbi + refit | 4.27 s/tensor | 1.97 s/tensor |

**当機への換算**【実測 + 計算】
- 当機の greedy beam16 は 4.60 tensor/s/GPU (0.217 s/tensor) だった (`build/quant-logs/quant-g1-20-39.log`)。GB10 の 1.03 s/tensor の 4.7 倍速い。
- 同じ比で換算すると、Viterbi は 1.4.9 (手元の `5be88657`) で約 0.97 s/tensor、1.5.1 で約 0.48 s/tensor になる。
- 全 46,080 テンソルを 2 GPU で処理すると、**1.4.9 で約 6.2 h、1.5.1 で約 3.1 h**【計算】。
- 1.4.9 と 1.5.1 の Viterbi は relerr が同じ (0.2619 / 0.2617) なので、品質のために image を作り直す必要は無い。
- 手元の 5be8 には `refit_scales` が無い (`quantize.py` を grep しても該当なし)。そのため `requant_probe.py` の `refit_identity` (純 torch) を持ち込む。

**手順** (GPU が空いてから)
1. `build/quantize_1017.sh` の `gpu` 分岐から `--greedy --beam $BEAM` を外した版を作り、`DST=DeepSeek-V4.1-Flash-1017-EXL3-2.0bpw-vit` に出力する。
   - スクリプトは `--greedy` が無ければ `quantize_tiles` (Viterbi) を使う (`quantize_experts_exl3_sm120.py:256-330`)。
2. `quantize_experts_exl3_sm120.py` の `_quantize_group` の後段に refit を足す。
   - 処理: trellis から復元し、`refit_identity(w, q)` で r と c を求め、`suh *= r`、`svh *= c` とする。relerr が下がったときだけ採用する。
   - または、出来上がった pack を `requant_full.py` の encode 部分 (`:469-560`) を移植して後処理する。
3. ハーネスで採点する: `stream_ref.py --teacher-in work/teacher/{agent24,ja16}.jsonl --teacher-ref <教師.pt> --exl3-pack <dst>`。区分別の KL と top-1 一致を、greedy 版と比べる。

### 3.2 exllamav3 の Hessian 校正 (LDLQ、L1)

- `quantize_exl3(weight, H_data, …)` は、H の LDL 分解 → `regularize` (H の対角から out scale を判断) → `ldlq` (16 列の帯ごとに Viterbi をかけ、誤差を後ろの帯へ補償) の順に処理する (`quantize.py:1242-1420`)。
- 現行のスクリプトは単位行列の H を使い、LDLQ を通らない高速経路 (all-tiles fallback) を使っている (`quantize_experts_exl3_sm120.py:9-11, 124-180`)。
- **H を実データの値にするには、活性を取る仕組みが要る。** 手元で最も近い実装は sudoingX の pass1 である (§3.3)。

### 3.3 sudoingX のパイプライン (校正、L1 の土台)

**中身【資料】** (`scripts/quantize/p4_pass1.py:1-30`)
- 層 l の stream は、量子化済みの層 0..l−1 を通したものを使う。
- expert e の Hessian は次の 2 つ。
  - `H_gu = Σ (g x)(g x)ᵀ + τ·mean(g²)·H_all/N` (全トークンの事前分布を混ぜ、冷たい expert でも条件を保つ)
  - `H_d = Σ a aᵀ`、ここで `a = g·silu(gate)·up` はネイティブの expert の中間活性
- 符号化後に、量子化した expert の出力で stream を進める。
- `--part i/n` で expert を複数機に分け、part ファイルを交換する。

**当機に移すための改修** (r2_quant_tools §8-4 に、今回の調査分を加えた)
1. codebook を `mcg` にする。`dsv41q/exl3.py:36` の `quant_args(..., codebook="mul1")` を、`encode_expert` (`:43`) から mcg で呼ぶようにする。ds41rt は MCG を必須にしている。
2. 1 回の実行で全層を K2 にする (ds41rt の raw は一律 K、`r2_ds41rt.md §2.1`)。mixed-K は staged manifest に進むときに考える。
3. `DSV41_CUDA_MEM_FRACTION` (既定 0.70、`stream.py:39`) を 0.95 程度にする。
4. Engram の reader を vLLM の拡張から切り離す (`refmodel.py:198`)。当ハーネスの `HostEngramEmbedding` (`stream_ref.py:106-153`) を流用する。
5. `--part 0/2` と `1/2` で GPU0 / GPU1 の 2 プロセスにし、part の交換はローカルの FS で行う。
6. 出力を sfxnz / diffbot の形式 (テンソル名 `layers.L.ffn.experts.E.wN.{trellis,suh,svh,mcg}`) に並べ直す。plugin 用の名前からの変換になる。
7. L2 を足す: 各層で「符号化 → refit (H 重み付き) → su/sv の局所蒸留 (§2.2) → stream を進める」の順にする。

**所要**【推測】
- sudoingX の 2x GB10 での実測は、pass1 が約 20 分/層、40 層で約 13 h (`RECIPE.md:89-104`)。
- GB10 と当機の符号化速度の比 4.7 倍 (§3.1) と、GEMM 帯域の比 (約 6 倍) から、**2 GPU で 3〜5 h**。pass0 と感度測定で 0.5 h、L2 で +0.5〜1.5 h が加わる。
- **合計で GPU 4〜8 h**。改修の実装は 2〜4 日【推測】。

### 3.4 Hessian の保存量の設計

| 方式 | 1 層 | 40 層 | 判断 |
|---|---:|---:|---|
| expert 別の完全な H (H_gu 5120² + H_d 2304²、fp32) | 48.4GB | **1.94TB** | 保存は不可。GPU にも 1 層分は載せない |
| 同上、上三角 fp16 | 12.1GB | 484GB | 不可 (空きは約 220〜360GB) |
| **層共有の H_x** (MoE 入力の共分散、5120² fp32) | 105MB | **4.2GB** | **保存する**。事前分布の項 τ·H_all と、refit のやり直しに使う |
| **expert 別の対角** (5120 + 2304、fp32) と、トークン数と routing 重みの和 | 1.1MB | **456MB** | **保存する**。mixed-K の配分と感度の推定に使う |
| 層入力 X そのもの (256K トークン、bf16) | 2.6GB | 105GB | 任意。層を並列に処理する (教師入力で一括量子化する) 場合だけ要る |

**推奨**
- expert 別の H は保存せず、各層の処理中に、GPU 上の X と routing から必要なときだけ作る (sudoingX と同じ)。
  - expert e の H_gu は `X_eᵀ diag(g²) X_e` で、n_e × 5120 の GEMM で済む。
- 保存するのは、層共有の H_x と expert 別の対角だけにする (合計約 4.7GB)。
- 校正データの領域ごとの重み付けは H には入れない。データを含めるだけにする。根拠は sudoingX の層 10 のプローブで、Hessian 内の領域の重み付けは「データを入れること」に対して追加効果が無かった (`FINDINGS.md:58-62`、`scripts/quantize/hessian_domain_probe.py`)。

---

## 4. 日本語性能

### 4.1 現状の問題【実測】

- sudoingX の校正データ (`data/caldata/conversations.jsonl` 500 件、`instruct_v3.jsonl` 392 件) を、JSON を戻したうえで数えた。
  - 仮名を 50 字以上含む会話は 10 件 (1.1%)。仮名は計 4,738 字で、全 737 万字の 0.06%。
- 現在走っている量子化は H=I で、データを使わないので、日本語が不利になることは無い。一方で、有利になることも無い。
- **L1 / L2 で公開の校正データだけを使うと、日本語は「薄い領域」になり、相対的に悪化するおそれが高い。** 根拠は sudoingX の multilingual で、KLD が他人の pack の 1.34 倍になった (`FINDINGS.md:56-60`)【資料】。

### 4.2 対策 (L1 / L2 に組み込む)

1. **校正データの構成案** (合計 0.5〜1M トークン。`work/x2/req` で抽出済みの GLM 要求ログを使う)

| 区分 | 割合 | 出典 |
|---|---:|---|
| 日本語 (チャット、文章、QA、thinking を含む) | 30% | 要求ログの ja 窓 (後半だけで prefill 1.35M + decode 1.22M トークン、`r2_engram_hit.md §2.5`)、`work/teacher/ja16.jsonl` の系統。評価用とは分ける |
| エージェント / ツール呼び出し | 30% | 要求ログ (`agent24.jsonl` の系統) |
| コード | 20% | 要求ログの code 窓と sudoingX の code |
| 英語ほかの多言語、数学 | 20% | sudoingX の `conversations` / `instruct_v3` |

   - 応答部分は GLM の生成文になる (1017 の生成ではない)。ハーネスで 1017 に生成させるのは decode が 4.6 tok/s しか出ず (`harness/run1.log`)、現実的でない。
   - 校正で必要なのは入力の統計 (H) と教師の出力なので、teacher-forced で通せば文体の違いは大きな問題にならない【推測】。
2. **日本語に偏った experts を守る。**
   - 教師の実行で得られる `STORE.counts` (層 × expert) を、ja16 と agent24 で別々に集計する。
   - 日本語でだけ多く使われる expert (全体では冷たいが日本語では熱いもの) を特定する。
   - mixed-K に進むとき (staged manifest) は、その expert を K3 にする。
   - L2 では、その expert の早期終了の基準を日本語トークンの誤差に置く。
3. **L2 / L3 の損失で、日本語トークンの重みを上げる** (例: 2 倍)。
   - 量子化誤差の予算を日本語側に寄せる効果がある。「一律量子化と比べて日本語を良くする」という意味での向上は、この方法で狙える【推測】。
4. **評価は区分別に、保留データで行う。**
   - 教師集合を ja / agent / code / general に分け、それぞれ 64 系列以上にする。現状は ja16 (採点 9,014 トークン) と agent24 (11,284 トークン)【実測】。
   - KL、top-1 一致、NLL を区分別に出す。
   - 実タスクは、日本語 QA と、要求ログの日本語エージェント会話の再生で見る。
   - G1 の「日本語 QA の低下 ≤3 点」(DECISION §4.2) で判定する。

### 4.3 「元の 1017 より良くする」ことについて

- 蒸留の目標は教師 (1017) なので、L0〜L3 で到達できる上限は 1017 と同等である。
- 1017 を超えるには、日本語の正解データ (人手か、より強い教師) に対する CE / SFT が要る。
- この機材でできるのは、L3 と同じ経路 (suh/svh、norm、shared だけを学習) で損失を日本語 SFT に替えることくらいである。
  - 学習できる自由度は 342M param 程度しかない。他の能力 (ツール、コード) が下がる危険の方が大きい【推測】。
- **勧めない。** 日本語を強めたい場合は、量子化とは別の課題として扱う (LoRA は ds41rt に経路が無い点にも注意)。

---

## 5. 「動的量子化」と「Tensorfold」との関係 (QAD の側から見た要件)

- **動的量子化** (活性を実行時に量子化する W4A8 / A8 のようなもの) は、速度のための手段で、品質を上げる手段ではない。
  - ds41rt の EXL3 経路は weight-only (bf16 活性) で、活性の量子化は入っていない。この課題の品質目標とは独立している。
- **expert ごとにビット数を変える構成** (静的な混合) は、ds41rt の staged manifest で隣り合う 2 段 (K2/K3) まで表せる。
  - residency は、expert × 射影ごとに tier を記述子 `(tier<<9)|local` で持つ (`v41_exl3_residency.rs:202-208`)。
  - ただし、実行中に精度を切り替える仕組みは無い。
  - 一律 K2 の raw 経路では、L0〜L2 の成果はそのまま使える。
- **Tensorfold 系の形式**を別途設計する場合、QAD の側から求める条件は次のとおり。いずれも【推測】による設計案である。
  1. expert ごとに K を変えられる (ragged per-expert)。
  2. suh / svh を fp16 の任意の値として持てる (符号だけの bit ベクトルにしない)。QuIP# が指摘するとおり、FT を前提にすると fp16 が必要になる (付加は 0.01 bit/重み未満)。
  3. trellis の codebook は MCG に固定する。ds41rt の kernel と量子化器を共用するため。
  4. 学習した shared expert と norm を、非 expert の shard に書き戻せる (hardlink ではなく実体のコピー)。

---

## 6. 推奨手順

| 順 | 段 | 中身 | GPU (2 枚) | 実装【推測】 | 判定 |
|---|---|---|---|---|---|
| 0 | CPU (今すぐ) | 校正データ (§4.2) と、保留の評価集合 (ja / agent / code / general 各 64 系列) を作る。`refit_identity` を量子化スクリプトに持ち込む | 不要 | 0.5〜1 日 | — |
| 1 | **L0** | 現行の greedy 版の量子化が終わったら、`--greedy` を外した Viterbi 版 + refit を作る (`…-2.0bpw-vit`) | 6.2 h (1.4.9) / 3.1 h (1.5.1) | 0.5 日 | ハーネスで greedy 版と区分別の KL を比べる (評価 1〜2 h)。ds41rt で G1 / G2 |
| 2 | **L1 + L2** | sudoingX の pass1 を当機と MCG 向けに改修する。日本語 30% の校正データで、LDLQ → H 重み付き refit → su/sv の局所蒸留 (日本語の重み 2 倍) → stream を進める | 4〜8 h | 3〜5 日 | L0 と比べて区分別の KL、特に ja と agent |
| 3 (任意) | shared expert の層単位補正 | L2 の損失を層単位にし、shared expert を FP8 の STE 付きで学習する | +1〜2 h | 2 日 | KL の改善がノイズを超えるか |
| 4 (任意) | **L3** | 2 GPU のパイプラインで、全体の KL を損失に、su/sv、norm、shared だけを学習する。教師は層ストリーミングで、1M トークンの top-32 を取る | 教師 0.5〜2 h と学習 7〜10 h | 1〜2 週 | L2 からの上積みが KL で 10% を超えるか |
| 5 (研究) | PV 型の離散更新 | 連続空間での勾配から目標の重みを作り、変化の大きい tensor だけ Viterbi を再実行する | 1 回 3〜6 h | 1 週以上 | — |
| × | 本格 QAD | 全重みの学習、億単位のトークン | 不可 | — | — |

**ディスク**
- 各版は experts の約 137〜150GB と、非 expert shard のコピーになる。現行の出力 (137GB) と並べると、1 版ずつ入れ替える運用になる。
- 版を比べ終えたら負けた方を消す (DECISION §5.2 の手順と同じ)。

**GPU の使い方**
- 段 1 と段 2 は、どちらも 2 GPU を数時間占有する。GLM と ds41rt の切り替え運用 (DECISION §6.1) に合わせて、まとめて時間を取る。
- 段 2 の pass の中で、評価用の教師 logits も同時に出せる。Streamer の `iter_logits` (`stream.py:161`) を使えば、ハーネスで別に取り直す時間 (1M トークンで 2 h) を省ける。

---

## 7. 未解決事項

1. 1017 で L0 がどれだけ効くか。0909 での NLL −42% は sfxnz の測定で、1017 では未確認。1017 と 0909 の量子化耐性はほぼ同じとされる (Lucebox の値、`r2_quant_tools.md §5`)。
2. ds41rt の native kernel で、gate/up の svh が clamp の前に掛かっているか。sfxnz の refit pack は p2b kernel で検査済みだが、ds41rt では未確認。
3. L2 (su/sv の局所蒸留) の効果。2bit で、Hessian 校正の後に上積みがどれだけ残るか。QuIP# の FT は、未量子化層の補償も含めた値である。
4. 日本語に偏った experts が実在するか。ja16 の教師実行の `STORE.counts` で確かめる。
5. sudoingX の pass1 を mcg に替えたときに、品質の傾向が変わらないか。sudoingX は mul1 だけで測っている。
6. 層ストリーミングの教師の実速度 (1M トークンで 15〜30 分という見込みの検証) と、Engram gather の待ち時間。
7. shared expert を FP8 の STE で学習するとき、e4m3 の丸めで更新が消えずに残るか。
8. 現行の greedy 版 (作成中) を、L0 の版ができた後に残すか消すか。

---

## Sources

- [NVIDIA Model Optimizer: Recovering W4A4 NVFP4 Accuracy with QAD (Qwen3.6-35B-A3B)](https://nvidia.github.io/Model-Optimizer/announcements/qwen36-w4a4-qad.html)
- 手元のカード: `$MODELS/GLM-5.3-Flash-FP4-Local/README.md` (FP4 QAD 1.0、500M トークン × 3 epoch)
- [QuIP#: arXiv 2402.04396](https://arxiv.org/abs/2402.04396)
- [QTIP: arXiv 2406.11235](https://arxiv.org/abs/2406.11235)
- [PV-Tuning: arXiv 2405.14852](https://arxiv.org/abs/2405.14852)、[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/091166620a04a289c555f411d8899049-Abstract.html)
- [EfficientQAT: arXiv 2407.11062](https://arxiv.org/abs/2407.11062)
- [Kimi K2 Thinking モデルカード](https://huggingface.co/moonshotai/Kimi-K2-Thinking)
- [NeMo AutoModel: DeepSeek V4.1 Flash](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/deepseek-ai/DeepSeek-V4.1-Flash)
- [datanorth: NVIDIA releases DeepSeek-V4.1-Flash-NVFP4](https://datanorth.ai/news/nvidia-releases-deepseek-v4-1-flash-nvfp4)、[nvidia/DeepSeek-V4-Flash-0731-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Flash-0731-NVFP4)
- 手元のリポジトリ: `ext/sfxnz_recipe` (`results/RESULTS.md`、`model-card.md`、`results/2026-09-24-review/campaign/s1-microbench/summary.json`)、`build/qrecipe/tools/{requant_full,requant_probe,quantize_experts_exl3_sm120}.py`、`build/exllamav3_5be8/exllamav3/modules/quant/exl3_lib/quantize.py`、`ext/sudoingx_dsv41` (`RECIPE.md`、`FINDINGS.md`、`scripts/quantize/p4_pass1.py`、`dsv41q/`)、`ext/ds41rt/rust/crates/ds41rt-loader/src/{v41_exl3_residency,v41_exl3_staging,v41_exl3}.rs`、`harness/stream_ref.py`、`work/teacher/teacher.log`、`build/quant-logs/quant-g1-20-39.log`
