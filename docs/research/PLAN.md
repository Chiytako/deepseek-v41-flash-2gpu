# DeepSeek-V4.1-Flash-1017 を 2x RTX PRO 6000 + RAM 128GB + NVMe 1 本で動かす配置計画

作成日: 2026-10-18 / 対象: `deepseek-ai/DeepSeek-V4.1-Flash-1017`
前提資料: 同じフォルダの `web_engines.md`、`engram_techreport.md`、`local_images.md`、`arch_spec.md` (以下 WE / ET / LI / AS)。
見積もりスクリプト: `research/tmp/plan/estimate.py` (`python3 -I estimate.py` で表の数値を再現できる)。IOPS 測定は `research/tmp/plan/iops.py` と `bw.py`。
凡例: 【実測】はこの調査で測った値、【資料】は出典付きの値、【計算】は式で出した値、【推測】は根拠が間接的な値。

---

## 0. 結論

1. **全重み 510GB のうち、置き場が決まらないのは Engram ではなく routed experts 288.8GB である。**
   - Engram 表 202.8GB は NVMe と RAM の階層に置ける。読む量は 1 トークンあたり 12.4KiB、行番号は入力トークンだけで決まるので先読みできる。
   - experts は 1 ステップで数十 GB を読む。そのため c16 の速度を出すには、全部を VRAM に置くしかない (§2.3 の案 C で示す)。
   - VRAM の expert 予算は 2 枚で約 153GB【計算】。384 experts を残すなら平均 **約 2.2 bit/重み** が上限になる。
2. **推奨は案 A。** 中身は次の 4 点。
   - 1017 から experts を **EXL3 の平均約 2.2bpw** に自作量子化する。よく使われる experts には多くのビットを配る。全 experts を VRAM に置く。
   - Engram は **FP8 のまま外さない**。NVMe 上の全表と、RAM の 60GB 行キャッシュの 2 段に置く。
   - DSpark を使う。
   - エンジンは本家 vLLM + EXL3 プラグインから始める。
   - 見込みは decode c1 90〜140 tok/s、**c16 合計 500〜700 tok/s** (GLM-5.3 の現行 384 の 1.3〜1.8 倍)、prefill 5〜9k tok/s、1 本 1M 文脈【推測】。
3. **最大のリスクは 2.2bpw での品質低下である。** 参考値として、V4.1-Flash の EXL3 では 2.0bpw の KL が 0.36、3.0bpw が 0.055 だった【資料 WE §3.2】。
   - そこで、案 B2 (REAP で 192 experts に刈り込み、FP4 のまま LIL b12x で動かす) を同じ評価基盤で並べて比べる。
   - 比べる前に、層ごとに重みを NVMe から流して元の重みのまま推論する「参照ハーネス」を作る。これで教師 logits とルーティング分布を取る (実験 X6)。
4. **既存エンジンをそのまま使う案 (E) は、どれも 2 GPU では成り立たない。**
   - LIL `ds41-flash` を TP2 にすると、experts だけで 1 GPU あたり 144GB になる。
   - SGLang は Engram を disk に置けない。
   - LibertAIDAI 1017-EXL3-2.9bpw は VRAM に入りきらず、約 46GB を host に出すことになる。c16 は 120〜320 tok/s に落ちる【計算】。
   - ただし E2 (2.9bpw + host experts) は**品質が最も良く、最短で動かせる比較基準**として価値がある。
5. **Engram を活かすための要点は 4 つ。**
   - 行キャッシュはページキャッシュではなく**アプリ側の行単位**で持つ。4KiB のページに当たり行は 1 本しかなく、ページキャッシュでは容量効率が 1/15 になるため。
   - NVMe 上の表は **4KiB 境界に揃えて並べ直す**。現状は 16 行に 1 行がページをまたぐ【実測】。
   - prefill の行は**受付時点で先読み**し、decode の行はドラフト確定時点で先読みする。
   - prefix cache に当たった区間は Engram を引かない。
   - NVMe の 4K ランダム読み出しは**約 23 万 IOPS が上限**だった【実測】。キャッシュが無いと prefill は約 6.7k tok/s で頭打ちになり、ヒット率 65% なら約 19k tok/s まで上がる【計算】。decode では IOPS は問題にならない (c16 でも 2.7〜7.7 万 IO/s)。

---

## 1. 前提値 (見積もりの根拠)

### 1.1 ハードウェア

| 項目 | 値 | 根拠 |
|---|---|---|
| VRAM | 97,887 MiB = 102.6GB/GPU。利用率 0.95 で 97.5GB | GLM 運用時の nvidia-smi 値 (メモ) |
| GDDR7 実効帯域 | 1.45TB/s/GPU (公称 1.8 の 80%) | 【推測】 |
| PCIe H2D と GPU 間 | 約 20GB/s/GPU (Gen5 x8) | 課題文、GLM 実測 (P2P 20.6GB/s、host→GPU 20〜21GB/s) |
| DDR5 | 実効 約 50GB/s。MemTotal 127.4GB (121.5GiB) | 課題文、`/proc/meminfo` 【実測】 |
| RAM の現状 | used 85GiB、/dev/shm 49GB (GLM の LMCache)、zram swap 121.5G のうち 14G 使用 | `free -g`、`df -h /dev/shm`、`lsblk` 【実測】 |
| **NVMe 4K ランダム読み (O_DIRECT)** | QD1 で 12.7k IOPS (78µs)、QD8 で 100k、QD32 で 259k (123µs)、**QD64 以上で約 227〜231k (281µs) が頭打ち** | `iops.py` (Python スレッド、GLM の重みファイル 8 本)。4 プロセス×16 と 8 プロセス×16 でも合計 227k と 231k で、ドライブ側の上限と判断した【実測】。ダウンロードの書き込みと並行した値なので、単独なら少し上がる可能性がある |
| NVMe 1MiB ランダム読み | 5.6〜6.2GB/s | `bw.py` 【実測】 |
| NVMe 空き | 884GB | `df -h` 【実測】 |
| CPU | 7950X 16C/32T、AVX-512、AMX なし | 課題文 |

### 1.2 モデル側 (AS / LI の集計)

| 項目 | 値 |
|---|---|
| routed experts (本体) | 288.78GB。1 expert 18.80MB (MXFP4 4.25bpw)、層あたり 7.22GB |
| DSpark experts | 7.22GB (128×3 層) |
| experts と Engram を除いた重み | 11.5GB。TP2 で 1 GPU あたり約 6.9GB (wq_b / wo_a / wo_b / shared / embed / head を分割し、wq_a / wkv / gate / hc / Engram wkv / ViT を複製)【計算】 |
| Engram 表 | 202.76GB (weight 196.6 + scale 6.14)。2 ファイル (shard 47/48) |
| KV | 890B/token と、窓 2.9MB/本。4M トークンのプールで 3.6GB/GPU (MQA なので両 GPU に複製) |
| decode の重み読み (B=1) | 13.0GB/token |
| prefill の FLOPs | 1 トークン 34〜36 GFLOP。CED により decoder 層 (20〜39) は末尾 128 トークンだけ計算するので、**実質 約 18.5 GFLOP/token** (encoder 20 層 + Engram 射影 + 索引器)【計算】 |

**CED が prefill を半減させることの確認**: LIL の実装は decoder 側をリクエストごとに `kept = min(q, WINDOW=128)` 行だけ計算する (`vllm/models/deepseek_v4_1/ced.py:19, 179, 186`、`nvidia/model.py:654-675`)。TR §2.2 の「prefill は encoder だけ、活性 8B」と一致する。SWA の Bounded Replay は TR では劣化「ほぼ無し」とされている (ET §2.4)。

### 1.3 Engram が載る位置

- **層 1 と 14 はどちらも encoder 側**にある。そのため prefill では全トークンについて 2 層とも引く必要がある。
- 圧縮 ID とハッシュは入力トークンだけで決まる (AS §4.1)。

**Engram の safetensors 上の位置**【実測: headers.json とファイルサイズ】

- データの開始位置は shard 47 で 664B、shard 48 で 672B。
  - 計算: ファイル末尾 101,535,150,936 からテンソル末尾 101,535,150,272 を引く。shard 48 も同様。
- 256B の行は `664 + 256r` から始まる。始まりを 4096 で割った余りは {152, 408, …, 3992} を巡回する。
  - 余りが 3992 の行だけが 4KiB の境界をまたぐ。**16 行に 1 行は 8KiB 読み (2 IO) になる。**
- scale (8B/行) は 8B に揃っていて、ページをまたがない。

### 1.4 見積もりの式 (`estimate.py`)

```
1 step 時間 = 固定費 + (VRAM 上の expert バイト/2 + 密層 3.6GB + head 0.66GB + DSpark 0.8GB) / 1.45TB/s × kq
          + 0.5 × (B × 34 GFLOP / 2 GPU / 120 TFLOPS) + host 側 expert バイト / host 帯域
1 層で触れる expert 数 = E × (1 − (1 − 6/E)^B)    (B = 1 step で検証する総トークン数)
c1: B=6 (DSpark block 5 + 1)、受理長 τ=3.5    c16: B=64 (adaptive で 1 本あたり約 4)、τ=2.8
固定費: b12x 系 9ms、本家 vLLM + EXL3 15ms    kq: MXFP4=1.0、EXL3=1.25 (trellis 復号の負担)
```

**較正**: この式は公開実測とよく合った (`estimate.py` を条件だけ変えて計算)。

| 構成 | 指標 | 式の値 | 公開実測 | 判断 |
|---|---|---|---|---|
| LIL TP4 ネイティブ | C1 | 250 | 249 | 一致 |
| LIL TP4 ネイティブ | C8 | 695 | 805 | 一致 |
| nktlabs 2.0bpw (2x96GB) | c8 | 500 | 450 | 一致 |
| nktlabs 2.0bpw (2x96GB) | c1 | 140 | 80 | 式が高すぎる |

nktlabs の c1 が式に届かないのは、Engram の同期 preadv など固定費が大きいためと推測する。以下の表では、式の値を上限とし、実効の見込みは幅で示す。

---

## 2. 配置案の比較

### 2.1 一覧

| 案 | 中身 | VRAM/GPU | host RAM | c1 tok/s | c16 合計 tok/s | prefill tok/s | 最大文脈 | 品質リスク | 実装コスト |
|---|---|---|---|---|---|---|---|---|---|
| **A (推奨)** | 384E を EXL3 平均 2.15〜2.25bpw (使用頻度でビット配分)、全部 VRAM。Engram は FP8 で NVMe + RAM 60GB 行キャッシュ。本家 vLLM + EXL3 | 94〜97GB | 約 90GB | 90〜140 (式 137) | 500〜700 (式 701) | 5〜9k | 1M/本 (プール 4M tok) | 中〜高 (2.2bpw) | 高 (量子化 + Engram 階層パッチ、2〜3 週) |
| A' | A と同じ重みを LIL b12x に載せる (2bit expert カーネルを自作) | 94〜97GB | 約 90GB | 150〜190 (式 192) | 650〜850 (式 852) | 8〜10k | 1M | A と同じ | 非常に高 (カーネル + ローダ、4〜8 週) |
| B1 | REAP で 256E に刈り込み、EXL3 3.0bpw、全部 VRAM | 約 89GB | 約 90GB | 90〜127 | 450〜640 | 5〜9k | 1M (KV に余裕あり) | 高 (刈り込み +14% PPL 級 + 3bpw) | 高 |
| B2 | REAP で 192E に刈り込み、**FP4 のまま**、LIL `ds41-flash` を TP2、Engram disk | 約 93GB | 約 30GB | 130〜165 (式 162) | 550〜740 (式 743) | 5〜7k (Engram の IOPS で律速) | 1M | **非常に高** (50% 刈り込み) | 中 (刈り込み + LIL の TP2 検証) |
| C | experts を FP4 のまま、上位 53% を VRAM、28% を pinned RAM (ゼロコピーか CPU 計算)、残り 19% は NVMe か刈り込み | 約 97GB | 約 110GB (限界) | 15〜28 | 30〜60 | 約 1.4k | 1M | なし〜中 | 高 (x86 で V4.1 の host expert を扱うエンジンが無い) |
| D | Engram の置き場の比較。A / A' / B に組み合わせる (§2.4) | — | 6〜100GB | — | — | 上限 6.7k〜67k | — | FP8 なら無し | — |
| E1 | LIL `ds41-flash` を TP2、1017 original、Engram disk (そのまま) | 154GB で**起動不可** | — | — | — | — | — | — | — |
| E2 | LibertAIDAI 1017-EXL3-2.9bpw。experts の約 23% (46GB) を pinned host、Engram を NVMe (nktlabs prestage) | 約 97GB | 約 95〜105GB | 50〜100 | 120〜320 | 3〜6k | 1M | **低** (code PPL +1.8%) | 低〜中 (数日) |
| (参考 F) | GPU 4 枚で LIL TP4、ネイティブ、Engram は disk+RAM | 約 77GB | 約 30〜90GB | 229〜249 | C8 805 / C32 1,220 | 13.7〜18k | 1M | なし | 機材 (X670E の 2 スロットでは不可) |

- c1 と c16 は §1.4 の式の上限と、較正から見た実効の幅。GLM-5.3 の現行値は c1 224、c16 384 (メモ)。
- B2 の c16 が A を大きく上回らないのは、FP4 の 4.25bit を読むため。A は約 2.2bit なので、触れる expert 1 個あたりのバイトが半分で済む。
- F は LIL の実測 (WE §2.2、§4)。

### 2.2 各案の VRAM と RAM の内訳 (1 GPU あたり、GB)【計算】

| 内訳 | A / A' | B1 | B2 | E2 | C |
|---|---|---|---|---|---|
| routed experts | 73.1 (146.1/2) | 68.0 | 72.2 | 76.3 (VRAM 側) | 76.3 (FP4 の 53%) |
| DSpark experts (FP4) | 3.6 | 3.6 | 3.6 | 約 2.5 (3bit) | 3.6 |
| experts 以外の重み | 6.9 | 6.9 | 6.9 | 6.9 | 6.9 |
| KV プール 4M tok と窓 16 本 | 3.6 | 3.6 (増やせる) | 3.6 | 3.6 | 3.6 |
| 活性化 / graph / 作業領域 | 7.0 | 7.0 | 7.0 | 7.0 | 7.0 |
| **合計 (上限 97.5)** | **94.2** | 89.1 | 93.3 | 約 96.3 | 97.4 |
| host RAM | Engram 行キャッシュ 60、scale 6.1、エンジン 15、OS 8 → **約 89** | 同左 | Engram scale 5.7、エンジン 15、OS 8 → 約 30 | host experts 46、Engram キャッシュ 20、scale 6、23 → 約 95 | host experts 80、scale 6、23 → 約 109 |

- いずれの案も **GLM-5.3 (glm53v) の停止が前提**。VRAM が足りないうえ、/dev/shm の 49GB と GLM の RSS を解放する必要がある。
- zram swap があるので、pinned や mlock の領域が swap 対象にならないことを確認する (§4 の X8)。
- 1M 文脈は 1 本あたり 0.94GB なので、4M トークンのプールで「1M を 4 本」または「256k を 16 本」を同時に保持できる。1M×16 本を同時に持つには 15GB/GPU が要り、expert の予算は約 1.9bpw に下がる。現実的ではない。

### 2.3 案ごとの根拠と注意点

**案 A (EXL3 約 2.2bpw、全 experts を VRAM)**
- 予算: 2×(97.5 − 6.9 − 3.6 − 7.0 − 3.6) = 152.7GB。288.78GB×b/4.25 ≤ 152.7 から b ≤ 2.25【計算】。安全側の目標は 2.15bpw (146GB) とする。
- ビット配分の方針【推測】
  - 実トラフィックでよく使われる experts と、品質に効く層 (先頭と末尾の数層) は 3bpw にする。
  - 使われない experts は 1.6〜2bpw にする。
  - LibertAIDAI 1017 版も「層 18〜22 だけ 2bpw」という層別配分をしている (WE §3.1)。
- decode
  - c1 は 1 step で 1 GPU あたり約 12GB を読む → 式で 25.5ms、τ 3.5 で 137 tok/s。
  - c16 は B=64 で 1 層あたり 244 experts に触れ、1 GPU あたり 49GB → 式で 64ms、701 tok/s。
  - nktlabs の 2.0bpw 実測 (c8 450、c1 80) と同じ系統の数値になる (WE §4)。
- prefill
  - 計算上限は、CED で 18.5 GFLOP/token、EXL3 の大きな M で 1 GPU あたり 60〜100 TFLOPS 実効と見て 6.5〜10.8k tok/s。
  - Engram の IOPS 上限はヒット率 65% のとき 19k (§2.4)。
  - diffbot は 46K で 10k tok/s を出している (WE §4)。見込みは 5〜9k。
- 必要なもの
  - 1017 用の EXL3 量子化 (exllamav3 の V4.1 対応。LibertAIDAI が初日に 1017 版を出しているので道具はある)。
  - 本家 vLLM nightly + EXL3 プラグイン (diffbot / nktlabs と同じ系統)。
  - Engram 階層のパッチ: nktlabs の prestage を、非同期 io_uring + 行キャッシュ + 並べ直した表に置き換える。
- ブロッカー
  - 本家 vLLM の SM120 で、DSpark と sparse-MLA prefill の組み合わせが落ちる不具合 (#56771, WE §2.1)。
  - EXL3 プラグインが 1017 と DSpark の組み合わせで動くか。
  - 2.2bpw の品質。

**案 A' (A の重みを LIL b12x で)**
- LIL には V4.1 専用の部品が揃っている: CED、`compressed_sparse_mla`、Engram disk、DSpark K7、PCIe allreduce、ViT。固定費が小さい (較正では 9ms)。
- 足りないのは 2bit 前後の MoE カーネルだけ。
  - SM120 のテンサーコアは FP4 / FP6 / FP8 までなので、2bit 符号は FP8 にレジスタ内で展開して `mma.sync e4m3` にかける。GLM の W4A8 と同じ作り方になる (メモ: 生の mma で 870 TFLOPS、FP4→FP8 の LUT 展開も検証済み)。
- A で品質と構成が固まってから移す第 2 段階とする。

**案 B (刈り込み)**
- B2 は **既存の LIL b12x 経路をそのまま使える唯一の案** で、Engram disk、CED、DSpark も含む。
  - 192E×40×18.8MB = 144.4GB で予算に収まる。
- 品質
  - V4.1 では REAP 256E (33% 削減) で text PPL が +14.1% になった (WE §3.2)。
  - 50% 削減ではさらに悪化する見込み。用途がエージェント型のコードに偏っているなら、自分のトラフィックで較正すれば許容範囲に入る可能性はある【推測】。
- 未検証: b12x の MoE と DSpark が E=192 で動くか。
  - 192 にしたのは 64 の倍数を想定したため。272E は vLLM の expert 数テーブルに無くてロードできなかったという報告がある (WE §3.2)。
- B1 (256E × 3bpw) は A と速度がほぼ同じで、品質が刈り込みと量子化の二重に劣化する。A と B2 の中間の比較点として残す。

**案 C (ホット experts を VRAM、コールドを host)**
- RAM が足りない。
  - VRAM に 152.7GB を置くと、残りは 136GB になる。
  - Engram と OS を除くと RAM に置けるのは約 80GB。残りの約 56GB は NVMe か刈り込みに回るしかない。
- 速度が出ない【計算】。
  - DeepSeek 系は noaux_tc の bias で expert の負荷を均すよう学習されている (AS §1.7)。このため、host 側に当たる割合 h はほぼ容量比 (0.25〜0.5) になると見るのが安全である。
  - c1 (投機なし) は zero-copy でも CPU 計算でも 14〜28 tok/s。
  - DSpark を入れると検証 6 トークン分の experts を読むので、かえって遅くなる (式で 19 tok/s)。
  - c16 は 1 step で host 側の experts を数十 GB 読むことになり、30〜60 tok/s。
- この案は単独では採らない。役割は「刈り込んだ experts の退避先 (めったに当たらない experts だけ RAM に残す)」に限る。

**案 E (既存エンジンをそのまま)**
- E1 (LIL TP2 + original): launcher は TP2 + disk + original の組み合わせを通す (LI §2.1) が、experts が 1 GPU あたり 144GB で入らない。
- SGLang (dev-cu13 / ormandj): Engram は device か host RAM だけで、disk が無い (LI §1.2)。このマシンでは不可。
- E2 (LibertAIDAI 2.9bpw + host experts)
  - experts は約 199GB で、VRAM 予算を約 46GB 超える。
  - 速度は h (host 側に当たる割合) で大きく変わる。
    - h=0.02: c1 97 / c16 322
    - h=0.10: c1 49 / c16 116【計算】
  - diffbot (V4.1-Flash、host 30%、Engram は pinned) の実測は agentic 1 本 159、4 並列 232 (WE §4)。実際の h は小さめと考えられる【推測】。
  - prefill: チャンクごとに host experts の 46GB を PCIe (2×20GB/s) で流すので、1 チャンク約 1.15s。8k チャンクで約 7k、4k チャンクで約 3.5k tok/s が上限。
  - **品質が最も良く (2.9bpw)、量子化の手間が無い**。参照と比較の基準として最初に立ち上げる価値がある。ただし 196GiB の追加ダウンロードが要る。
- ik_llama.cpp の GGUF (Q2_K 246GiB) や llama.cpp
  - SM120 で MXFP4 を GPU に載せると NaN になる不具合 (#2519) がある。
  - CPU に置く experts が多く、c16 の用途には向かない (WE §2.5)。比較対象から外す。

### 2.4 Engram の置き場 (D) の比較【計算 + 推測】

**前提**
- 1 トークン 48 行。prefill のチャンク内で重複しない行は 67% (自前の測定、ET §1.5)。
- NVMe は 230k IOPS (QD64 以上)。
- ページをまたぐ行の分 ×1.0625 (並べ直せば ×1.0)。

| 方式 | RAM | NVMe IO/token (prefill) | prefill の IOPS 上限 | decode への影響 | 品質 | 判断 |
|---|---|---|---|---|---|---|
| D0: すべて RAM (vLLM `cpu_offload`、SGLang host table) | 203GB | 0 | ∞ | なし | なし | **入らない** |
| D1: NVMe だけ (LIL disk、O_DIRECT、scale は RAM) | 5.7GiB | 34.2 | **6.7k** | c16 で 7.7 万 IO/s、層 1 の待ちは 1 step 0.3ms 程度 | なし | B2 では最初これ |
| D2: NVMe mmap + ページキャッシュ 60GB | 60GB | 約 21 (実効の行容量は約 1/15 で、ヒット 35% 程度) | 約 10k | 同上 | なし | 非推奨 (容量効率が悪い) |
| **D3: NVMe (4KiB 境界に並べ直した表) + アプリ側の行キャッシュ 60GB (FP8、頻度順)** | 約 66GB | 11.3 (ヒット 65% のとき) | **約 20k** (ヒット 50% で 14k) | c16 で 2.7 万 IO/s | なし | **推奨** |
| D4: Engram を MXFP4 (約 104GB、136B/行、30 行/4K) にして NVMe + 60GB キャッシュ | 約 64GB | 3〜6 (ヒット 80〜90% のとき) | 34〜67k | 小 | 低 (cos 0.9934、GSM8K 同等。WE §3.3) | prefill を重視するときの第 2 候補 |
| D5: Engram 3bit (約 80GB) をすべて RAM | 約 80GB + 他 | 0 | ∞ | なし | 不明 (高い) | 品質の検証次第 |
| D6: VRAM に上位行 ≤2GB | VRAM 2GB | — | — | ヒット 30〜37% | なし | 非推奨 (VRAM は experts に回す) |

**D3 の詳しい設計**
- **並べ直した表**
  - 1 行は weight 256B と scale 8B を合わせた 264B。1 ページに 15 行 (3,960B) を詰めて、ページ境界にそろえる。ファイルは約 209.7GB。
  - 1 行が 1 IO で済み、scale を RAM に置く必要も無くなる。
  - 行は頻度順に並べる。行番号の付け替え表は 768M×4B = 3.1GB で、RAM に置く。
  - 元の safetensors はそのまま残す。
- **行キャッシュ**
  - 実トラフィックで頻度を測った上位の行を固定で持ち、さらに LFU で動的に更新する。
  - 2-gram の区間は 1 層あたり 33.8GB で、ここに当たりが集中している (2-gram のヒット率は 68%、ET §1.5)。60GB あれば 2-gram の頻出行はほぼ全部入る【推測】。
- **先読み**
  - prefill: リクエストを受け付けた時点でプロンプト全体のハッシュを出し、ずっと先のチャンクまで IO を出しておく (ET §1.6、AS §4.4)。prefix cache に当たった区間は引かない。
  - decode: サンプリングとドラフトの確定直後に IO を出す。
    - 層 14 の分は、層 0〜13 を計算している間に隠れる。
    - 層 1 の分は、層 0 の計算と重ねる。QD64 での遅延は 281µs で、1 step のうち 0.3ms 程度が表に出る。
- **GPU への転送**は 1 トークン 12.4KiB だけ。c16 でも 1 step 約 0.8MB で、PCIe の負担にはならない。
- **ヒット率 65% は推測値**である。実際のヒット率は実験 X2 で、GLM の要求ログ (19GB、エージェント型、日本語を含む) を V4.1 の tokenizer で再トークン化して測る。

---

## 3. 推奨

**採用する順番**

1. **E2 で比較の基準を作る** (任意。ダウンロードの許可が要る)。
   - 品質の良い 2.9bpw で、API、ツール、画像、DSpark が動く経路を最短で確認する。
2. **本命は案 A** (EXL3 約 2.2bpw、全 experts を VRAM、Engram D3、DSpark)。
   - ビット配分は、実トラフィックでの expert 使用頻度と、教師 logits に対する KL で決める。
3. **B2 を同じ評価で比べる。**
   - 刈り込み率を 192 / 256E と変え、LIL b12x で動かす。
   - LIL の SM120 V4.1 経路と、Engram disk の実性能を確かめる目的も兼ねる。
4. A の品質が合格したら、**A' (LIL b12x への移植)** を検討する。c1 の速度を GLM 並み (150〜190 tok/s) に近づけるためである。

**Engram について**: 推奨構成ではすべて FP8 の表を変更せずに使う。量子化も刈り込みも Engram 以外 (experts) で行う。
- 根拠: Engram を外すと PPL は code 1.87→4.16、prose 6.43→9.06 に悪化する (WE §3.1)。事実知識系の性能は 29〜44% しか残らない (ET §1.3)。
- MXFP4 化 (D4) は、prefill の IOPS が足りないと分かった場合の予備に回す。

**運用上の注意**
- GLM-5.3 とは同時に動かせない。VRAM と RAM の両方が足りないため。
- 切り替えると、GLM の再起動 (6〜9 分) と DS の起動 (JIT とロード。本家 vLLM は初回の autotune に 75〜90 分という報告がある、ET §5.3) がかかる。

---

## 4. 実験計画 (優先順)

| ID | 目的 | 方法 | GPU | 全重み | 成功基準 |
|---|---|---|---|---|---|
| X1 | 重みの完全性と Engram の配置を確認する | 48 本の sha256 を HF の blob と照合する (47/48 は LibertAIDAI カードの値 `76a05106…` / `9dc51433…` とも照合)。safetensors のヘッダ長から Engram のデータ開始位置 (664 / 672B) を確かめる | 不要 | 要 | 全一致。開始位置が §1.3 と一致する |
| X2 | **Engram のヒット率を実トラフィックで測る** (D3 の容量を決める) | `research/tmp/engram_sim.py` を拡張する。`glm53/dump/reqlog` (19GB) のプロンプトを V4.1 の tokenizer で再トークン化し、半分で頻度表を作り、残り半分でヒット率を測る。キャッシュ 1 / 8 / 32 / 64GB、n-gram 次数別、チャンク内の重複除去率も出す | 不要 | 不要 (tokenizer のみ) | 曲線が取れること。**64GB でヒット 55% 以上**なら D3 を採る。40% 未満なら D4 (MXFP4) を併用する |
| X3 | NVMe の実 IOPS (io_uring) を測る | ダウンロード完了後、fio (未導入なので pacman での導入をユーザーに確認) か liburing のツールで shard 47 を対象に測る。4K randread、QD 16〜256、`--direct=1`、30 秒 | 不要 | shard 47 | QD64 で 200k IOPS 以上、p99 1ms 未満。下回れば prefill の目標を下げるか D4 にする |
| X4 | Engram 表の並べ直しと MXFP4 化 | CPU のツールで、頻度順・ページ境界揃え (15 行/4K) の表と付け替え表を作る。ランダムな 100 万行で元の表とビット一致を確かめる。MXFP4 版も作り、cos と相対 L2 を出す | 不要 | 47/48 | ビット一致。MXFP4 の cos が 0.99 以上 (NVIDIA 版 0.9934 と同等) |
| X5 | GPU の基本値 | GLM を止めた後に、`shared_memory_per_block_optin`、tilelang 0.1.12 の sm_120 / FP4 コンパイル、参照 kernel.py の単体実行を確かめる | 要 | 不要 | 値が取れる。参照カーネルが動くか、動かない箇所が特定できる |
| X6 | **層ストリーミングの参照ハーネス** (品質評価の基準) | 参照 `model.py` を層単位で NVMe から GPU に載せて (1 層 7.4GB、6GB/s で約 1.2s)、元の重みのまま forward する。Engram は CPU で gather する。これで次の 3 つを出す。(1) 実トラフィック 50〜100 万トークンでの expert 使用頻度 (層×expert)、(2) REAP の saliency、(3) 評価用プロンプトの教師 top-k logits | 要 (1 枚) | 要 | 短いプロンプトで自然な貪欲出力が出る。頻度表と教師 logits が保存できる |
| X7 | 量子化と刈り込みの品質比較 (A / B を決める) | EXL3 を一律 2.0 / 2.25 / 2.5 / 3.0bpw で作り、X6 の頻度を使った混合配分版も作る。REAP 192E / 256E (FP4) も作る。各版を教師に対する KL と top-1 一致、GSM8K-200、HumanEval、日本語 QA、エージェント要求の再生 (`glm53/replay_openai.py` のサンプル)、needle 128k で比べる | 要 | 要 | **experts 152GB 以内で KL ≤ 0.10 の構成**が見つかれば A を採る。無ければ E2 方式 (host experts) と比べて判断する |
| X8 | 案 A を立ち上げてベンチを取る | 本家 vLLM nightly + EXL3 プラグイン + Engram D3 パッチ + DSpark。c1 / c4 / c16、prefill 8k / 32k / 128k、1M needle、画像、ツール呼び出しを測る。pinned 領域と zram の関係も確かめる | 要 | 要 | **c16 で 450 tok/s 以上 (GLM の 384 を超える)**、c1 100 以上、prefill 5k 以上、1M needle 合格、NaN なし |
| X9 | B2 を LIL で立ち上げる (SM120 b12x TP2 の検証を兼ねる) | REAP 192E の checkpoint を作り (router / bias / bias_vl を切り出し、n_routed_experts を書き換える)、`ds41-flash` を TP=2、`engram-table-memory disk`、`seccomp=unconfined`、`memlock -1` で起動する | 要 | 要 | ロードと推論ができる。c16 と prefill の実測が §2.1 の幅に入る。Engram disk の実 IOPS が取れる |
| X10 | Engram の効果を 1017 で確認する | X8 の構成で、Engram FP8 / MXFP4 / Engram を切った状態を、知識 QA と code の PPL で比べる | 要 | 要 | FP8 と MXFP4 の差が誤差の範囲。Engram を切ると明確に悪化する (外さない判断の裏付け) |
| X11 | DSpark の受理長を測る | X8 と X9 で、実トラフィックの τ と adaptive verification の検証長を記録する | 要 | 要 | τ 3 以上 (c1)。式の前提値 (3.5 / 2.8) を実測値で置き換える |
| X12 (任意) | A' のカーネル試作 | b12x の fused_moe に 2bit 符号の FP8 展開経路を足す。GLM の w4a8 プロトタイプを流用する | 要 | 一部 | 単体で、帯域の床の 70% 以上 |

- 実施順の目安
  - ダウンロード完了前に CPU で X2 → ダウンロード後に CPU で X1 / X3 / X4
  - **GLM の停止はユーザーの判断**。止めた後に X5 → X6 → X7 → X8 / X9 → X10 / X11
- X6 と X7 は GPU 時間が長い (EXL3 の量子化は 1 版あたり数時間〜十数時間【推測】)。GLM を止めておく時間を、まとめて確保する必要がある。

---

## 5. 未解決事項

- expert の使用頻度にどれだけ偏りがあるか (X6)。偏りが大きければ、A の混合配分の効果が大きくなる。C や E2 の host 側に当たる割合 h も小さくなる。
- 本家 vLLM の SM120 TP2 で、DSpark と長いプロンプトを組み合わせたときの安定性 (#56771)。EXL3 プラグインが 1017 と DSpark の組み合わせに対応しているか。
- LIL b12x が E≠384 (刈り込み後) を受け付けるか。Engram disk に行キャッシュや並べ直した表を足せるか。b12x の DiskTable はライセンス上流用できるか (vLLM 部分は Apache-2.0 の SPDX ヘッダ)。
- LIL #736 は 4 GPU + NVMe Engram で 32K prefill 13.7k tok/s を出している (WE §2.2)。これは D1 の IOPS 上限 6.7k (NVMe 1 本) を超える。NVMe の本数、重複除去、内部キャッシュの有無が分かっていない。
- 1017 の CED Bounded Replay (末尾 128 トークンの近似) による長文脈での品質。TR では「ほぼ無し」とされている。
