# docs: 調査と結果

作業中に書いた調査の記録 (`research/`) と結果 (`results/`)。すべて日本語。書いた時点の判断と実測をそのまま残してあり、後の調査で覆ったものもある
(例: 初期の計画は vLLM 系を想定していたが、最終的に ds41rt を採った)。新しいものほど信頼できる。

## 公開版での置き換え

- 作業場の絶対パスは、作業場のルートからの相対パスにした。ルートそのものは `$DSV41`、重みの置き場は `$MODELS` と書いた。
  作業場の `tools/`・`harness/`・`build/` は、このリポジトリの `quant/` の下にある。`proxy/` はこのリポジトリの `proxy/`、`research/` と `results/` はこの `docs/` の下。
- GLM-5.3-Flash 時代の作業場 (`$GLM53`) と、そこにある要求ログ・スクリプトは公開していない。
- 作業中の一時置き場 (`<scratchpad (非公開)>`)、`work/` の中間生成物、`research/tmp/` (外部リポジトリの写し) も公開していない。
- Tailscale の IP は `<TAILSCALE_IP>` に、私的な要求ログのファイル名は `reqlog-X.log` に置き換えた。
- 私的な要求ログの中身 (利用者の発話、開発者指示、作業ディレクトリ、ツールの入出力) を引いていた箇所は削った (`r7_clients.md` §4.2 ほか)。件数などの集計は残してある。
- GLM 時代の要求ログを集計した記述では、クライアントの名前、system プロンプトの文面、ツールの一覧、利用した日時の細かい記述を一般化した (「コーディングエージェント型の実トラフィック (複数のクライアント)」、型 A〜F など)。件数と比率は残してある。
- 調査の記録で「QAD」と書いた段は、公開パックでは L2 の scale 校正 (各 expert の suh/svh だけを層ごとに教師の expert 出力へ合わせる局所蒸留) だけを指す。
  重みや離散コードは学習せず、モデル全体での end-to-end の蒸留もしていない。ファイル名はそのままにした。
- モデルの拒否・安全挙動を変えること (無検閲化) は範囲外として扱い、その文献整理と設計は公開していない。

## 結果 (`results/`)

| ファイル | 内容 |
|---|---|
| [QUALITY.md](results/QUALITY.md) | 量子化版 (v0 / v1 / L2 pilot / v1-l2e) の品質。参照ハーネスでの teacher-forced KL と top-1、中国語専用トークンの質量、画像 |
| [ENGINE_AB.md](results/ENGINE_AB.md) | ds41rt と vLLM (+ vllm-exl3) の速度の比較 |
| [L2E_DS41RT.md](results/L2E_DS41RT.md) | v1-l2e を ds41rt で確かめた結果 (速度、画像の正答率、生成の中国語混入) |

## 調査 (`research/`)

| 区分 | ファイル |
|---|---|
| 全体の計画と決定 | [PLAN.md](research/PLAN.md)、[DECISION.md](research/DECISION.md) |
| モデルの構造 | [arch_spec.md](research/arch_spec.md) (推論エンジン実装用の仕様)、[engram_techreport.md](research/engram_techreport.md)、[r2_engram_hit.md](research/r2_engram_hit.md) (Engram の行の当たり率)、[r3_routing_skew_measured.md](research/r3_routing_skew_measured.md) |
| エンジンの選定 | [web_engines.md](research/web_engines.md)、[local_images.md](research/local_images.md)、[r2_ds41rt.md](research/r2_ds41rt.md)、[r2_lil_lowbit.md](research/r2_lil_lowbit.md)、[r4_vllm_ab.md](research/r4_vllm_ab.md) |
| 量子化 | [r2_quant_tools.md](research/r2_quant_tools.md)、[r3_dynamic.md](research/r3_dynamic.md)、[r3_tensorfold.md](research/r3_tensorfold.md)、[SPEC_QUANT_JA.md](research/SPEC_QUANT_JA.md) (日本語性能を守る量子化の仕様) |
| scale 校正 (QAD) | [r3_qad.md](research/r3_qad.md)、[QAD_PLAN.md](research/QAD_PLAN.md) (実行計画)、[r6_vision_qad.md](research/r6_vision_qad.md) (画像を QAD に入れる設計) |
| 日本語と中国語の混入 | [r3_japanese.md](research/r3_japanese.md)、[r5_zh_mixing.md](research/r5_zh_mixing.md) (測り方と原因の切り分け)、[r5_serving_suppress.md](research/r5_serving_suppress.md) (配信時のマスク)、[SPEC_STEER_JA.md](research/SPEC_STEER_JA.md) |
| API プロキシ | [r7_clients.md](research/r7_clients.md) (クライアントが使う API)、[r7_ds41rt_api.md](research/r7_ds41rt_api.md)、[r7_options.md](research/r7_options.md)、[r7_proxy_spec.md](research/r7_proxy_spec.md) (ds41gw の仕様)、[r7_replay_real.md](research/r7_replay_real.md) (実要求の再生と追加遅延) |
