# r7: ds41rt の前に置く API 変換プロキシの実装手段の比較

作成 2026-10-19。対象は DeepSeek-V4.1-Flash-1017 (EXL3 2bit QAD) を載せた ds41rt v15 (`ghcr.io/tpurtell/ds41rt-coordinator:v15`、口 8000)。
根拠の種類は【実測】(このセッションで動かして確かめた)、【コード】(ソースを読んだ)、【文書】(公式文書・第三者記事)、【推測】で示す。

---

## 0. 結論

1. **推奨は (d) の自作プロキシ。Rust (axum + tokio + reqwest) で書き、DeepSeek 公式の `deepseek-recipe` 0.1.0 クレート (b) を部品として使う。**
   - recipe から使うのは 2 つ。Responses / Messages 要求を Conversation に変換する部分と、Responses / Messages の SSE イベントを組み立てる `ChunkGenerator` である。
   - ds41rt 自身が同じ recipe `=0.1.0` で Chat Completions を変換している (`ds41rt-api/Cargo.toml:9-11`、`native_v41.rs:10-18`)。前段と後段が同じ変換規則に揃う。
   - recipe が扱わない部分は自前で書く。previous_response_id の保存、JSON Schema と strict の受け渡し、model 名の別名、Chat の正規化、reasoning の再注入、認証、要求ログである。
2. **LiteLLM (a) は不採用とする。** 実機 (1.106.0、ds41rt に接続) で次の問題を確認した。
   - Responses のストリームが壊れている。`sequence_number` がほとんどのイベントで欠けており、値も単調でない。`reasoning_summary_part.added` が無い。message 項目が tool call 項目をまたいで開いたままになる。
   - `/v1/messages` は provider の書き方によって本文が常に空になる。
   - `previous_response_id` は Postgres が無いと黙って無視され、会話履歴が失われる。
   - 同時 16 本で追加遅延が TTFT p50 +23〜46 ms / p95 +190〜410 ms、1 要求あたり +130〜270 ms ある。「遅延 <5 ms」を大きく超える (§3.1)。
3. **(c) vLLM の Responses 実装の移植は割に合わない。** EngineClient、レンダラ、パーサと密結合しており、HTTP の前段には移せない。保存機能は「メモリリークする」と自ら注記している。HF の responses.js は reasoning も状態保持も未実装である (§3.3)。
4. Python (aiohttp) で書いても性能は足りる。変換する最小試作を同時 16 本で測ると、追加は TTFT +0.8 ms、全体 +1 ms だった【実測 §3.4】。
   - ただし recipe の Python バインディングは解析済みのチャンクを受け取れない (`InferenceChunk` は生テキストか token id のみ)。Python で書くと、イベント生成は全部自作になる。
   - そのためプロトコルの正しさを公式実装に寄せられる Rust を第一候補とし、Python は第二候補とする。
5. **Responses 以外にも直す箇所がある。** ds41rt の Chat Completions を旧クライアントがそのまま叩くと、次の 4 点が問題になる【実測 §2】。
   - model 名 `GLM-5.3-Flash` は 400 になる。
   - `chat_template_kwargs.enable_thinking=false` は黙って無視され、思考が続く。
   - ストリームのクライアントは temperature を送らないので、greedy になる。
   - 非ストリームのクライアントは履歴に reasoning_content を一度も返していない (61/61)。
   - プロキシは Chat の正規化も担う必要がある。

---

## 1. 実クライアントは何を送ってくるか

### 1.1 GLM 時代の要求ログ (SGLang reqlog、19GB, 173 ファイル。コーディングエージェント型の実トラフィック、複数のクライアント)【実測】

- **全体の件数**
  - `request.received.openai` (Chat Completions) は 31,914 件だった。
  - Responses 経由の要求 (rid が `resp_`) は 98 件だった。
  - 98 件は **全件がストリーム**で、ツール付きは 97 件だった。
  - `previous_response_id` という文字列は全ファイルに 1 度も現れない (`grep -l -F '"previous_response_id"'` の結果が 0 件)。
  - 注意: SGLang の Responses 経路はテンプレート適用後の text しかログに残さない。そのため要求本体のフィールドまでは見えない。
- **直近 6 ファイルの Chat 190 件の内訳** (集計スクリプトは非公開、内容は下表)
  - 次の 2 系統のクライアントが混在している (A / B は本稿だけの呼び名)。

| | クライアント A (115 件) | クライアント B (75 件) |
|---|---|---|
| stream | true (`stream_options.include_usage`) | **false** |
| temperature / top_p / top_k | **送らない** | 0.6 / 0.95 / 20 |
| reasoning_effort | high 110 / low 5 | 送らない |
| chat_template_kwargs | `{"enable_thinking":true,"thinking":true}` | `{"enable_thinking":true}` |
| 履歴の reasoning_content | 一部の assistant にだけ付ける (110 件が "some") | **一度も付けない (61/61)** |
| model | `GLM-5.3-Flash` | 同じ |
| tools | 188/190 件がツール付き、`tool_choice=auto`、`parallel_tool_calls=true`、`n=1` | 同じ |
| 画像 | 0 件 | 0 件 |

- vLLM 時代のログは `/v1/chat/completions` が 15,879 件で、`/v1/responses` は 0 件だった。
  - ただし、この期間は再生ベンチが中心である。

### 1.2 OSS のコーディングエージェント 1 つの Responses クライアント実装【コード】

OSS のコーディングエージェント 1 つの Responses クライアント実装を読んだ (公開版ではクライアント名とファイル名を伏せた)。

- 要求は `stream: true` 固定、**`store: false`** である。reasoning が有効なときは `include = ["reasoning.encrypted_content"]` を付ける。
  - つまり **毎ターン入力の全履歴を送る**。previous_response_id は使わない。
- 過去の reasoning 項目は、保存しておいた JSON のまま送り返す。
  - プロキシが出した reasoning 項目 (content / summary / encrypted_content) は、そのまま戻ってくる。
- `response.reasoning_text.delta` と `response.reasoning_summary_text.delta` の両方を解釈する。`output_item.done` では `summary || content` を思考本文として採る。
  - reasoning_text 形式 (recipe の出力) でも summary 形式 (LiteLLM の出力) でも読める。
- 参考: Codex CLI は 2026-02 に `wire_api = "chat"` を廃止した。今は Responses だけを話す【文書: [opper.ai](https://opper.ai/blog/codex-router)、[jan#7413](https://github.com/janhq/jan/issues/7413)】。
  - 今後 Responses 専用のクライアントが増える前提で作るべきである。

---

## 2. ds41rt 側の現状 (ds41rt に直接投げた結果)【実測、max_tokens 5〜400、同時 1 本】

| 入力 | 結果 |
|---|---|
| `model: "GLM-5.3-Flash"` | **400** `model must be deepseek-ai/DeepSeek-V4.1-Flash` (0.4 ms、`native_v41.rs:268-269`) |
| `n: 2` | 400 `only n = 1 is supported`。`n: 1` は通る |
| `logprobs: true` | 200 で `logprobs: null` が返る (黙って無視) |
| `presence_penalty: 0.5` | 200 (黙って無視) |
| `chat_template_kwargs: {"enable_thinking": false}` | **無視され思考が出る** (reasoning_content あり)。止めるには `thinking: {"type":"disabled"}` が必要 |
| `reasoning_effort: "medium"` | 200 (受理) |
| `role: "developer"` | **400** `unknown variant developer` (recipe の Chat スキーマに無い、`chat_completion/request/schema.rs:51-70`) |
| temperature 省略 | greedy (`native_v41.rs:158`) |
| 思考の既定 | high (`native_v41.rs:261` `with_default_thinking_mode(true)`) |

- **ストリームの形**
  - トークン単位で `reasoning_content` → `content` → `tool_calls` の差分が来る。
  - 並列ツールは index 0 の引数がすべて流れ終わってから index 1 が始まる (逐次)。
  - 引数は細切れの差分で来る。id は `call_chatcmpl-<uuid>_<n>` の形である。
  - 最後に `finish_reason: "tool_calls"`、続いて `[DONE]` が来る。
- **履歴の reasoning とキャッシュ**
  - ds41rt のエンコーダは、ツールがあるとき過去の assistant の思考を消さずに描画する (`encoding.rs:102` `drop_thinking = tools.is_empty() && ...`)。
  - 1 ターン後の要求で比べた結果は次のとおりで、reasoning を付けたほうが命中が 54 tok 多い (`proxy/r7_eval/cache_rt.py`)。
    - reasoning を付けない場合: prompt 1,566 tok のうち命中 1,515 tok
    - reasoning を付けた場合: prompt 1,582 tok のうち命中 1,569 tok
  - 欠けると、直前の生成ターンを毎回 prefill し直すことになる。加えて、モデルが生成時と違う文脈を見ることになる。
  - 後者は品質への影響があるとみているが、規模は未計測である【推測】。
  - 非ストリームのクライアント B は reasoning を返さない (§1.1)。プロキシ側で tool_call_id をキーに reasoning を保存し、再注入する価値がある。

---

## 3. 候補ごとの評価

### 3.1 (a) LiteLLM proxy 1.106.0 (2026-10-18 公開)

- **構成**: uv venv に `litellm[proxy]==1.106.0` を入れ、ds41rt の前に要求を記録する素通し (`proxy/r7_eval/tee.py`) を挟んで、上流への実要求を見た。設定は `proxy/r7_eval/litellm_config.yaml` である。
- **重さ**: venv は 573 MB、243 パッケージ (polars、boto3、azure、mcp、prisma 系などを含む)。
  - 1.106 では master key が必須になった (未設定だと起動を拒否する)。
- **ライセンス**: MIT。ただし `enterprise/` ディレクトリは別ライセンス (`LICENSE` 冒頭)。
  - background mode などは enterprise パッケージに入っている【文書: docs.litellm.ai/docs/response_api】。

#### 確認した挙動【実測】

1. **Responses の経路**
   - `hosted_vllm/...` をそのまま設定すると、`/v1/responses` が **ds41rt にそのまま転送されて 404** になる。
   - `use_chat_completions_api: true` を付けると Chat に変換される。この指定は `openai/` でも `hosted_vllm/` でも効く。
2. **Responses のストリームのイベント列**: 本文は reasoning、1 文、並列ツール 2 本 (`rtest.py r_order`)。

   ```
   seq=None response.created / None in_progress
   seq=None output_item.added (rs_ oi=0)            ← reasoning_summary_part.added が無い
   seq=4    reasoning_summary_text.done / 5 summary_part.done / 6 output_item.done
   seq=None output_item.added (msg oi=1) / content_part.added
   seq=None output_item.added (fc0 oi=2) / output_item.added (fc1 oi=3)   ← msg を閉じないまま次の項目を開く
   seq=None function_call_arguments.done (fc0) / 40 output_item.done (fc0) / ... 42 (fc1)
   seq=None output_text.done (msg) / content_part.done / seq=1 output_item.done (msg)   ← 最後に seq=1
   seq=None response.completed / data: [DONE]
   ```

   - 問題は 3 点ある。
     - `sequence_number` の欠落と逆行。
     - `reasoning_summary_part.added` の欠落。
     - 項目の入れ子 (OpenAI の実サーバーは項目を 1 つずつ開いて閉じる)。
   - 厳密な状態機械を持つクライアントや、`sequence_number` で並べ替えるクライアントは壊れる。
   - reasoning は `reasoning_summary_text` として出る (`streaming_iterator.py:718-777`)。
3. **非ストリームの output**
   - reasoning 項目の content の型が `"output_text"` になる (正しくは `reasoning_text`)。summary も無い。
   - function_call の前に、本文が空の message 項目が挟まる。
   - 要求していない `parallel_tool_calls: false` が応答に入る。
4. **reasoning の往復**
   - 入力の reasoning 項目の `content[].text` は、直後の tool_calls 付き assistant の `reasoning_content` に正しくまとめられる (上流への実要求で確認)。
   - content があると summary は使われない。
   - DeepSeek V4 向けのまとめ処理がある (`transformation.py:805-887`)。
5. **previous_response_id**
   - DB 無しでは **エラーにならず黙って無視**され、2 ターン目は履歴無しで生成された (「名前を知らない」と答えた)。
   - 履歴の復元は Postgres の spend_logs を prisma で引く方式である。`session_handler.py:297` に `if prisma_client is None: return []` とある。
6. `include: ["reasoning.encrypted_content"]` を付けても encrypted_content は返らない。
7. **Anthropic `/v1/messages`**
   - `openai/` + `use_chat_completions_api` では、Messages → Responses アダプタ → Chat という二重変換になる。応答の `content` は **常に `[]`** だった。思考の有無に関係なく 5 回中 5 回である。
   - `hosted_vllm/` では正しく変換され、`/v1/messages` も `/v1/responses` (bridge 付き) も 1 つの定義で通った。
   - thinking の signature は、非ストリームで `null`、ストリームで `""` だった。
   - `budget_tokens: 1024` は `reasoning_effort: "low"` に写される。
   - 並列ツールの content_block は、`openai/` 経路では block 2 の start が block 1 の stop より先に来た。`hosted_vllm/` 経路では逐次だった。
8. **性能**: mock 上流 (後述) に対し同時 16 本 × 4 回で測った。

| 経路 | TTFT p50 / p95 (ms) | 1 要求の時間 p50 / p95 (ms) | 差分間隔 p99 / 最大 (ms) |
|---|---|---|---|
| mock 直結 (基準) | 11.3 / 12.5 | 892 / 893 | 11.8 / 12.1 |
| LiteLLM 4 workers, chat | 43.3 / 203.6 | 1,024 / 1,217 | 35.4 / 237.7 |
| LiteLLM 4 workers, responses | 34.0 / 422.7 | 1,081 / 1,262 | 20.1 / 25.6 |
| LiteLLM 4 workers, messages | 57.1 / 374.3 | 1,159 / 1,439 | 37.7 / 269.1 |
| LiteLLM 1 worker, responses | — | 1,287 / 1,445 | 20.1 / 328.5 |
| LiteLLM 4 workers, responses, **同時 1 本** | 16.2 / 16.9 | 921 / 922 | 18.9 / 20.9 |

   - 公式の値 (p50 2〜12 ms、mock・Postgres あり) は、短い非ストリーム要求での測定である【文書: [docs.litellm.ai/docs/benchmarks](https://docs.litellm.ai/docs/benchmarks)】。
   - ストリームを同時に多数流すと、チャンク毎の処理 (pydantic 化、ロギングのコールバック) が効いて遅れが大きくなる【実測 + 推測】。

- **評価**
  - 実クライアントとの互換: 中。
    - §1.2 のクライアントは summary 形式でも読めるので、おそらく動く。
    - ただし seq 欠落と項目の入れ子は、OpenAI 公式 SDK の `responses.stream()` を使うクライアントで要確認。
  - イベント順序の正しさ: **不合格**。
  - previous_response_id: 黙って壊れる。
  - 性能: **不合格**。
  - 保守: リリースが頻繁で (1.106)、挙動が版ごとに変わる。
  - **不採用。** 参考実装として、DeepSeek 向けの reasoning のまとめ処理は読む価値がある。

### 3.2 (b) deepseek-ai/deepseek-recipe (MIT)

- **リポジトリ** ([github](https://github.com/deepseek-ai/deepseek-recipe))
  - 2026-09-10 公開で、commit は 2 つ (`57b9c842`、`8cadfede`)。
  - crates.io の `deepseek-recipe` / `-core` / `-encoding` は 0.1.0、PyPI の `deepseek-recipe` は 0.1.1 (manylinux wheel、OpenCV 同梱で 32 MB)。
  - ソースの写しは `research/tmp/r7/recipe_crates/` にある。
- **設計**
  - 3 種類の API (Chat / Responses / Messages) を 1 つの `Conversation` に変換し、V4/V4.1 のプロンプトに描画する。
  - 出力側は `StreamProcessor` がモデルの生テキスト (DSML) を `OutputChunk` に解析し、各 API の `ChunkGenerator` がイベントにする (`src/stream/mod.rs:42-74`)。
  - 推論・HTTP・ツール実行は呼び出し側が持つ。
- **未対応** (`README.md:34-43`): logprobs、JSON Schema / regex 制約と strict の強制、`n>1`、apply_patch 以外のカスタムツール、**previous_response_id による会話の保存**、**encrypted_content**。
- **Responses の入力** (`responses/request/schema.rs`)
  - 受理する型: message (役割は user / assistant / system / developer)、function_call、function_call_output、reasoning、custom_tool_call(_output)、namespace ツール、input_image (data / http URL)。
  - reasoning 項目は **`content[].text` だけを読む** (`:160-165`)。summary は読まず、encrypted_content は無視する。
  - `text.format: json_schema` は「受理して無視」する (`:227-228`)。tool の `strict` は素通しで、強制は呼び出し側が担う (`:264`)。
  - 未知のトップレベルのフィールドは無視する。`store`、`include`、`previous_response_id`、`prompt_cache_key` も同じ。
- **Responses の出力** (`responses/response/chunk_generator.rs:287-412`)
  - 項目は常に 1 つだけ開き (`close_item` → `open_item`)、`sequence_number` は単調に増える。
  - reasoning は `response.reasoning_text.delta/done`、本文は `output_text`、ツールは `function_call_arguments` / `custom_tool_call_input` で出す。
  - 終端は `completed` か `incomplete` (max_output_tokens)。
  - LiteLLM で見つかった順序の不具合は、構造上起きない【コード】。
- **Messages の出力**: thinking の signature の既定値は応答 ID になる (`anthropic/messages/response/chunk_generator.rs:34-38`)。空文字や null にはならない。
- **ds41rt の前でどう使えるか**: 3 通りある。
  1. **ds41rt に組み込む (b1)**
     - `native_v41.rs` は既に `ChatCompletionRequest::convert` → `render_conversation` → `StreamProcessor::new(chunk_generator)` の形をしている (`:254-295`)。
     - `ResponsesRequest` / `MessagesRequest` の route を足すのが最短で、余分な往復も無い。
     - ただし次の 3 点が壁になる。
       - 稼働中の ds41rt の再ビルドと再起動が要る。
       - ツール制約の検証器 (`tools::CompletionValidator`) が Chat のチャンク JSON を前提にしている。
       - 上流 (tpurtell/ds41rt、v15 で開発が活発) との差分を抱え続けることになる。
     - → 長期的には upstream への PR 候補にする。今回は採らない。
  2. **Rust プロキシで部品として使う (b2、推奨)**
     - 要求側の流れ: `ResponsesRequest` / `MessagesRequest` → `Conversation` → (自前の逆写像) Chat の messages → ds41rt。
     - `InputMessage` は System / User(+image_sources) / Assistant(reasoning_content, tool_calls) / Tool(+image_sources) / LatestReminder の 5 種類である (`deepseek-recipe-core/src/messages.rs`)。
     - ds41rt の Chat スキーマもこの 5 役割をそのまま受ける。画像はプレースホルダ `<｜image｜>` で content に埋め込まれる (`core/src/multimodal/mod.rs:15`) ので、そこで分割して image_url の部品に戻す。逆写像はほぼ無損失である【コード、要単体試験】。
     - 応答側は、ds41rt の Chat の差分を `OutputChunk` に写して `ChunkGenerator::generate` に渡す。
       - `reasoning_content` は `Reasoning` に、`content` は `Raw` に写す。
       - 新しい `tool_calls[i].id` は `ToolCall{name, ""}` に、引数の差分は `ToolArgumentsDelta` に写す。
       - `finish_reason` は `Finish` に写す。
     - `ChunkGenerator` と `OutputChunk` は pub である (`stream/mod.rs:42-74`)。ds41rt は tool_calls を逐次に流すので、recipe の「項目を 1 つずつ開く」形と合う【実測 §2】。
  3. **Python バインディング**
     - 要求の変換 (`ResponsesRequest(body).convert()`) は使える。
     - しかし `StreamProcessor` が受け取れるのは `InferenceChunk` の text / token だけで、`OutputChunk` は渡せない (`_native.pyi` の `InferenceChunk.ready/text/token/finish`)。
     - ds41rt が解析済みで返す差分を食わせるには、DSML の生テキストを組み立て直す細工が要る。→ Python では出力側に使いにくい。

- **評価**
  - 互換: Responses / Messages の意味論は DeepSeek 公式に揃う。
  - 順序: ◎
  - reasoning の往復: content による往復は◎。summary だけを返すクライアントには自前の補完が要る。
  - previous_response_id: 無い。
  - 画像: ◎
  - 保守: commit が 2 つの 0.1 なので API が変わる恐れはあるが、ds41rt と同じ版に固定すれば差が出ない。

### 3.3 (c) vLLM の Responses 実装 / OpenAI 系の互換シム

- **vLLM** (`ext/lil_src/vllm/entrypoints/openai/responses/`、Apache-2.0、5,518 行)
  - `serving.py:23-89` は `EngineClient`、`renderers`、`ParserManager`、`SamplingParams`、harmony を直接 import している。トークン列でエンジンを呼ぶ前提であり、HTTP の前段には移せない。
  - 保存 (`VLLM_ENABLE_RESPONSES_API_STORE`) はプロセス内の dict で、`serving.py:153-180` に「may cause a memory leak since we never remove responses」と書かれている。
  - 流用できるのは pydantic のスキーマ (`protocol.py`) とイベント名 (`streaming_events.py`) くらいである。
  - **移植は不採用。** イベント順序の検算の参照に留める。
- **HF responses.js** (MIT、Express)
  - Chat の上に Responses を作るシムで、ストリームと function calling は対応している。
  - しかし README は「Not implemented: … stateful API」としており、Reasoning も未チェックの項目である【文書: [README](https://raw.githubusercontent.com/huggingface/responses.js/main/README.md)】。
  - DeepSeek の思考の往復が無いため **不採用**。Node 依存も増える。
- **Open Responses** ([openresponses/openresponses](https://github.com/openresponses/openresponses)、Apache-2.0、2026-01 公開)
  - Responses を元にした中立仕様で、`src/lib/compliance-tests.ts` (1,784 行) にサーバー向けの適合試験がある【文書: [Simon Willison 2026-01-15](https://simonwillison.net/2026/Jan/15/open-responses/)】。
  - 実装候補ではなく、**試験の道具として使う** (写しは `research/tmp/r7/openresponses_compliance-tests.ts`)。
- **OpenAI 公式の汎用シム**: Chat から Responses への公式な変換シムは見つからなかった【文書調査の範囲】。

### 3.4 (d) 自作

- **Python (aiohttp)**: 性能を実測した (mock 上流、同時 16 本 × 4 回、`proxy/r7_eval/bench.py`)。
  - バイト素通し (`pass.py`、要求毎に ClientSession を作る素朴な実装): TTFT +1.8 ms、1 要求あたり +1 ms。差分間隔の p99 は 11.4 ms (基準 11.8 ms)。
  - Chat から Responses に変換する最小試作 (`rmini.py`、チャンク毎に `json.loads` / `dumps`、`sequence_number` を振る): TTFT +0.8 ms、1 要求あたり +1 ms。
    - 差分間隔の p99 は 19.8 ms (基準 11.8 ms) で、イベントループ内で 2 チャンクがまとめて送られる事例がある。
  - 1 本あたり約 100 イベント/秒 × 16 本 ≈ 1,600 イベント/秒は、1 プロセスで余裕がある。
  - 本番の decode は 1 本 30〜200 tok/s 程度なので、この負荷の範囲に収まる【推測】。
  - **「同時 16 本で追加 <5 ms」は Python でも満たせる。**
  - 欠点は 2 つある。
    - recipe の出力側を使えないので、Responses / Messages のイベント状態機械、apply_patch、namespace をすべて自分で書くことになる (見込み 1,500〜2,000 行)【推測】。
    - 型の誤りが実行時まで出ない。
- **Rust (axum + recipe)**
  - 性能は Python より余裕があり、単一バイナリで依存も軽い。
  - プロトコル部分の大半を recipe に任せられ、自作は Chat の正規化、逆写像、状態保存、ログで済む (見込み 1,200〜1,800 行)【推測】。
  - ホストには cargo が無い。recipe は edition 2024 なので、Rust 1.85 以上の docker イメージでビルドする (ds41rt と同じやり方)。

---

## 4. 評価表

凡例: ◎ 十分 / ○ 可 / △ 手当てが要る / × 不可。

| 評価軸 | (a) LiteLLM 1.106 | (b1) ds41rt に組み込む | (b2+d) Rust プロキシ + recipe | (d) Python 自作 | (c) vLLM 移植 / responses.js |
|---|---|---|---|---|---|
| OpenAI Responses クライアント (コーディングエージェント, Codex, SDK) | △ seq 欠落・項目の入れ子 | ◎ | ◎ recipe の生成器 | ○ 自作次第 | × / △ reasoning 無し |
| Anthropic SDK / Claude Code | △ provider 次第で本文が空、signature null | ◎ | ◎ signature = 応答 ID | ○ | × |
| OpenAI SDK (Chat) の正規化 | ○ drop_params・別名 | △ ds41rt 本体の改造 | ◎ 自作層 | ◎ | — |
| ストリームのイベント順序 | × | ◎ | ◎ | ○ 試験次第 | — |
| ツール呼び出しの往復 | ○ | ◎ | ◎ | ○ | ○ |
| reasoning の往復 (encrypted 無し) | ○ content→reasoning_content | ◎ | ◎ content。summary だけのクライアントは補完 | ○ | × |
| previous_response_id | × Postgres 必須・黙って無視 | △ 自作 | ○ LRU + sqlite を自作 | ○ 自作 | △ メモリリーク |
| 画像 | ○ | ◎ | ◎ | ○ | ○ |
| 同時 16 本で遅延 <5 ms | × +23〜46 ms (TTFT p50) | ◎ 0 | ◎ (<1 ms 見込み) | ◎ 実測 +1 ms | — |
| 稼働中の ds41rt に触れない | ◎ | × 再起動が要る | ◎ | ◎ | ◎ |
| 依存と保守 | × 573 MB・243 パッケージ・頻繁な版上げ | △ 上流との差分を抱える | ○ recipe=0.1.0 固定 | ○ aiohttp のみ | × |
| ライセンス | MIT (+enterprise 別) | MIT | MIT | — | Apache-2.0 / MIT |

---

## 5. 推奨構成と設計の要点

```
クライアント (コーディングエージェント / Anthropic SDK / OpenAI SDK, 同時 12〜16)
   │  :30000 (GLM 時代の口。今は空いている)
   ▼
ds41-gw (Rust 単一バイナリ, axum/tokio/reqwest, deepseek-recipe =0.1.0)
   │  常に stream=true で ds41rt を呼び、非ストリーム要求はプロキシで集約する
   ▼
ds41rt :8000 /v1/chat/completions (変更しない)
```

### 5.1 必須 (must)

1. **`/v1/responses` (stream と非 stream)**
   - recipe で `Conversation` に変換し、逆写像で Chat にして ds41rt に投げる。応答は `ChunkGenerator` で組み立てる。
   - 入力の reasoning 項目は、content → summary → 自前発行の encrypted_content の順に本文を探して補完する (recipe は content しか読まない)。
   - 出力の reasoning 項目は `content: [{type: reasoning_text}]` を基本とする。summary にも同じ本文を入れる設定を持つ (Codex 系の表示対策)【要確認】。
2. **イベント順序の保証と検査**
   - 次の 3 点を満たすことを、自動試験と本番の両方で検査する。
     - `sequence_number` が単調増加する。
     - 開いている項目は常に 1 つ。
     - added と done が対になる。
   - Open Responses の適合試験、OpenAI Python SDK の `responses.stream()` の最終組み立て、実クライアントの実機動作で回帰を見る。
3. **`/v1/messages` (Anthropic)**
   - thinking の signature は非空にする。ストリームの block は逐次にする。`stop_reason` は `tool_use` / `max_tokens` / `end_turn` に正しく写す。
   - Claude Code が叩く `/v1/messages/count_tokens` は、概算 (recipe の encode と tokenizer) で返す。
4. **`/v1/chat/completions` の正規化**: 素通しに近い形で、次の処理を入れる。
   - **model 名の別名**: `GLM-5.3-Flash` などを `deepseek-ai/DeepSeek-V4.1-Flash` にする。`/v1/models` も別名を返す。
   - **role の写像**: `developer` を `system` にする。
   - **思考指定の変換**: `chat_template_kwargs.{enable_thinking,thinking}` を `thinking: {type: enabled|disabled}` に、Responses の `reasoning.effort` と Anthropic の `budget_tokens` を ds41rt の `reasoning_effort` (none / low / high / max) に写す。
   - **temperature を省いたときの方針**: ds41rt は greedy になる。クライアント A は temperature を送らない。サンプリングにすると約 5% 遅い (`r2_ds41rt.md` §6)。既定値を注入するかどうかは決定事項である。
   - **黙って無視される項目**: `logprobs: true` などは、400 で弾くか素通しするかを設定で選べるようにする (ds41rt は黙って無視する)。
5. **reasoning 再注入ストア**
   - ds41rt の出力から tool_call_id ごとに reasoning_content を保存する (LRU、上限は件数と MB)。
   - 履歴の assistant(tool_calls) に reasoning が無ければ補う。
   - 対象はクライアント B (返さない 61/61) と、Responses で reasoning を落とすクライアントである。
   - 効果は prefix キャッシュの命中 (§2 の実測で、直前のターンの分) と、生成時の文脈の一致である。
6. **JSON Schema と strict の受け渡し**
   - recipe は無視するので、生の本文から取り出して ds41rt の `response_format.json_schema` と `tools[].function.strict` に入れる。
   - 取り出す元は、Responses の `text.format.json_schema` と `tools[].strict`、Anthropic の `output_format` 相当である。
7. **要求ログ**
   - `replay_openai.py` が読める形式 (`"request.received.openai"` 行 + Chat 変換後の obj) で JSONL を永続化する。元の API 種別と元の本文も残す。
   - GLM 時代の reqlog / loop_watch の監視の盲点 (Responses 要求が記録されない) を繰り返さない。
8. **過負荷と失敗の伝搬**
   - ds41rt の 429 (`Retry-After: 1`、`native_v41.rs:305-308`) と、ストリーム途中の失敗を区別して返す。
   - Responses では `response.failed` / `error` イベント、Anthropic では `error` イベントにする。正常終了には見せない。

### 5.2 あると良い (nice-to-have)

- **previous_response_id と store**
  - プロセス内の LRU と、任意の sqlite で保存する。`GET /v1/responses/{id}`、`DELETE`、`input_items` も用意する。
  - §1.2 のクライアントは使わない (store: false) が、OpenAI Agents SDK などに備える。
  - 見つからないときは **400 を明示**する (LiteLLM のように黙って無視しない)。
- **自前の encrypted_content**: reasoning 本文を自前の署名付きで base64 化して返し、受け取ったら復元する。encrypted_content しか往復しないクライアント向けである。
- **API キー認証**: 複数のキーと、キー毎の同時実行上限。
- **ツール呼び出し回数の上限**: GLM 時代の `GLM_MAX_TOOL_CALLS=8` 相当。
- **ループ遮断**: GLM_LOOP_BREAK 相当で、出力の反復を検知して中断する。
- **観測**: `/metrics` と `/v1/stats` で、追加遅延のヒストグラム、API 種別毎の件数、再注入の命中率を出す。
- **`prompt_cache_key` の記録**: §1.2 のクライアントがセッション ID を送るので、ログに残して再生とセッション解析に使う。

---

## 6. リスク

- **recipe 0.1.0 の若さ**: commit 2 つ、release 無し。
  - 互換を破る変更が入る恐れがあるが、ds41rt と同じ `=0.1.0` に固定すれば変換規則は揃う。
  - ds41rt が recipe を上げたらプロキシも追従する必要がある。
- **逆写像の無損失性**
  - recipe の正規化 (developer → system の合成、apply_patch ツールの説明文の注入、LatestReminder) を経た `Conversation` を Chat に戻し、ds41rt がもう一度変換したときに同じプロンプトになるかは未検証である【要確認】。
  - 検証法: `render_conversation` の結果を前段と後段で比べる単体試験。ds41rt と同じ recipe をプロキシ内で動かせば、ds41rt を叩かずに確かめられる。
- **summary だけを返すクライアント**: recipe は summary を読まないので、補完しないと思考の往復が切れる。
- **Python バインディングの制約**: 解析済みチャンクを渡せない。Python 版を選ぶとイベント生成をすべて自作することになり、順序の不具合を作り込みやすい。
- **ストリームの集約**: 非ストリーム要求を ds41rt の stream で集約する場合、ds41rt の非ストリーム固有の挙動 (usage の形、`metrics` フィールド) と差が出る。
- **ds41rt を止めずに試験する**: 試験は同時 4 本以内、max_tokens 小さめとする。本番クライアントの切替は口 30000 で行い、ds41rt 側は変更しない。
- **LiteLLM を使う場合 (参考)**: 上記の不具合は 1.106.0 時点のものである。版上げで直る可能性はあるが、同時 16 本での遅延 (+数十〜数百 ms) は構造的な問題である【推測】。

---

## 7. 再現手順

- **評価スクリプト** (`proxy/r7_eval/`)
  - `mock.py`: ds41rt に似た Chat の mock。口 8010。1 要求で reasoning 40、content 40、tool 1 本 (引数 8 片)、10 ms 間隔。
  - `bench.py`: 同時 N 本で TTFT、1 要求の時間、差分間隔を測る。
  - `pass.py`: バイト素通し。
  - `rmini.py`: Chat から Responses への最小変換試作。
  - `rtest.py`: Responses / Messages の挙動を調べる。`r_stream_tool`、`r_order`、`r_nostream`、`r_roundtrip`、`r_prev`、`a_stream_tool`、`a_roundtrip` の各ケースがある。
  - `tee.py`: 上流への要求を JSONL に記録する素通し。
  - `cache_rt.py`: reasoning の有無で prefix キャッシュの命中を比べる。ds41rt を直接、同時 1 本で叩く。
  - `litellm_config.yaml`: LiteLLM の設定。master key は含まない。
- **LiteLLM の起動**: `uv venv --python 3.12 v && VIRTUAL_ENV=v uv pip install "litellm[proxy]==1.106.0"` のあと、`LITELLM_MASTER_KEY=sk-... v/bin/litellm --config litellm_config.yaml --host 127.0.0.1 --port 4000 --num_workers 4` で起動する。
- **出典ソースの写し** (`research/tmp/r7/`)
  - `recipe_crates/` (deepseek-recipe / -core / -encoding 0.1.0)
  - `deepseek_recipe_py_0.1.1_native.pyi`
  - §1.2 のクライアントの Responses、completions、Anthropic の各実装
  - `litellm_1.106.0/` (session_handler、streaming_iterator、transformation、reasoning_items、anthropic の architecture.md)
  - `openresponses_compliance-tests.ts`
- 試験後に、LiteLLM と mock 系のプロセス (口 4000 / 8009〜8012) はすべて停止した。ds41rt (口 8000) には触れていない。

## 8. 出典 URL

- deepseek-recipe: <https://github.com/deepseek-ai/deepseek-recipe>、crates.io `deepseek-recipe` 0.1.0 (MIT)、PyPI `deepseek-recipe` 0.1.1
- LiteLLM: <https://docs.litellm.ai/docs/response_api>、<https://docs.litellm.ai/docs/anthropic_unified>、<https://docs.litellm.ai/docs/benchmarks>、PyPI `litellm` 1.106.0
- HF responses.js: <https://github.com/huggingface/responses.js>
- Open Responses: <https://github.com/openresponses/openresponses>、<https://simonwillison.net/2026/Jan/15/open-responses/>
- Codex CLI の chat wire 廃止: <https://opper.ai/blog/codex-router>、<https://github.com/janhq/jan/issues/7413>
- 競合による LiteLLM の測定 (参考): <https://agentgateway.dev/blog/2026-06-26-benchmarking-agentgateway-vs-litellm/>
