# r7: 実クライアントが使っている API の洗い出し (GLM-5.3-Flash 時代の要求ログから)

作成: 2026-10-19。対象: GLM-5.3-Flash を `:30000` で出していた約 10 日間 (SGLang 期とその後の vLLM 期) の要求。
中身はコーディングエージェント型の実トラフィックで、複数のクライアント (OpenAI 互換のエージェント、Responses API を使うエージェント、補助の短い要求、手動の試験) が混ざっている。
目的: ds41rt (DeepSeek-V4.1-Flash, `localhost:8000`) の前段に置く proxy (`proxy/`) に何が要るかを、実測の件数で決める。
公開版では、クライアントの名前、system プロンプトの文面、ツールの一覧、利用した日時の細かい記述を一般化した。件数と比率はそのまま残した。

---

## 0. 結論 (先に)

| 優先度 | 機能 | 根拠 (件数) | ds41rt の現状 |
|---|---|---|---|
| **P0** | **model 名の別名受理** (`GLM-5.3-Flash` ほか任意名を ds41rt の名に置換) | chat 31,913 件の 99.6% が `GLM-5.3-Flash`、残りは `<model>` 98 件と `x` 20 件 | 400 `model must be deepseek-ai/DeepSeek-V4.1-Flash` (実測、`native_v41.rs:268-270`) |
| **P0** | **temperature / top_p 省略時の既定値注入** | chat の 27.1% (8,664 件) と Responses の全件が temperature を送らない (複数のクライアント) | 省略すると **greedy** (`native_v41.rs:172` で `unwrap_or(0.0)`) |
| **P0** | **`/v1/responses` → chat 変換** (SSE、function tools、全履歴の再送、並列関数呼び出し、reasoning 項目の往復) | サーバーログで 2,910 件 (chat の 8.8%)。Responses しか使わないクライアントがある | エンドポイントが無い (`r2_ds41rt.md:394-395`) |
| **P0** | **tool メッセージの非 text パーツ (`tool_reference`) をテキストに変換** | ツールを遅延ロードするクライアントの要求 346 件 (そのクライアントの 5.7%)。1 回ロードするとそのセッションは以後の全ターンで含む | 400 `data did not match any variant of untagged enum ChatCompletionRequestContent` (実要求の再送で 10/10 件再現) |
| **P0** | **同時 12〜16 本の非ストリーム + tools 要求を落とさない** (ds41rt の 429 を proxy 側の待ち行列で吸収) | 最多のクライアント (自動コーディング) は 21,063 件すべて非ストリーム、tools 付き 98.8% | 並列 16 が上限、溢れると 429 + `Retry-After: 1` (`native_v41.rs:300-308`) |
| P1 | `chat_template_kwargs` (`enable_thinking` / `thinking` / `clear_thinking` / `reasoning_effort`) を ds41rt の `thinking` / `reasoning_effort` に翻訳 | 30,097 件 (94%) が送る。ほぼ全部が `enable_thinking: true` で既定と同じ。`false` は 44 件 (大半が試験) | 未知フィールドとして無視 (実測: `enable_thinking:false` でも思考が出る) |
| P1 | `developer` ロール → `system` | chat では 0 件。Responses では全件 | 400 `unknown variant developer` (実測) |
| P1 | 要求ログ (生 JSON、User-Agent、経路、所要時間) | 今回、Responses の生 JSON と User-Agent がどこにも残っていなかった | — |
| P1 | Responses の `function_call_output` に画像が来た場合の扱い (画像を読むツール) | Responses のクライアントで画像を読むツールの呼び出しが 44 回 (履歴の延べ数) | tool ロールに画像は不可 (`convert.rs:289-296` の `into_text`) |
| P2 | `reasoning_effort` の値の正規化 (`medium`/`minimal`/`xhigh` の扱い) | `medium` 419 件、`minimal` 2 件 | native 経路は受理済み (実測、`convert.rs:218-226`)。旧 OpenAI 経路 (`request.rs:228-240`) は拒否 |
| 不要 | `/v1/completions`、`/v1/messages`、embeddings、logprobs、`n`>1、penalty、`response_format`、`seed`、`stop`、`user`、`previous_response_id`、http(s) 画像 URL、API キー認証 | いずれも実クライアントの使用 0 件 (§2、§3) | — |

要点は三つある。

1. ds41rt に今のクライアントをそのまま向けると、**model 名で全件 400** になる。名前を直しても、**temperature を送らない 27% が greedy になり**、ツールを遅延ロードするクライアントの一部が `tool_reference` で 400 になる。
2. `/v1/responses` は件数こそ chat の 1 割弱だが、Responses 専用のクライアントがあるので、無いと使えない。中身は「状態を持たない全履歴再送 + function tools + reasoning の往復」で、chat への変換で実装できる。
3. 逆に、logprobs・n・penalty・response_format・seed・stop・`/v1/completions` などは実クライアントが一度も使っていない。proxy で作る必要はない。

---

## 1. データ源と、読めること・読めないこと

| 源 | 中身 | 注意 |
|---|---|---|
| SGLang の要求ログ (173 ファイル、19GB) | SGLang `--log-requests-level 3` の JSON。chat は `request.received.openai` に要求本体、Responses は `request.received` (rid が `resp_`) に描画済みプロンプトだけ | 下記 (a)〜(c) |
| SGLang のサーバーログ (38 本) | uvicorn のアクセスログ `"POST /v1/... HTTP/1.1" 200` | User-Agent は出ない |
| vLLM のサーバーログ (28 本) | 同上 (vLLM) | 実験 E0〜E8 の再生 (`replay_openai.py`) が主で、実クライアントと区別できない |
| 障害記録 (24 件) | 内部形式 (GenerateReqInput) の要求とサーバーログ | OpenAI 形式の要求本体は無い |

読めないこと (解析の限界):

- (a) **既定値と明示値を区別できない。** SGLang は要求を pydantic で検証したあと、`None` 以外のフィールドを既定値ごと書き出す。`n: 1`、`logprobs: false`、`frequency_penalty: 0.0`、`parallel_tool_calls: true` は全件に出る (SGLang `protocol.py:858-883`)。`tool_choice` は、未指定なら tools の有無で `auto` / `none` が入る (`protocol.py:994-1004`)。したがって tool_choice の 29,298 件 `auto` は「tools 付き件数」と同じ意味しか持たない。
- (b) **未定義のフィールドは消えている。** SGLang の `ChatCompletionRequest` は `extra` 設定が無い (pydantic 既定の ignore) ため、`metadata`・`store`・トップレベル `thinking` などを送っていても記録に残らない。`user`・`seed`・`stop`・`response_format`・`logit_bias`・`top_logprobs` は定義済み (既定 None) なので、出現 0 件は「非 null では一度も送られていない」と読める。
- (c) **Responses の生 JSON は残っていない。** SGLang は Responses の要求を `request.received.openai` で書き出さない。残っているのは chat テンプレートで描画した後のプロンプト文字列とサンプリング設定だけである。§4 の観測は、ここからの**推定復元**である。
- (d) **User-Agent はどこにも無い。** クライアントの判別は system プロンプトとツール名で行った (§3)。接続元 IP でも判別できない。

---

## 2. エンドポイント別の件数

### 2.1 SGLang 期 (アクセスログ)

| エンドポイント | 件数 | 呼び出し元 |
|---|---:|---|
| `POST /v1/chat/completions` | 33,105 (200) + 7 (500) + 1 (400) | 実クライアント + 試験 |
| `POST /v1/responses` | **2,910** (すべて 200) | 2,812 件は要求ログの開始前でクライアント不明。残る 98 件は Responses 専用のクライアント 1 つ |
| `GET /v1/models` | 3,141 | クライアントの起動・ヘルス確認 |
| `GET /get_model_info` | 3,000 | 約 30 秒ごとに外部から来るポーリング (SGLang 固有、非推奨警告あり)。vLLM 期には 404 を 2,009 回返したが支障は出ていない |
| `POST /generate` | 860 | 内部 (診断、再生試験) |
| `POST /flush_cache` | 2,770 | 内部 (監視) |
| `GET /health`、`/metrics`、`/get_server_info` 他 | 約 140 | 監視 |
| `POST /v1/embeddings` | 3 (400) + 2 (GET 405) | 試験 |
| `/v1/completions`、`/v1/messages` | **0** | — |

### 2.2 vLLM 期

`POST /v1/chat/completions` が 15,879 件で、`/v1/responses` は 0 件だった。`GET /v1/models` は 2,151 件、`GET /get_model_info` の 404 は 2,009 件ある。ただしこの期間は `replay_openai.py` による実験 E0〜E8 の再生が主で、実クライアントの数字としては使えない。replay は chat しか送らない。

### 2.3 参考: クライアント側の中継

一部のクライアントは、API キーで認証して上流に中継する中継を通していた。通すのは `models`、`chat/completions`、`completions` だけで、`responses` は通らない。上流には `Authorization` を渡さない。
この中継を経由するクライアントには、ds41rt 側の API キー認証は要らない。ただし、どのクライアントが中継を経由しているかはログからは分からない。

---

## 3. `/v1/chat/completions` の中身 (要求ログ `request.received.openai` 31,913 件)

### 3.1 クライアントの型別 (system プロンプトとツール名で判別)

クライアントの名前は伏せ、型で分けた。

| クライアントの型 | 件数 | stream | tools | 主な設定 |
|---|---:|---|---|---|
| **A: 自動コーディング** (ツール 6 個 bash/edit/glob/grep/read/write。方針立案・比較の tools なし要求を含む) | 21,063 (66%) | **0%** | 98.8% | `temperature 0.6`、`top_p 0.95`、`top_k 20`、`max_tokens 32000`、`chat_template_kwargs {"enable_thinking": true}`。tools なしの要求は `temperature 0.7 / 0.3`、`max_tokens 12000`。履歴に reasoning は返さない |
| **B: 対話型のコーディングエージェント** (ツール 23〜37 個。ツールの遅延ロードあり) | 6,003 | 99.8% | 99.6% | `max_tokens 131072`、`reasoning_effort high 5,353 / medium 417 / low 178 / max 8`、`chat_template_kwargs {"thinking": true, "enable_thinking": true}`、temperature なし。履歴の assistant に `reasoning_content` あり 77.5%。user content は配列 (`type:text`) |
| B の補助要求 (サブエージェントなど) | 675 | 100% | 95% | `reasoning_effort low 437 / high 204`、`max_tokens 131072 / 6000 / 300` |
| **C: 対話型のコーディングエージェント (新版)** (ツール 12 個) | 1,071 | 100% | 100% | `max_completion_tokens 131072 / 16384`、model `GLM-5.3-Flash` 979 + `<model>` 92、画像 14.1% (最大 7 枚/要求)、`reasoning_content` を返す 97% |
| C の旧版 (ツール 11 個) + タイトル生成 | 153 + 36 | 100% | 100% / 0% | `max_tokens 32000`、タイトル生成は model `<model>` 6 件 |
| **D: CLI のコーディングエージェント** (ツール 27〜48 個) | 578 | 100% | 100% | max_tokens・temperature とも送らない。履歴が長い (メッセージ数 中央値 554、最大 1,131、本文 最大 1.25MB) |
| E: 短い要約の補助要求 (日本語の system プロンプト) | 2,070 | 0% | 0% | `temperature 0.2`、`max_tokens 120`、`reasoning_effort low`、`chat_template_kwargs {"clear_thinking": true, "thinking": true, "enable_thinking": true}` |
| 手動試験・ベンチ (system なし、1 メッセージ) | 192 | 47% | 0% | 画像 64 枚までの試験、`model "x"` 14 件など |
| その他 | 72 | — | — | — |

ツールの型はすべて `function` で、`strict: true` は 0 件だった。ツールは最大 58 個である。

### 3.2 フィールドの出現頻度 (31,913 件中)

| フィールド | 出現 (非 null) | 値の内訳 | 備考 |
|---|---:|---|---|
| `stream` | 全件 | true 8,651 (27.1%) / false 23,262 | 非ストリームの 9 割は型 A |
| `stream_options` | 8,626 | すべて `{"include_usage": true}` (SGLang が `continuous_usage_stats:false` を補う) | ストリーム要求のほぼ全部 |
| `tools` | 29,298 (91.8%) | 1〜58 個、type は function のみ | |
| `tool_choice` | (既定で全件) | auto 29,298 / none 2,615 | §1(a) のとおり、明示かどうかは不明 |
| `parallel_tool_calls` | (既定で全件) | true | 同上 |
| `response_format` | **0** | — | |
| `reasoning_effort` | 8,961 | high 5,758 / low 2,771 / medium 419 / max 10 / minimal 2 / none 1 | |
| `chat_template_kwargs` | 30,097 | `{"enable_thinking":true}` 21,063 / `{"thinking":true,"enable_thinking":true}` 6,876 / `{"clear_thinking":true,…}` 2,070 / `{"enable_thinking":false}` 44 / `{"reasoning_effort":"low"}` 28 / ほか | ds41rt は無視する |
| `max_tokens` | 30,158 | 32000 20,974 / 131072 6,571 / 120 2,070 / 12000 248 / その他 | ds41rt の出力上限は 393,216 なので 131072 は通る (`/v1/models` の `max_output_tokens`) |
| `max_completion_tokens` | 1,101 | 131072 1,009 / 16384 92 (型 C の新版だけ) | |
| `temperature` | 23,249 | 0.6 20,821 / 0.2 2,070 / 0.7 182 / 0.0 69 / 0.3 66 / 1.0 40 | **省略 8,664 件** |
| `top_p` | 20,855 | 0.95 20,854 | |
| `top_k` (SGLang 拡張) | 20,815 | 20 (型 A のみ) | ds41rt は受理する (実測) |
| `stop` / `seed` / `user` / `logit_bias` / `top_logprobs` | **0** | — | |
| `n` / `logprobs` / `presence_penalty` / `frequency_penalty` | (既定で全件) | 1 / false / 0.0 / 0.0 | 既定以外の値は 0 件。ds41rt は既定値の明示を受理する (実測) |
| `ignore_eos` | 40 (true) | 試験だけ | |
| `metadata` / `store` | 不明 | — | §1(b) |
| `model` | 全件 | `GLM-5.3-Flash` 31,795 / `<model>` 98 / `x` 20 / `default` 1 | |

### 3.3 メッセージの形

| 項目 | 件数 | 備考 |
|---|---:|---|
| ロール | system 29,395 / user 56,713 / assistant 549,382 / tool 586,725 (延べ) | **developer は 0** |
| user content が配列 (`type:text` / `image_url`) | 6,807 要求 | 型 B はすべて配列 |
| 画像 (`image_url`) | 259 要求、延べ 1,180 枚 | **すべて `data:` URL**、`detail: "auto"`。実クライアントの最大は 7 枚/要求。8 枚以上は試験だけ (16 / 32 / 64 枚) |
| assistant の `reasoning_content` (履歴で送り返す) | 6,678 要求 (延べ 108,818 メッセージ) | 型 B 77.5%、型 C 97%、型 D 92%。ds41rt は受理する (`schema.rs:59-63`)。`reasoning` という名前は 0 件 |
| assistant の `tool_calls` | 24,429 要求 (延べ 536,364) | `{"id","type":"function","function":{"name","arguments"}}` |
| tool の content | 文字列 586,369 / **配列 `[{"type":"tool_reference","name":…}]` 439** | 後者はツールの遅延ロードの結果 (346 要求) |
| `name` (メッセージ) | 0 | |

`tool_reference` の形 (id、ツール名、思考の本文は置き換えた):

```json
{"role": "assistant", "reasoning_content": "…",
 "tool_calls": [{"id": "call_0001", "type": "function", "function": {"name": "load_tools", "arguments": "{\"names\":[\"some_tool\"]}"}}]}
{"role": "tool", "tool_call_id": "call_0001", "content": [{"type": "tool_reference", "name": "some_tool"}]}
```

ds41rt の native 経路は、content を `String | [Text | ImageUrl | File]` の untagged enum で受ける (`deepseek-recipe-0.1.0/src/protocol/openai/chat_completion/request/schema.rs:87-118`)。このため `tool_reference` は 400 になる。

### 3.4 実要求を ds41rt に送った互換性試験 (2026-10-19 実施)

要求ログの実要求を、`model` だけ `deepseek-ai/DeepSeek-V4.1-Flash` に替えて送った。`max_tokens=1`、非ストリーム、逐次。クライアントの型ごとに最大 6 件を無作為に選び、`tool_reference` を含む要求以外は 80KB 未満に限った。

| クライアントの型 | 結果 |
|---|---|
| A、B (tool_reference なし)、B の補助、C 新旧、タイトル生成、D、E、試験、その他 | すべて 200 (計 60 件) |
| B + `tool_reference` | **400 が 6/6 件** |
| その他 + `tool_reference` | **400 が 4/4 件** |

model 名を替えない場合は、全件が 400 になる (`{"model":"GLM-5.3-Flash",…}` で実測)。
追加の単体確認では、`reasoning_effort: "medium"` と `"minimal"` は 200、トップレベルの `thinking: true` (bool) は 400、`n: 2` は 400、`logprobs: true` は 200 (無視される)、`metadata` と `store` は 200 (無視される)、`role: developer` は 400 だった。

---

## 4. `/v1/responses` の中身

### 4.1 判明していること

- **クライアント**: 要求ログに残る 98 件は、すべて Responses 専用のコーディングエージェント 1 つ (型 F) からだった (developer 指示の書き出しで判別)。
  - 残る 2,812 件は要求ログの開始前なので、中身は不明である。
- **ストリーム**: 98/98 件が `stream: true` だった。
- **ツール**: 主要求 97 件はすべて 27 個の function tools (シェル、ファイルの読み書きと編集、検索、画像の読み込み、サブエージェント、Web 取得など) を持つ。描画結果には `description / name / parameters` だけが残る。`strict` の有無は分からない。
- **max_output_tokens**: 描画後の `max_new_tokens` は 127,998 (97 件) と 62 (1 件) だった。SGLang は `min(max_output_tokens, 既定) − 2` を使う (`sglang/srt/entrypoints/openai/protocol.py:1996-2003`)。プロンプト長が違っても値が一定なので、**`max_output_tokens` は 128000 と 64 (タイトル生成) と推定**する。
- **temperature / top_p**: 描画後は全件 1.0 / 0.95 だった。サーバー既定と一致するので、送っていないと推定する。
- **reasoning**: 主要求は `Reasoning Effort: Max`、タイトル生成は `High` で描画されていた。launcher の既定 (`--default-chat-template-kwargs {"reasoning_effort":"high"}`) が Responses 経路で効いていないのか、クライアントが `reasoning.effort` を送ったのかは区別できない。
- **状態**: `--enable-response-store` は起動引数に無い。SGLang はこの場合、`previous_response_id` や `background` を付けた要求をエラーにする (`serving_responses.py:294-298`)。全 2,910 件が 200 なので、**`previous_response_id` は使っていない (状態なしで毎ターン全履歴を再送)** と判断する。実際、入力項目数は 1 会話の中で 4 から 200 まで単調に増えていた。
- **入力項目の型** (描画から復元、98 件の延べ数):

| 型 | 延べ数 | 要求あたり |
|---|---:|---|
| `message` role=developer | 98 | 各 1 (先頭) |
| `message` role=user (`input_text`) | 389 | 3〜8。先頭 3 件は本文、実行時の文脈、skills 一覧 |
| `message` role=assistant (`output_text`) | 420 | |
| `reasoning` (前ターンの思考を送り返す) | 1,683 | 94 件にあり |
| `function_call` | 3,362 | 94 件にあり。並列 (1 ターンに 2〜4 個) が多い |
| `function_call_output` | 3,362 | 例 `…[status: completed, exit code: 0]` |
| `input_image` | 0 | 画像を読むツールを 44 回呼んでいるが、画像は SGLang に届いていない (`image_data` 0 件) |

呼ばれたツールの延べ数: シェル 2,021、ファイルの書き込み 558、読み込み 369、編集 133、そのほか (TODO、ジョブ出力、依存の読み込み、画像の読み込み、利用者への質問) が各 38〜78。

- **reasoning の往復**: SGLang は思考を `reasoning.encrypted_content` に `sglang-reasoning-v1.` + base64(zlib) で詰め、送り返されたらこれを解く (`responses_adapters.py:194-215`)。履歴に思考が描画されているので、クライアントは reasoning 項目 (encrypted_content か content/summary のどちらか) を毎ターン返している。proxy は、この往復を自前の形式で再現する必要がある。

### 4.2 推定復元した実例 (公開版では省略)

元の文書には、描画済みプロンプトから推定復元した Responses 要求の実例 4 件 (会話の最初のターン、タイトル生成、並列関数呼び出しを含む途中のターン、200 項目の長い会話) を載せていた。
私的な要求ログの中身 (開発者指示、利用者の発話、作業ディレクトリ、ツールの入出力) を含むので、公開版では削った。形は次のとおりである。

- 先頭に `developer` の message が 1 つ (約 2 万字の指示)、続いて `user` の message が 3〜8 個 (本文、実行時の文脈、skills の一覧)。
- 途中のターンでは `reasoning` (前ターンの思考)、`function_call` (1 ターンに 2〜4 個の並列が多い)、`function_call_output`、`assistant` の `output_text` が繰り返す。
- 毎ターン全履歴を送るので、長い会話では 1 要求が 260KB (入力 200 項目) に達する。

### 4.3 proxy の Responses 実装に要るもの (上の観測から)

1. `input` の項目変換: `message` (developer→system、user/assistant、`input_text`/`output_text`)、`function_call`→assistant の `tool_calls`、`function_call_output`→`tool` メッセージ、`reasoning`→次の assistant の `reasoning_content`。連続する function_call は 1 つの assistant メッセージにまとめる (並列呼び出し)。
2. `tools` の平坦形 `{type,name,description,parameters,strict}` → chat の `{type:"function",function:{…}}`。
3. `max_output_tokens` → `max_tokens`。`reasoning.effort` → `reasoning_effort`。temperature を省略されたら既定値を注入する (P0 の 2 番と同じ)。
4. SSE を Responses のイベント列 (`response.created`、`response.output_item.added`、`response.reasoning_text.delta` / `response.output_text.delta`、`response.function_call_arguments.delta`、`response.output_item.done`、`response.completed`) に組み直す。非ストリームは使われていないが、実装は容易である。
5. reasoning 項目に `encrypted_content` を載せて返し、送り返されたら復号して `reasoning_content` に戻す (状態なしで往復させる)。
6. `previous_response_id`・`store`・`background` は実例が 0 件なので後回しにしてよい。来たら 400 を明示する。

---

## 5. ストリーム・画像・同時実行のまとめ

- **ストリームの割合**: chat 全体で 27.1%。型 A を除くと 98% がストリームである (型 B〜D と B の補助はほぼ 100%)。Responses は 100% ストリーム。
- **非ストリームの主力**: 型 A は約 2 日で 21,063 件 (約 20 件/分)、同時 12〜13 本だった。クライアント並列は 16 に決めている。ds41rt の並列上限 16 と同じなので、溢れた分の 429 は proxy で待たせて吸収したい。
- **画像**: data URL だけで、http(s) は 0 件。実クライアントの最大は 7 枚/要求で、ds41rt の上限 16 枚に収まる。型 C の新版の 14% が画像付き。Responses の `input_image` は 0 件。
- **思考**: ほぼ全件が思考ありを要求している (`enable_thinking: true` か `reasoning_effort`)。ds41rt の既定 high と合う。思考を切る要求は試験を除くと数件しかない。

---

## 6. 残る不確実性

1. 要求ログの開始前の Responses 2,812 件のクライアントと中身は不明。型 F と同じかどうかは確認できない。
2. Responses の `include`・`store`・`text.format`・`truncation`・`tool_choice`・`strict` は不明 (§1(c))。proxy では、受理して無視するか、意味のある値だけ解釈する作りにしておく。
3. `metadata`・`store`・トップレベル `thinking` など、SGLang が捨てたフィールドの使用状況は不明 (§1(b))。ds41rt は未知フィールドを無視するが、トップレベル `thinking: true` (bool) は 400 になる (実測)。来る可能性に備え、proxy で `{"type":"enabled"}` に正規化しておくと安全である。
4. User-Agent が無いので、クライアントの版は分からない。proxy で記録を始める。
5. どのクライアントが ds41rt に向く予定かは、このログからは分からない。件数の 66% は型 A 1 本の約 2 日分なので、比率はその影響を強く受けている。
6. 履歴の `reasoning_content` を DeepSeek-V4.1 のテンプレートがどう扱うか (過去ターンの思考を捨てるか残すか) は、本調査の範囲外である。
