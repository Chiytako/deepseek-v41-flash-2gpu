# ds41gw: ds41rt 前段の API 変換プロキシ

ds41gw は ds41rt (DeepSeek-V4.1-Flash-1017 EXL3 2bit、scale 校正版、`localhost:8000`) の前に置くプロキシである。
GLM-5.3-Flash 時代に使っていた OpenAI / Anthropic 互換のクライアント (コーディングエージェントなど) を、設定を変えずに使えるようにする。
仕様は [`docs/research/r7_proxy_spec.md`](../docs/research/r7_proxy_spec.md) (以下「仕様」) にある。ds41rt 本体には手を入れていない。

- 実装は Rust (axum 0.7、hyper 1、tokio、reqwest)。ds41rt と同じ `deepseek-recipe` / `-core` / `-encoding` `=0.1.0` を部品に使う。
  中身は ds41rt 改修版 (v15-effort) と同じ数値 effort 対応版で、`vendor/deepseek-recipe-patched/` に写し、`ds41gw/Cargo.toml` の
  `[patch.crates-io]` で差し替える (`Cargo.lock` ではこの 3 つが path 依存になる)。`cargo.sh` は `vendor/` を `/vendor` に、
  `docker/Dockerfile` は `/vendor` に置くので、どちらも `--locked` で通る。
- 上流は ds41rt の `/v1/chat/completions` だけを使う。常に `stream:true` で呼び、非ストリームの要求はプロキシで集約する。
- 待ち受けは既定で `127.0.0.1:30001` だけ。LAN や Tailscale に出すときは `DS41GW_LISTEN_IPS` にその IP を足す (例 `DS41GW_LISTEN_IPS="127.0.0.1 100.x.y.z"`)。docker で `--network host --restart unless-stopped` として動かす。
- `:30000` (GLM 時代の口) にはまだ出していない。切り替えの手順は[後述](#30000-への切り替え)。

## 起動・停止・ログ

```sh
cd proxy
./run_proxy.sh            # イメージをビルドして (再) 起動する
./run_proxy.sh build      # イメージだけ作る (動いているコンテナには影響しない)
./run_proxy.sh start      # 既存イメージで (再) 起動する
./run_proxy.sh stop       # 止めて消す
./run_proxy.sh logs       # sudo docker logs -f ds41gw (起動・bind・エラー)
./run_proxy.sh test       # cargo test --release (rust:1.90-bookworm コンテナで実行。ds41rt は使わない)
```

| 環境変数 | 既定 | 内容 |
|---|---|---|
| `DS41GW_PORT` | `30001` | 待ち受けの口。空白区切りで複数を指定できる (例 `"30001 30000"`)。ヘルスチェックには先頭の口を使う |
| `DS41GW_LISTEN_IPS` | `127.0.0.1` | 待ち受ける IP。口と IP のすべての組で待ち受ける |
| `DS41GW_UPSTREAM` | `http://localhost:8000` | 上流 (ds41rt) |
| `DS41GW_CONFIG` | `config/ds41gw.toml` | 設定ファイル |
| `DS41GW_TOKENIZER` | (必須) ds41rt が読んでいる snapshot の `tokenizer.json` (DeepSeek-V4.1-Flash の公式のものと同じ) | reasoning_tokens と count_tokens の計数に使う |

- `run_proxy.sh start` は、設定の `listen` と `upstream.url` を環境変数で書き換えた写し (`state/ds41gw.rendered.toml`、0600) を作り、それをコンテナに渡す。
  このため、設定ファイルの `listen` 行は `run_proxy.sh` で起動する限り効かない。口は `DS41GW_PORT` と `DS41GW_LISTEN_IPS` で変える。
- `start` は既存のコンテナを `docker rm -f` で消してから起動し直す。**実行中の要求は切れる**ので、先に `curl -s localhost:30001/v1/proxy/stats` で
  `upstream_inflight` と `queue_waiting` が 0 であることを確かめる。
- コンテナは uid 1000 で動く。書き込むのは `state/` (sqlite と HMAC 鍵) と `logs/` (要求ログ) で、どちらも 0700 にしてある。
- bind に失敗した口 (tailscaled が未起動のときなど) は、5 秒ごとに bind をやり直す。ほかの口は先に動き出す。
- SIGTERM を受けると、新しい要求を 503 にし、実行中のストリームの終了を最大 30 秒待つ (`server.shutdown_grace_s`)。
- ホストに cargo が無いので、ビルドは `cargo.sh` (rust:1.90-bookworm コンテナ) か `docker/Dockerfile` (多段ビルド) で行う。
  recipe は edition 2024 と let-chains を使うので、Rust 1.88 以上が要る (仕様 §15.2 の 1.86 ではビルドできない)。
- 状態の確認は `curl -s localhost:30001/health/ready` (ds41rt の `/health` と待ち行列) と `curl -s localhost:30001/v1/proxy/stats` で行う。

### 要求ログ

- 置き場は `logs/req/YYYYMMDD.jsonl` (UTC) で、ファイルは 0600。前日分は zstd で圧縮し、14 日または 50 GiB を超えたら古い日から消す。
- 受信の行は `"event":"request.received.openai"` で、`obj` (ds41rt に送った Chat 本文) を持つ。GLM 時代の再生器 `replay_openai.py` (非公開) や
  `tests/replay_capped.py` でそのまま再生できる。ほかに `orig` (受け取った本文)、`user_agent`、`client_ip`、`key` (名前だけ)、`normalized`、`warnings` を残す。
- 終了の行は `"event":"request.finished"` で、`status`、`queue_ms`、`ttft_ms`、`overhead_ms`、`dur_ms`、`upstream_retries`、`usage`、`reasoning_tokens`、
  `tool_calls`、`reinjected`、`client_disconnected` などを持つ。
- 画像の data URL と Anthropic の base64 画像 (`source.data`) は、既定でハッシュに置き換える (`log.images="hash"`)。画像つきの要求を再生したいときは `"full"` にする。
- 画像を伏せる処理と直列化は、要求ログの書き込みスレッドで行う。上流へ送る前の待ち時間には入らない。
- API キーやヘッダの値はログに残さない。

```sh
tail -f logs/req/$(date -u +%Y%m%d).jsonl | jq -c 'select(.event=="request.finished") | {api,status,ttft_ms,overhead_ms,dur_ms,finish_reason}'
```

## エンドポイント

| パス | 内容 |
|---|---|
| `POST /v1/chat/completions` | 正規化して ds41rt に渡す (仕様 §5)。ストリームと非ストリームの両方。n は 8 まで |
| `POST /v1/responses` | recipe で `Conversation` にし、Chat に戻して送る。出力は recipe の生成器にプロキシの後処理を足したもの (仕様 §6) |
| `GET /v1/responses/{id}`、`DELETE /v1/responses/{id}`、`GET /v1/responses/{id}/input_items` | 状態を保持する (sqlite、30 日)。input_items は OpenAI の項目の形で返し、自動ページ送りに対応する |
| `POST /v1/responses/{id}/cancel` | background が無いので、常に 400 を返す |
| `POST /v1/messages` | Anthropic Messages (仕様 §7)。ストリームと非ストリームの両方 |
| `POST /v1/messages/count_tokens` | ds41rt を呼ばずに tokenizer で数える (実際の `input_tokens` と一致することを確認済み) |
| `POST /v1/completions` | Chat に包んだ擬似実装。思考は off、`max_tokens` の既定は 16。`x-ds41gw-warning: completions_emulated_via_chat` を付ける |
| `GET /v1/models`、`GET /v1/models/{id}` | 正規名と別名を返す。OpenAI と Anthropic の両 SDK が読める形にしてある |
| `GET /health`、`GET /health/ready` | liveness と、ds41rt の `/health` と待ち行列を合わせた readiness |
| `GET /metrics`、`GET /v1/proxy/stats` | Prometheus 形式の計測と、統計の JSON |

`/chat/completions`、`/responses`、`/completions`、`/models` (`/v1` を付けない形) も同じものを返す。定義していないパスには、API の形の 404 を返す。

## 対応・非対応の機能

| 機能 | Chat | Responses | Messages | 補足 |
|---|:-:|:-:|:-:|---|
| ストリーム / 非ストリーム | ○ | ○ | ○ | 上流は常にストリーム。非ストリームは集約し、途中で切れたら 1 回だけ最初からやり直す |
| ツール (並列・strict・`tool_choice` の auto / none / required / 名前指定) | ○ | ○ | ○ | Responses の `allowed_tools` と namespace、`apply_patch` (custom) にも対応する |
| 構造化出力 | ○ `response_format` | ○ `text.format` | ○ `output_config.format` | json_schema (strict) と json_object |
| 画像入力 (data: と http(s)、16 枚まで) | ○ | ○ | ○ | 17 枚目からは古い順に `[image omitted]` にする (`limits.image_overflow`)。ツール結果の画像も可 |
| 思考 (reasoning) | ○ `reasoning_content` | ○ reasoning 項目、summary、encrypted_content | ○ thinking ブロック (signature 付き) | 既定は on・high。指定の解き方は下の「主な吸収」 |
| 思考の再注入 | ○ | ○ | ○ | 思考を送り返さないクライアントの履歴に、生成時の思考を補う |
| 状態保持 (`store`、`previous_response_id`) | - | ○ | - | sqlite に 30 日・20 GiB まで。再起動しても続く |
| `stop` / `stop_sequences` | ○ (ds41rt) | - | ○ (プロキシで照合) | |
| `n` | ○ (8 まで) | - | - | seed を省くと、本ごとに別の seed を入れる |
| `stream_options.include_usage` | ○ | - | - | OpenAI と同じく `choices:[]` の別チャンクで送る |
| usage の `reasoning_tokens` と `cached_tokens` | ○ | ○ | ○ (`cache_read_input_tokens`) | reasoning_tokens は tokenizer で数えて補う |
| `logprobs` / `top_logprobs` | × 400 | × 400 | - | ds41rt に機能が無い |
| `frequency_penalty` / `presence_penalty` | △ | - | - | 範囲だけ検査して渡す。0 以外なら `x-ds41gw-warning: ignored=…` |
| 組み込みツール (web_search、file_search、computer、bash など) | - | × 400 | × 400 | Responses では `compat.ignore_builtin_tools` に書いた型を黙って落とせる |
| `background`、`conversation`、`prompt`、`input_file` | - | × 400 | - | |
| `document` ブロック | - | - | × 400 | |
| legacy completions の `echo`、`suffix`、`logprobs`、`best_of` | × 400 | | | `/v1/completions` のみ |
| API キー認証 | ○ | ○ | ○ | 既定は off。`/v1/models`、`/health*`、`/metrics` は鍵を要求しない |

### 主な吸収 (ds41rt 直では 400 や別の挙動になるもの。仕様 §0 の 4)

- **model**: どの名前でも `deepseek-ai/DeepSeek-V4.1-Flash` に置き換えて送り、応答には要求した名前を返す (`models.alias_mode="any"`)。
- **既定値**: temperature を省くと `1.0` / `top_p 1.0` を入れる (ds41rt は省くと greedy になる)。`max_tokens` を省くと 262,144 を入れる。`max_completion_tokens` は `max_tokens` として扱う。
- **思考の指定**: `thinking` (object / bool)、`reasoning_effort`、`chat_template_kwargs` の `enable_thinking` / `thinking` / `reasoning_effort` / `clear_thinking`、
  トップレベルの `enable_thinking`、`budget_tokens` を解く。ds41rt には `thinking:{type}` と整数 (JSON number) の `reasoning_effort` だけを送る。
  詳しくは次の「reasoning effort」。
- **reasoning effort** (Chat の `reasoning_effort` と `chat_template_kwargs.reasoning_effort`、Responses の `reasoning.effort`、
  Messages の `output_config.effort` と `thinking.budget_tokens`、count_tokens、legacy completions のすべてで同じ):
  - 内部では「思考なし」か「整数 1〜100」(V4.1 のプロンプトの `Reasoning Effort: N`) のどちらかにする。
  - 段階名 (大小文字は区別しない) は次の値に写す: `none`/`off`/`disabled`→思考なし、`minimal`/`low`→50、**`medium`→62**、`high`→75、**`xhigh`→88**、`max`/`ultra`→100。
    表は `thinking.effort_map` で上書きできる (書いた鍵だけ。新しい名前も足せる)。
  - **`medium`→62 と `xhigh`→88 は公式と違う、このプロキシと ds41rt 改修版 (v15-effort) だけの独自の対応である。**
    - 根拠: DeepSeek の公式の encoding が段階の名前で定めるのは `low` (50)・`high` (75)・`max` (100) の 3 つだけで、
      `medium` と `xhigh` は無い。公式の `deepseek-recipe` 0.1.0 は OpenAI / Anthropic の API の名前を受けるために `medium` と `xhigh` を
      `high` と同じ 75 に写している (ds41rt の素の v15 も同じ)。そのままでは `medium`・`high`・`xhigh` の 3 つが同じプロンプトになり、
      クライアントが段階を変えても何も変わらない。
    - V4.1 は `Reasoning Effort: N` を 1〜100 の整数で受けるので、`medium` は low と high の間 (62、62.5 の切り捨て)、
      `xhigh` は high と max の間 (88、87.5 の切り上げ) に置いた。
    - 公式どおりに戻すには `thinking.effort_map = { medium = 75, xhigh = 75 }` と書く。ただし ds41rt 改修版も `medium`/`xhigh` を
      62/88 に写すので、ds41rt に直接送る要求とは食い違う (ds41gw は整数で送るので、ds41gw 経由の要求には影響しない)。
    - 未指定 (`thinking.default_effort = "high"`) は 75 のままで、公式と同じ。数値の `62` / `88` を送ったときと `medium` / `xhigh` は同じプロンプトになる。
  - 数値は JSON number でも数値文字列 (`"60"`) でもよい。1〜100 の整数はそのまま、`0` は思考なし、`0 < x < 1` は ×100 (旧来の互換、`0.6`→60)、
    それ以外の小数は四捨五入する。範囲外 (負、100.5 以上) と非数は 400 `invalid_reasoning_effort` (`param` は来た場所の名前、例 `reasoning.effort`)。
  - 未指定は `thinking.default_effort` (`"high"` = 75)。`budget_tokens` は 8192 未満→50、32768 未満→75、それ以上→100 (0 以上の整数か数値文字列 `"2048"`。負・小数・非数は 400 `invalid_thinking`、`param` は `thinking.budget_tokens`)。
  - 優先順位は `thinking` の on/off の明示 > `reasoning_effort` (Responses / Messages の effort もここに入る) > `chat_template_kwargs.reasoning_effort` > `budget_tokens` > 既定。
    `chat_template_kwargs.reasoning_effort` の `0` / `none` で思考が止まるのは、トップレベルの `reasoning_effort` が無いときだけ (`{"reasoning_effort":60,"chat_template_kwargs":{"reasoning_effort":0}}` は 60)。
    空文字列と空白だけの値は「指定なし」として扱う (Responses の応答にも既定の `"high"` を返す)。
  - `thinking.default_effort` は起動時に検査する。1〜100 に解けない値 (`none`、`0`、`101`、未知の名前) は設定の誤りとして起動しない (思考を既定で止めるには `thinking.default = "disabled"`)。
  - Responses の応答オブジェクトの `reasoning.effort` は要求で来た値そのまま (`60` も `"60"` も `"High"` も)。未指定なら `"high"`、思考 off なら `"none"`。
    数値で送ると数値が返る。openai SDK の型 (`Literal`) の厳密な検証には通らないが、SDK の通常の読み取りでは `r.reasoning.effort == 60` として読める (T-E1)。
  - count_tokens は ds41rt と同じ `Reasoning Effort: N` を描いて数える。
  - `thinking.numeric_upstream=false` にすると、ds41rt には段階名 (≤61→`low`、62〜88→`high`、89 以上→`max`) を送る。段階名しか受けない旧 ds41rt (v15 以前) に戻したとき用。
    境界はほぼ最寄りの段階 (中間は 62.5 と 87.5) だが、`medium` (62) と `xhigh` (88) は low / max より 1 だけ遠い `high` に含める。
    最寄りで丸めると `medium`→`low` (50)、`xhigh`→`max` (100) になり、意図 (low と high の間、high と max の間) から 12〜13 ずれるうえ、
    公式の写し方 (どちらも high) とも変わる。3 段階しか描けない上流では、公式どおり high に落とすのがいちばん驚きが少ない。
    ds41rt 改修版の V4 (数値の尺度が無い旧モデル) の描画も同じ境界 (`ReasoningEffort::level()`) にしてある。
- **messages**: `developer`→`system`、`function`→`tool`、null の content を直す。`input_text`/`output_text`/`refusal`→text、`tool_reference`→`Tool loaded: X` に変える。
  文字列の `image_url`、`input_image`、Anthropic 形式の `image` は `image_url` にそろえる。system と assistant の画像は `[image omitted]` にする。
  `reasoning`/`reasoning_text` は `reasoning_content` にし、object の arguments は JSON 文字列にする。id の無い tool_call には id を振る。数値の content は文字列にする。
- **ツール履歴の補修**: 結果の無い呼び出しには `[no result: the tool call was interrupted]` を補う。間に挟まった user は後ろに移し、孤立した tool は user にし、重複は最初の 1 件だけ残す。
  Responses の `function_call` / `function_call_output` にも同じ補修をする (recipe はそのままだと 400 にする)。補修したら `x-ds41gw-warning: repaired_tool_history` を付ける。
- **ツール定義**: `type` を補い、平坦な形を `function` の形にする。`parameters` が `{}`、`type` 無し、`["object"]` のときは `type:"object"` にする。
  名前が `^[A-Za-z0-9_-]{1,128}$` に合わないもの (`a.b` や Responses の namespace の `ns::name`) は、写した名前で送り、出力で元に戻す。
- **`tool_choice:"none"` (または tools 無し) で履歴にツールがある場合**: 履歴を平文 (`[tool call] name(args)` / `[tool result] …`) にして送る。
  さらに、出力の content に漏れた DSML ブロック (`<｜DSML｜ … </｜DSML｜ calls>`) を取り除く。
- **Responses の `max_tool_calls:0`**: `tool_choice:"none"` として送る (応答の `tool_choice` には要求どおりの値を返す)。1 以上は呼び出しの本数の上限として扱う。
- **思考の再注入**: 生成時の思考を、tool_call id、本文の SHA-256、Responses の reasoning 項目の id で覚える (プロセス内 512 MiB、24 時間)。
- **reasoning 項目の本文**: content → summary → encrypted_content (自前の形式と SGLang の `sglang-reasoning-v1.`) → id (再注入ストアか保存済みの応答) の順に探す (仕様 §6.5)。
  改ざんされた encrypted_content は空として扱い、警告ヘッダを付ける。
- **ds41rt に無いもの**: `usage.completion_tokens_details.reasoning_tokens` を補う。
- **圧縮された要求本文**: `Content-Encoding: gzip / deflate / zstd` は展開する。展開後も `max_body_mb` で抑える。

### Responses と Messages の出力の細部 (仕様 §6.7、§6.8、§7)

- Responses の `sequence_number` はプロキシが送る直前に 0 から振り直す。項目は常に 1 つだけ開く (recipe の生成器の性質)。
- `reasoning.summary` が来たら、思考を `response.reasoning_summary_*` イベントで流す。`reasoning_text.*` と二重には流さない。
- `include:["reasoning.encrypted_content"]` では `ds41gw1.<base64url(zstd(JSON))>.<HMAC-SHA256 の先頭 16 バイト>` を付ける。`output_item.done` と `response.completed` で同じ値になる。
- 応答オブジェクトに `instructions`、`metadata`、`tools`、`tool_choice`、`reasoning`、`text`、`store`、`temperature`、`top_p`、`truncation:"disabled"`、`service_tier`、`background` を補う。
- usage は ds41rt の最終値で差し替える (`cached_tokens` は `prompt_cache_hit_tokens`)。途中失敗は `response.failed` にする。
- Messages の thinking の `signature` は `ds41gw-sig-v1.` に HMAC を付けたもの (空にしない)。入力の signature は検証しない。
- Messages の `stop_sequences` はプロキシが本文と照合し、一致したら上流を切って `stop_reason:"stop_sequence"` と一致した文字列を返す。
- Messages の `message_start` の `input_tokens` は tokenizer の見積もり、`message_delta` の usage は ds41rt の最終値である。

## 同時実行・時間切れ・キャンセル (仕様 §11)

- 上流への同時数は 16 (`upstream.max_concurrency`)。超えた分は FIFO で待つ (256 本、600 秒)。満杯と時間切れは 429 (`Retry-After` 付き)、Anthropic は 529 で返す。
- n 本のストリーム (n>1、completions の複数プロンプト) は n 枠をまとめて取る。上限を超える本数は 400 `too_many_choices` にする。
- ds41rt の 429 は、バックオフ (0.5→1→2→4 秒、`Retry-After` を下限とする) して再試行する。接続拒否は 30 秒まで再試行して、だめなら 503 を返す。ds41rt の 503 は再試行しない。
- ds41rt の応答ヘッダも `upstream.idle_timeout_s` (600 秒) で諦めて、503 `upstream_timeout` を返す。
- ストリームは 15 秒無音が続くと keep-alive を送る (Chat / Responses / completions は `: keep-alive`、Messages は `event: ping`)。
  待ち行列で 10 秒待ったら先に 200 と SSE ヘッダを返し、その後の失敗はストリーム内のエラーで伝える。
- 途中失敗 (finish_reason の前の EOF) は、Chat ではエラーイベント (`code: upstream_disconnected`、`[DONE]` 無し)、Responses では `response.failed`、
  Messages では `event: error` で伝える。ds41rt のストリーム内の `event: error` の message は、そのまま理由として渡す。
- クライアントが切断したら、上流の接続も閉じる (ストリーム・非ストリームとも)。待ち行列で待っている間の切断でも、待ち人数を正しく戻す。
- 受信側の時間切れ (slowloris 対策): 要求ヘッダは 60 秒 (`server.header_read_timeout_s`) で読み終えなければ接続を切る。
  本文は、かたまりの間隔が 60 秒 (`server.body_idle_timeout_s`) を超えるか、全体で 600 秒 (`server.body_read_timeout_s`) を超えると 408 にする。
  ヘッダの時間切れは keep-alive で次の要求を待つ間にも数えるので、60 秒使わなかった接続はプロキシ側から閉じる。
- 1 応答のツール呼び出しは 32 本まで (`limits.max_tool_calls_per_response`)。Responses では `max_tool_calls` (1 以上) のほうが優先する。

## 認証・ブラウザからの要求 (仕様 §12)

- API キー (既定 off): `auth.keys = [{name, sha256, max_concurrency?}]` に書く。`Authorization: Bearer` と `x-api-key` の両方を見て、どちらかが合えば通す。
  `auth.allow_without_key` の送信元 (既定は 127.0.0.1、::1、172.17.0.0/16) は鍵が無くても通す。
- CORS (既定 off): `cors.allow_origins` に書いた Origin だけに CORS ヘッダを返す。
- 別サイトの攻撃を断る: 許可していない `Origin` からの preflight の無い POST (`text/plain` など) は 403 にする。
- DNS rebinding を断る: ブラウザの要求 (`Origin` か `Sec-Fetch-Site` が付いたもの) で Origin が許可されていないものは、
  Host が IP リテラル、`localhost`、`server.allowed_hosts` のどれかでなければ 403 `host_not_allowed` にする。
- SDK や curl は `Origin` も `Sec-Fetch-Site` も付けないので、上の 2 つの影響を受けない (Host に MagicDNS の名前を使ってもよい)。

## 計測

- `/metrics` に出すもの: `ds41gw_requests_total{api,stream,status}`、`ds41gw_queue_waiting`、`ds41gw_queue_wait_seconds`、`ds41gw_upstream_inflight`、
  `ds41gw_upstream_retries_total{reason}`、`ds41gw_midstream_failures_total`、`ds41gw_client_disconnects_total`、`ds41gw_ttft_seconds{api}`、
  `ds41gw_overhead_seconds{api}`、`ds41gw_reinject_total{result}`、`ds41gw_normalizations_total{kind}`、`ds41gw_dsml_leak_total`、`ds41gw_tool_call_cap_total`、
  `ds41gw_cross_site_rejects_total`、`ds41gw_body_read_timeouts_total`、ds41rt の `/v1/stats` の写し (`ds41rt_*`)。
- `ds41gw_overhead_seconds` は、本文を読み終えてから上流へ送り出すまでにプロキシ内で費やした時間である (待ち行列の時間は含まない)。
  仕様 §13.4 の定義のうち「上流の最初のバイトからクライアントへの最初のバイトまで」は数えていない (イベントを受けたその場で書き出すので、マイクロ秒の単位である)。

## 設定 (`config/ds41gw.toml`)

書いていない項目は既定値になる (仕様 §14)。変えたら `./run_proxy.sh start` で再起動する。主な項目は次のとおり。

| 項目 | 既定 | 内容 |
|---|---|---|
| `server.max_body_mb` | 96 | 要求本文の上限 (展開後) |
| `server.header_read_timeout_s` / `body_idle_timeout_s` / `body_read_timeout_s` | 60 / 60 / 600 | 受信側の時間切れ。0 で無効 |
| `server.allowed_hosts` | `[]` | ブラウザの要求で許す Host 名 |
| `upstream.max_concurrency` | 16 | 上流への同時数 |
| `upstream.idle_timeout_s` | 600 | 上流の応答ヘッダと無音の上限 |
| `queue.max_waiting` / `queue.timeout_s` | 256 / 600 | 待ち行列 |
| `client.keepalive_interval_s` / `early_headers_after_s` | 15 / 10 | keep-alive と早期ヘッダ |
| `models.aliases` | `DeepSeek-V4.1-Flash` ほか | `/v1/models` に出す別名 (`alias_mode="any"` なら名前は問わない) |
| `sampling.default` | `"recommended"` | temperature 省略時。`"greedy"` で ds41rt の素の挙動 **[要決定]** |
| `thinking.default` / `default_effort` | `"enabled"` / `"high"` | 思考の既定 |
| `thinking.effort_map` | `low = 50`、`medium = 62` (独自)、`high = 75`、`xhigh = 88` (独自)、`max = 100` ほか | 段階名 → effort (0〜100、0 は思考なし)。書いた鍵だけ上書きする |
| `thinking.numeric_upstream` | `true` | ds41rt に effort を整数で送る。`false` で段階名 (≤61 low、62〜88 high、他 max。旧 ds41rt 用) |
| `thinking.auto_off_below_max_tokens` | 0 (off) | これより小さい max_tokens で思考を自動で切る **[要決定]** |
| `limits.default_max_tokens` | 262144 | max_tokens 省略時 **[要決定]** |
| `limits.max_n` / `max_tool_calls_per_response` | 8 / 32 | |
| `compat.unsupported` | `"reject"` | 非対応のパラメータを 400 にする (`"ignore"` で警告だけ) |
| `compat.tool_none_strategy` | `"flatten"` | `tool_choice:"none"` での履歴の扱い (`"keep_tools"` も可) |
| `reasoning.reinject` / `reinject_store_mb` / `reinject_ttl_h` | true / 512 / 24 | 思考の再注入ストア (プロセス内。再起動で消える) |
| `reasoning.output_mode` / `always_encrypted` | `"auto"` / false | Responses の思考の出し方 |
| `state.memory_items` / `memory_mb` / `ttl_days` / `max_gb` | 2000 / 1024 / 30 / 20 | Responses の状態 (LRU と sqlite) |
| `log.bodies` / `outputs` / `images` | `"full"` / false / `"hash"` | 要求ログ。`bodies="meta"` で本文を残さない **[要決定]** |
| `log.retention_days` / `max_total_gb` | 14 / 50 | |
| `auth.keys` / `allow_without_key` | `[]` / ローカルと docker0 | API キー |
| `cors.allow_origins` | `[]` | CORS |
| `completions.enabled` / `thinking` / `default_max_tokens` | true / false / 16 | `/v1/completions` |

## `:30000` への切り替え

GLM 時代のクライアントは `:30000` を向いていた (GLM の起動スクリプトや中継の `LLM_UPSTREAM_URL`)。今は `:30000` で何も待ち受けていない (`ss -ltn`)。
切り替える方法は 2 つある (仕様 §15.3)。

1. **プロキシを `:30000` でも待ち受ける** (クライアントの設定を変えずに済む)。`:30001` と併存できる。

   ```sh
   cd proxy
   curl -s localhost:30001/v1/proxy/stats | jq '{upstream_inflight, queue_waiting}'   # どちらも 0 を確かめる
   ss -ltn | grep ':30000 '                                                           # 何も出ないこと
   DS41GW_PORT="30001 30000" ./run_proxy.sh start
   curl -s localhost:30000/v1/models | jq -r '.data[].id'
   ```

   - 再起動しても同じ口で上がるのは、`--restart unless-stopped` が同じ写し (`state/ds41gw.rendered.toml`) を読むからである。
     次に `./run_proxy.sh start` を実行するときも `DS41GW_PORT="30001 30000"` を付ける (付け忘れると `:30001` だけに戻る)。
   - 戻すときは `DS41GW_PORT=30001 ./run_proxy.sh start` を実行する。
2. **クライアント側の base URL を `:30001` に変える**。プロキシ側は何も変えない。

`:30000` で vLLM / SGLang をまた動かす場合は、先にプロキシの `:30000` を外す (口が競合して、後から起動した側の bind が失敗する)。

## 既知の制約

- ds41rt の 429 を再試行している間も、プロキシの枠を持ち続ける (仕様 §11.3 との差)。枠を返して並び直すと、後から来た要求に抜かれるためである。
  プロキシの同時数は 16 に絞ってあり、ds41rt の 429 は通常は起きない。
- Responses の `developer` ロールは recipe の意味論どおり user として描画される (仕様の表では system としていた。`recipe/…/responses/request/convert.rs` の `push_message`)。
  GLM 時代の SGLang は system として描画していたので、Responses のクライアントの約 2 万字の developer 指示は user ターンになる。
- 小さな `max_tokens` で思考が on だと、思考だけで上限に達して本文が空になりうる。T-A9 と実要求の再生では、短い要約の補助要求 (max_tokens 120、effort low、思考ありを明示) で
  10 回中 8 回、タイトル生成 (max_output_tokens 64、effort high) で 2 回とも本文が空だった。`thinking.auto_off_below_max_tokens` で切れるが、既定は off にしてある (要決定)。
- 思考の再注入ストアはプロセス内にあるので、再起動で消える。sqlite に保存した Responses の状態 (`previous_response_id` と reasoning 項目の id での引き当て) は再起動しても残る。
- `max_tool_calls` は関数呼び出しの本数として扱う (OpenAI では組み込みツールの呼び出し数を指す)。
- 文脈長の事前検査 (`limits.precheck_chars`、300 万字) は文字数で判定する。data URL は数えない。超えたものだけ tokenizer で数える。
- n>1 の要求ログの `obj` は分割前の本文である (各本の seed は `seed+i`、省略時は乱数から作る)。
- 要求ログの画像はハッシュに置き換えるので、画像つきの要求はログからそのままでは再生できない (`log.images="full"` にすれば残る)。
- ヘッダの時間切れ (60 秒) は keep-alive の待ちにも数える。60 秒より長く接続を寝かせて使い回すクライアントは、閉じた接続に送って 1 回失敗することがある
  (openai / anthropic SDK は自動で再試行する。Node の fetch と httpx の既定の寝かせ時間は 4〜5 秒である)。
- DNS rebinding の検査はブラウザの要求だけが対象である。ブラウザ以外から Host を偽る要求は、もともと Origin の制約を受けないので対象にしていない。
- ds41rt の HTTP サーバーは TCP_NODELAY を付けていない (`ds41rt-daemon/src/commands/coordinator.rs:148` の `axum::serve`)。
  このため、接続を使い回すクライアントでは ds41rt 直だと TTFT が毎回約 40 ms 遅れる。プロキシ経由でも、上流との接続プールのせいで 1 割強の要求に +20〜40 ms の外れ値が出る。
  直すには ds41rt 側に `.tcp_nodelay(true)` を足す (ds41rt には触れない約束なので直していない)。
- ds41rt の `0.0.0.0:8000` は、プロキシを経由しない直接のアクセスを許したままである (要決定)。

## 試験

公開ツリーには試験の結果ファイル (`tests/results_*.json*`) と要求ログ (`logs/`)、状態 (`state/`) を含めていない。以下の件数は手元での実行結果である。

### 単体・結合試験 (`./run_proxy.sh test`、ds41rt 不要): 98 件すべて合格

| 置き場 | 件数 | 内容 |
|---|---:|---|
| `ds41gw/src/*.rs` の `#[cfg(test)]` | 27 | 思考と effort の表 (段階名・数値・数値文字列・小数・範囲外・優先順位・`effort_map` の上書きと旧形式・`numeric_upstream`・`default_effort` の起動時検査、T-U1 の一部)、DSML フィルタを全文字境界で分割 (T-U7)、stop の照合 (T-U8)、名前の写像、encrypted_content の往復・改ざん・SGLang 形式 (T-U5)、再注入 LRU と期限、sqlite の連鎖・削除・期限 (T-U6)、LRU のバイト上限、画像プレースホルダの分割、ログの圧縮と期限・画像の伏せ字、Host の判定、事前検査の文字数 |
| `ds41gw/tests/chat.rs` | 20 | Chat の非ストリーム / ストリーム、正規化の表 (T-U1)、ツール履歴の補修 (上流本文を recipe の Chat 変換に通して確認、T-U2)、`tool_choice:none` の平文化と DSML 除去、n=2、エラー変換 (400/500/503/429 再試行/接続拒否)、途中失敗 (T-M3)、キャンセル (T-M5、T-M6)、待ち行列 (T-M1 の縮小版)、早期ヘッダと keep-alive (T-M4)、再注入、要求ログ、API キー、CORS、`/metrics` |
| `ds41gw/tests/responses.rs` | 10 | 非ストリームと補完項目、GET / input_items / DELETE、`previous_response_id` 3 段、イベント順序の検査器 (T-U4) と並列ツール、encrypted_content での思考の引き継ぎと done / completed の一致、summary、json_schema、tool_choice、組み込みツール 400、namespace と画像、`response.failed` / `incomplete`、SGLang 形式 |
| `ds41gw/tests/messages.rs` | 6 | Messages の非ストリーム / ストリーム、tool_use 往復 (`is_error`)、signature、stop_sequences、エラー、count_tokens、completions、構造化出力 |
| `ds41gw/tests/review.rs` | 13 | 敵対的レビューで直した不具合の再現 (待ち中の切断、n 本の枠の取り合い、応答ヘッダの時間切れ、両方のキーのヘッダ、別サイトの POST、警告ヘッダの長さ、`response.completed` の前の保存、圧縮本文と展開爆弾、ストリーム内 error) と端の場合 (思考だけ、思考なし、1 チャンクに 2 本の呼び出し、引数の途中の切断、40 MiB の data URL を 3 API で) |
| `ds41gw/tests/hardening.rs` | 7 | 最終段で直した点: `max_tool_calls:0`、reasoning 項目の id だけからの思考 (再注入ストアと sqlite の両方)、DNS rebinding、遅いヘッダの切断、遅い本文の 408 (間隔と合計)、`ds41gw_overhead_seconds` と要求ログの整形 |
| `ds41gw/tests/effort.rs` | 10 | effort を Chat (`reasoning_effort`、`chat_template_kwargs`)、Responses (応答の `reasoning.effort`、ストリーム、GET も)、Messages (`output_config.effort`、`budget_tokens` の境界)、count_tokens (描画が `Reasoning Effort: N`)、completions のそれぞれで、数値・数値文字列・段階名・0・範囲外を確かめる。`numeric_upstream=false`、`effort_map` の上書き、vendoring した recipe が数値を解けること |
| `ds41gw/tests/lossless.rs` | 5 | T-U3: Responses / Messages → `Conversation` (A) → Chat → recipe の Chat 変換 (B) で、描画したプロンプトが一致する |

上流は `ds41gw/tests/common/mod.rs` の mock である (要求本文の `#mock:<名前>` で振る舞いを選ぶ)。

### 実機試験 (ds41rt 経由、逐次、`max_tokens` ≤ 600)

```sh
tests/.venv/bin/python tests/sdk_test.py --out tests/results_sdk.json                        # 24/24 合格
tests/.venv/bin/python tests/sdk_conformance.py --out tests/results_sdk_conformance.json     # 29/29 合格
```

どちらも openai 3.35.1 と anthropic 1.15.0 を使う (`tests/requirements.txt`)。

- `sdk_test.py` は機能を見る。T-C1〜C7、T-R1〜R7、T-A1〜A6、T-A9、T-X1、T-X2 (仕様 §16.3) と T-E1 (reasoning effort の数値・数値文字列・段階名・0・101 を 3 API で)。
  - T-C2: 2 ターン目で思考を返さなくても再注入され、`prompt_cache_hit_tokens` が 310 になった。
  - T-C5: `tool_choice:"none"` とツール履歴で、DSML は漏れなかった。
  - T-R2: temperature 1 なので effort low (50) でも思考が `max_output_tokens` を使い切ることがある。そのときは `response.incomplete` (reason `max_output_tokens`) で終わるのを正しい終わり方として検査する。
  - T-A6: `count_tokens` の 45 と実際の `input_tokens` の 45 が一致した。
- `effort_live.py` は effort を公式 SDK で 5 つの口 (Chat の `reasoning_effort` と `chat_template_kwargs`、Responses、Messages の `output_config.effort` と `budget_tokens`、completions) に送り、要求ログの `obj` で ds41rt に届いた値を確かめる (数値 1/37/60/100、`"60"`、0.37、59.5、全段階名、未指定、不正値 12 種、ストリーム、count_tokens と実際の入力トークン数): 170/170。
- `effort_direct.py` は temperature 0・同じ seed で、プロキシ経由 (3 API) と ds41rt 直 (effort 1/37/50/60/62/75/88/100/思考なし) の出力を突き合わせる: 37/37 が一致 (`medium`→62、`xhigh`→88 を含む)。
- `sdk_conformance.py` は形と順序を見る。K-C1〜C9 (Chat)、K-R1〜R11 (Responses)、K-X1 (completions)、K-A1〜A8 (Messages)。
  - SDK の HTTP 層に記録用の transport を挟み、生の SSE の各イベントと非ストリームの本文を SDK の型で pydantic の厳密検証に通す。
  - イベント列の文法 (連番、開閉の対応、delta の連結と done の一致、終端の output と done の列の一致) と、SDK の補助 API (`stream`、`parse`、`pydantic_function_tool`、自動ページ送り) を確かめる。
  - キャンセル (ストリームを途中で閉じる、非ストリームの時間切れ) の後、要求ログの `request.finished` が `client_disconnected:true` で即座に出ることを確かめる。

### GLM 要求ログの再生 (仕様 §16.4、詳細は [`docs/research/r7_replay_real.md`](../docs/research/r7_replay_real.md))

- 再生に使った要求ログと要求集 (`tests/replay_corpus.jsonl`) は私的なデータなので公開していない。道具 (`tests/build_replay_corpus.py`、`replay_*.py`) は自分の要求ログ (`--reqlog`) を用意すれば使える。クライアントの型の判別規則は環境変数 `REPLAY_LABELS` の JSON で与える (`tests/replay_compat.py` の `label()` の説明を参照)。

- 互換性の総当たり (`tests/replay_compat.py`、max_tokens=1、非ストリーム): 77/77 件が 200。ds41rt 直では 6/6 件 400 だった `tool_reference` の形も通った。
- ストリームの再生 (`tests/replay_capped.py`、同時 4、max_tokens 64、ストリームのクライアントの 24 件): 途中失敗 0 件。
- 実要求の再生 (`tests/build_replay_corpus.py` → `tests/replay_real.py run`、同時 4、出力 256): Chat 66 件と、描画済みプロンプトから復元した Responses 30 件で 96/96 合格。
  ツール呼び出し 58 本のうち arguments が読めなかったのは、出力上限で切れたものだけだった。
- 変形の再生 (`tests/replay_variants.py`、28 件): ストリームと非ストリームの入れ替え、encrypted_content と summary、Responses の 2 ターン往復で 28/28 合格。

### 追加遅延

- mock (T-M2、同時 16、`r7_eval/bench.py`): TTFT p50 の追加は +0.3〜0.6 ms。chat の差分間隔 p99 は悪化しない。
- 実機 (`tests/replay_real.py pairs --conc 1 --abba --max-tokens 8`): 実要求 68 件を、ds41rt 直とプロキシ経由で D P P D / P D D P の順に送って比べた。
  68 組すべてが両経路とも 200 だった。TTFT の差 (プロキシ経由 − 直) は次のとおり。

  | | 組 | p50 | p10 | p90 |
  |---|---:|---:|---:|---:|
  | Chat | 36 | +1.6 ms | −0.1 ms | +11.8 ms |
  | Responses | 20 | +2.9 ms | +1.9 ms | +22.3 ms |
  | 全体 | 56 | +2.4 ms | +0.2 ms | +20.9 ms |

  要求ログの整形を書き込みスレッドに移す前は、全体の p50 が +3.5 ms (Chat +3.4 ms) で、本文の大きさに比例していた。
  p90 の +20 ms 前後は、ds41rt が TCP_NODELAY を付けていないことによる外れ値である ([既知の制約](#既知の制約))。
  結果は `tests/results_replay_pairs_c1_final.jsonl` (最終版) と `tests/results_replay_pairs_c1.jsonl` (移す前)。
- `ds41gw_overhead_seconds` (受信から上流へ送り出すまで、要求ログの `overhead_ms`): 上の再生と SDK 試験を含む最終版の要求では、
  Chat が p50 0.62 ms / 最大 2.4 ms (113 件)、Responses が p50 1.03 ms / 最大 2.9 ms (85 件)、Messages が p50 0.17 ms (20 件) だった。

## 決めていただきたいこと (仕様 §18.1) と現在の値

| 事項 | 現在の値 (仕様の提案どおり) | 設定 |
|---|---|---|
| temperature 省略時の既定 | 1.0 / top_p 1.0 | `sampling.default = "recommended"` (`"greedy"` で ds41rt の素の挙動) |
| `max_tokens` 省略時の既定 | 262,144 | `limits.default_max_tokens` |
| `medium` / `xhigh` の扱い | 62 / 88 (独自の対応。公式は 75 / 75。上の「reasoning effort」を参照) | `thinking.effort_map.medium` / `.xhigh` (例 `75` で公式と同じ) |
| `:30000` への切り替え | 未実施。30001 だけ | [`:30000` への切り替え](#30000-への切り替え) |
| ds41rt の `0.0.0.0:8000` | 未変更 | ds41rt の次の計画的な再起動で判断 |
| 小さな max_tokens で思考を自動 off | off | `thinking.auto_off_below_max_tokens` |
| 要求ログの本文 | 全文 (14 日・50 GiB) | `log.bodies = "full" \| "meta"` |

## ファイル

```
ds41gw/            Cargo プロジェクト (src/ 約 7 千行、tests/ 約 2 千行)
  src/lib.rs         ルータ、受信 (hyper http1 と時間切れ)、本文の読み取りと展開
  src/chat.rs        Chat の正規化 (全 API 共通の最後の段) とハンドラ
  src/responses.rs   Responses の前処理・逆写像・イベント後処理・状態 API
  src/messages.rs    Messages と count_tokens
  src/completions.rs /v1/completions
  src/core.rs        文脈とログ、Source (ツール上限・stop 照合・集計)、早期ヘッダ、SSE
  src/upstream.rs    待ち行列・再試行・SSE 解析
  src/reverse.rs     Conversation → Chat
  src/auth.rs        API キー、CORS、別サイトと DNS rebinding の拒否
  src/{thinking,names,dsml,reasoning,store,reqlog,tokens,metrics,models,config,util,error}.rs
docker/Dockerfile  多段ビルド (rust:1.90-bookworm → debian:bookworm-slim)。文脈に vendor/ を含む
vendor/deepseek-recipe-patched/  数値 effort 対応版の recipe 3 クレート (ds41rt 改修版の写し)
config/ds41gw.toml 設定 (コンテナ内のパス)
run_proxy.sh       ビルドと起動
cargo.sh           rust コンテナで cargo を実行する補助
tests/             SDK 試験、GLM ログの再生、結果
state/ logs/       実行時に作られる (0700)
r7_eval/           以前の評価スクリプト (bench.py)
```
