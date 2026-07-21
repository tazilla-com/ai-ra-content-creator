# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-`0.1.0` history is available in `git log`.
`1.0.0` is the first stable production release.

## [1.1.0] — 2026-07-21

Both fixes in this release share one root cause: the published repository depended on files that
existed only on a developer machine, so a clean clone could not build or run the harness.

### Fixed
- **Harness could not start from a clean clone** — `harness/docker-compose.harness.yml` declared a
  `depends_on: mock-tazilla`, but that service was defined only in the gitignored
  `docker-compose.override.yml`. The harness now defines `mock-tazilla` itself, so `http`-mode catalog
  runs work from a fresh checkout with no local-only files.
- **Docker build failed from a clean checkout** — `Dockerfile` did `COPY constraints.txt ./constraints.txt`
  unconditionally in both the builder and runtime stages, while `constraints.txt` was gitignored in
  `1.0.0`. `COPY` of a missing path is a hard error, so `docker compose up --build` — the README's own
  quick start — failed before installing any dependency. The surrounding `if [[ -f constraints.txt ]]`
  guards show the file was always meant to be optional; the `COPY` simply ran before them. Now
  `COPY constraints.tx[t] ./`, whose bracket glob is a no-op when the file is absent. The empty
  placeholder `constraints.txt` has been removed, so a local build and a clean-clone build are
  identical. Drop a real pin file in place to re-enable constrained installs.

## [1.0.0] — 2026-07-16

### Added
- **Evaluation & evidence harness** (`harness/`) — an opt-in, containerised live harness that drives the wizard chain and the Asset-workshop agent against the configured provider/catalog, verifies results, induces real faults, and renders testing/deployment evidence. Excluded from `docker compose up`; see `harness/README.md`.
- **Cancelled-call usage recovery** — when the client closes the HTTP connection before the wizard response is delivered (e.g. user navigates away while the LLM call is in flight, but Azure has already consumed tokens), the call's usage is persisted to a per-session ledger in `session.wizard_metadata["pending_usage"]` (keyed by model). The next successful `/wizard/*` response rolls these recovered tokens into its `usage` field. `UsageMetrics` gains two backward-compatible fields: `includes_recovered: bool` and `recovered_cost_usd: float` (recovered cost is *included* in `cost_usd`). New helper module `wizard/pending_usage.py`; aggregate-per-model storage shape cannot grow beyond the number of allowed models, so no eviction logic is needed.
- **Per-model TPM enforcement** — new env `WORKSHOPS_AI_WIZARD__OPENAI_CAPACITY_LIMITS` (per-model tokens-per-minute capacity, matches Azure deployment quotas). Enforced via per-model 60-second sliding window in `OpenAIThrottle`: calls that would push the rolling sum over the limit wait until the window slides, instead of failing. Independent from per-request limits (which still raise `RequestTooLargeError` with batch-split hint).
- **Default per-request limits** updated for current allowed models: `gpt-5-mini` (300k), `gpt-5-nano` (500k), `gpt-5.1-chat` (300k), `gpt-5.4-mini` (300k), `gpt-5.4-nano` (500k), `gpt-4.1` (300k). `gpt-5` (plain) removed from defaults — use `gpt-5-mini` or `gpt-5-nano` instead.
- **Langfuse spans** now include `reasoning_effort` in metadata for all 5 wizard generators (`services`, `assets`, `impacts`, `risks`, `controls`) and `web_search`. Resolves the cascade (body > env > default) so traces show what was actually applied.

- **`LICENSE`** — the project is now released under the **Apache License 2.0**.

### Removed
- **`openapi.json`** (repo root) — a stale generated artifact advertising version `1.0.0` with none of
  the `/wizard/*` endpoints; it predated the `1.0.0` → `0.1.0` baseline reset and contradicted both
  the code and the README. FastAPI serves a live, always-current schema at `/openapi.json` when
  `WORKSHOPS_AI_ENABLE_DOCS=true`.
- **Root scratch files untracked** — `resp.json`, `test_sc.py`, `test_asset_agent.py`, `commands.txt`,
  `constraints.off`, `constraints.txt`. Ad-hoc working files; the real suite is in
  `workshops_ai/tests/`. Now gitignored (local copies unaffected).
- **`.gitignore`** additionally covers local-only material: `docker-compose.override.yml`, `.claude/`,
  `support/`, `_lf_*.json`.

### Changed
- **Renamed** `WORKSHOPS_AI_LLM__HTTP_TIMEOUT_SEC` → `WORKSHOPS_AI_LLM__CALL_TIMEOUT_SEC` (and `LLMSettings.http_timeout_sec` → `call_timeout_sec`). The old name was confusing — it's a per-LLM/agent-call timeout, not an HTTP-protocol-level timeout, and was being conflated with `WORKSHOPS_AI_CHAT__TIMEOUT_SEC` (whole chat-request timeout including multiple LLM calls + tools). **Backward compatible** — the old env name is still accepted via `AliasChoices`.

### Documentation
- **`docs/websocket-api.md`** — new. The Asset-workshop WebSocket protocol, which OpenAPI cannot
  describe and which was previously undocumented in the repo: handshake and query params,
  authentication (including the 4401 close code and why browser clients must use `?api_key=`), all
  four client frames and all six server frames, error codes, limits, and the catalog
  `scraped-content` contract. Corrected against the implementation while writing: the source
  document described a `tool_status` frame that no code path emits, a 120 s request timeout (the
  setting is 75 s and is **not enforced** — the agent task is awaited without a deadline), and a
  10-message history cap (the real cap is 20 messages / 10 turns). It also omitted authentication
  entirely, plus the `progress` and `ping`/`pong` frames, which are emitted. Gaps are now marked
  rather than silently documented as working.
- **`docs/websocket-api.md` — field tables for `AssetInput` and `Organization`,** and a fix to the
  `chat` example, which did not validate. `Organization.numberOfEmployees` is `Optional[str]` — free
  text like `"50-100"` or `"~200"` is the point — but the example passed `150` as an integer, and
  Pydantic v2 does not coerce int → str, so the payload was rejected with *"Input should be a valid
  string"*. Both payload models are `extra="forbid"`, so a wrong or unknown field is a hard error
  rather than an ignored value; the example is now verified against the real `ChatPayload` model.
- **`README.md` — the Asset workshop is documented for the first time.** The README described only
  the wizard half; the LangGraph ReAct chat agent, its seven tools, the `/demo` UI, and the fact that
  both halves share one session (the agent reads the wizard's services and writes back into
  `wizard_steps["assets"]["assets"]`) were entirely absent. Also documents the catalog `file`/`http`
  modes and `scripts/mock_tazilla_api.py` as a reference catalog backend for local `http`-mode work.
- **`README.md` rewritten to match the current API.** It had drifted far enough to be actively
  misleading: it documented `GET`/`DELETE /wizard/state` (never existed — the real endpoints are
  `GET`/`DELETE /session`), described sessions as keyed by `org_id`/`user_id` query params (they are
  keyed by the `X-Session-Id` header), and used `number_of_services` / `number_of_assets` request
  fields (the real fields are `min_services`/`max_services` and `min_assets`/`max_assets` — and since
  wizard inputs are `extra="forbid"`, the documented bodies returned **HTTP 422**). It also pointed
  wizard model config at `WORKSHOPS_AI_LLM__MODEL` / `__TEMPERATURE`, which configure the chat/agent
  path and are silently ignored by the wizard (`WORKSHOPS_AI_WIZARD__DEFAULT_MODEL` /
  `__TEMPERATURE` are the wizard's). Now documented: both **OpenAI and Azure OpenAI** providers,
  `reasoning_effort`, per-model TPM throttling, the control/threat catalogs, Langfuse prompt
  management and tracing, Docker deployment, cancelled-call usage recovery, and the
  `DISABLE_ALL_AUTH` / `CORS_ORIGINS=["*"]` / `ENABLE_DOCS` security notes.

### Fixed
- **`/wizard/controls` 500 errors ("The model produced invalid content")** — the batch controls call used the full `PlannedControl` / `AssetControls` models as the OpenAI Structured Outputs `response_format`. Their `Optional[Decimal]` cost fields and nested threat-coverage objects generate `anyOf` + regex `pattern` schemas that the system prompt simultaneously forbids; under gpt-5.4-mini (medium reasoning) the contradiction drove the model off-schema and Azure returned 500. The model now emits a minimal name-only DTO (`ControlSelection` / `AssetControlSelection`); full control metadata is enriched from the catalog after generation. **HTTP API output unchanged.**
- **`/wizard/risks` 500 errors (and intermittent JSON truncation)** — same class of contradiction: `Risk` / `AssetRisk` carried `Optional[Decimal]` computed fields (`inherentRisk`, `residualRiskCalculated`, `estimatedResidualRisk`) that the prompt forbids. The model now emits a minimal DTO (`RiskSelection` / `AssetRiskSelection`, `threatName` + `probability`); full risk objects are rebuilt after generation with the computed values left unset, as before. Verified on gpt-5.4-mini / medium: 67% → 0% batch-call failures. **HTTP API output unchanged.**
- **Malformed JSON example in the controls/risks prompts** — the `{controls_json_schema}` / `{risks_json_schema}` example values used escaped `{{ }}` braces, which LangChain substitutes verbatim, so the model was shown literal `{{ }}` instead of valid JSON. Corrected to plain `{ }`.

## [0.1.0] — 2026-05-07

First tracked release on the `0.x` versioning scheme.

### Config & request-schema changes
- **Added** — optional body field `reasoning_effort` on all `/wizard/*` endpoints (`services`, `assets`, `impacts`, `risks`, `controls`). Accepts `minimal | low | medium | high`. Backwards compatible — omitting it preserves prior behaviour.
- **Cascade priority** for `reasoning_effort`: per-call body > env `WORKSHOPS_AI_WIZARD__REASONING_EFFORT` > Pydantic default (`medium`).
- **Web search reasoning effort** stays separate — `WORKSHOPS_AI_WIZARD__WEB_SEARCH_REASONING_EFFORT` is independent of wizard reasoning effort.
- No environment variables added, removed, or renamed.
- `env.example` / `WORKSHOPS_AI_VERSION` baseline reset from `1.0.0` to `0.1.0`.

### Breaking changes
- **Langfuse prompt variable** `relevant_controls_json` (controls prompt) — shape changed from per-asset list (`[{assetName, controls: [...]}]`) to flat deduplicated catalog (`[{name, description, category, risk_coverage}, ...]`). Field `catalog_coverage` renamed to `risk_coverage`. **Custom Langfuse prompts referencing `{relevant_controls_json}` must be reviewed** to align with the new shape.
- **HTTP API output is unchanged** — this only affects internal LLM prompt structure.

### Added
- **New Langfuse prompt variable** `controls_per_asset_json` (controls prompt) — `[{assetName, controls: [name1, name2, ...]}]`. Available as a fallback mapping; the default prompt does not reference it.
- **Reasoning effort cascade** end-to-end (schema → `create_wizard_llm_client` → all 5 generators incl. batch operations).
- **Control catalog deduplication** — controls in the LLM prompt are now deduplicated. The catalog is built from the original full catalog and filtered through the union of threat names across the batch; each control's `risk_coverage` includes all batch-relevant threats. Significantly reduces prompt token usage at scale.

### Fixed
- **Services session leakage** in `/wizard/assets`, `/wizard/impacts`, `/wizard/risks` — explicit `services: []` in the request body no longer falls back to session services. Previously `if not services` treated `[]` as truthy and triggered the session fallback. Now `if services is None` distinguishes "not provided" from "explicitly empty".
