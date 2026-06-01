# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-`0.1.0` history is available in `git log`.
`1.0.0` is reserved for the first stable production release.

## [Unreleased]

### Config
- **Renamed** `WORKSHOPS_AI_LLM__HTTP_TIMEOUT_SEC` → `WORKSHOPS_AI_LLM__CALL_TIMEOUT_SEC` (and `LLMSettings.http_timeout_sec` → `call_timeout_sec`). The old name was confusing — it's a per-LLM/agent-call timeout, not an HTTP-protocol-level timeout, and was being conflated with `WORKSHOPS_AI_CHAT__TIMEOUT_SEC` (whole chat-request timeout including multiple LLM calls + tools). **Backward compatible** — the old env name is still accepted via `AliasChoices`.

### Observability
- **Langfuse spans** now include `reasoning_effort` in metadata for all 5 wizard generators (`services`, `assets`, `impacts`, `risks`, `controls`) and `web_search`. Resolves the cascade (body > env > default) so traces show what was actually applied.

### Added
- **Cancelled-call usage recovery** — when the client closes the HTTP connection before the wizard response is delivered (e.g. user navigates away while the LLM call is in flight, but Azure has already consumed tokens), the call's usage is persisted to a per-session ledger in `session.wizard_metadata["pending_usage"]` (keyed by model). The next successful `/wizard/*` response rolls these recovered tokens into its `usage` field. `UsageMetrics` gains two backward-compatible fields: `includes_recovered: bool` and `recovered_cost_usd: float` (recovered cost is *included* in `cost_usd`). New helper module `wizard/pending_usage.py`; aggregate-per-model storage shape cannot grow beyond the number of allowed models, so no eviction logic is needed.
- **Per-model TPM enforcement** — new env `WORKSHOPS_AI_WIZARD__OPENAI_CAPACITY_LIMITS` (per-model tokens-per-minute capacity, matches Azure deployment quotas). Enforced via per-model 60-second sliding window in `OpenAIThrottle`: calls that would push the rolling sum over the limit wait until the window slides, instead of failing. Independent from per-request limits (which still raise `RequestTooLargeError` with batch-split hint).
- **Default per-request limits** updated for current allowed models: `gpt-5-mini` (300k), `gpt-5-nano` (500k), `gpt-5.1-chat` (300k), `gpt-5.4-mini` (300k), `gpt-5.4-nano` (500k), `gpt-4.1` (300k). `gpt-5` (plain) removed from defaults — use `gpt-5-mini` or `gpt-5-nano` instead.

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
