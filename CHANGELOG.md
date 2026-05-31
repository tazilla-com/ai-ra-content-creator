# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Pre-`0.1.0` history is available in `git log`.
`1.0.0` is reserved for the first stable production release.

## [Unreleased]

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
