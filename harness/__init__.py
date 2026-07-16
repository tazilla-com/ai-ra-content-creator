"""AI-RA content-creator evaluation & evidence harness.

A committed, live-only harness that drives the real wizard generators and the
Asset-workshop agent against whatever LLM provider and catalog provider the
application is configured with (see ``.env``), captures per-step results
(tokens, cost, latency, status), verifies correctness, and renders a report.

It has two jobs:

1. **Evidence** — produce representative, reproducible results for the
   "Testing and deployment evidence" appendix (normal operation, provider
   error, retry/fallback, user validation, release to production).
2. **Verification / looping** — re-run after changes to confirm the wizard and
   agent still behave correctly, and to gather stability statistics over N reps.

Design notes:
- **Live only.** There is no fake/scripted LLM. Runs use the configured
  provider (OpenAI or Azure) via the application's own client factories.
- **Config-driven catalog.** Whatever ``WORKSHOPS_AI_CATALOG__MODE`` is set to
  (``http`` → a Tazilla-compatible catalog service such as the bundled
  ``mock-tazilla``; or ``file`` → the bundled CSVs) is used as-is.
- **Runs inside the stack.** Intended to run in the ``api`` container so that
  the catalog endpoint (e.g. ``http://mock-tazilla:9000``) and provider
  credentials resolve exactly as in the deployed service.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
