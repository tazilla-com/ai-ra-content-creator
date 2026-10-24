# `harness/` — evaluation & evidence harness

A committed, **live** test harness that drives the real wizard generators and the
Asset-workshop agent against the application's configured LLM and catalog
providers, verifies the results, and renders a report. It exists to:

1. **Produce testing & deployment evidence** — representative, reproducible
   results across the categories for an audit.
2. **Verify correctness on demand** — re-run after a change to confirm the wizard
   and agent still behave correctly, and gather stability statistics over N
   repetitions (looping).

It is a **test feature, built and run separately from production** — it is *not*
started by `docker compose up`, and it is *not* baked into the application image.

## Principles

- **Live only.** No fake/scripted LLM. Runs use the configured provider
  (OpenAI **or** Azure) through the application's own client factories, so the
  harness exercises exactly what production runs.
- **Config-driven.** Whatever provider and catalog the application is configured
  with are used as-is (see [Configuration](#configuration)). The harness never
  hard-codes a provider or catalog; it fails fast with a clear message if the
  configured catalog is unreachable.
- **Containerised, no local Python.** It runs as its own one-shot container built
  from the application image, on the same compose network, so `mock-tazilla`,
  `redis` and the provider credentials from `.env` all resolve.
- **Real faults, not fakes.** Provider-error / timeout / retry evidence is
  produced by inducing *real* faults through configuration (a 1-second call
  timeout; a tiny per-request token cap that triggers a genuine
  `RequestTooLargeError` and the real recursive batch-split), never by scripting
  model output.
- **No secrets.** Credentials are read only indirectly via the application
  settings; the harness never prints or writes keys, and reports are
  redaction-safe.

## Layout

```
harness/
├── README.md                    # this file
├── docker-compose.harness.yml   # the opt-in `harness` service (compose profile: harness)
├── __init__.py
├── bootstrap.py                 # runtime init (settings, throttle), config summary, catalog preflight
├── steps.py                     # drive services→assets→impacts→risks→controls + the agent
├── verify.py                    # correctness verifiers per step (schema, CIA, catalog membership, HITL)
├── faults.py                    # config-induced REAL faults (timeout; tiny token cap → batch-split)
├── prompt_override.py           # --prompts-dir: prompts from files, to compare prompt versions
├── report.py                    # records → redaction-safe Appendix-D markdown
├── release_evidence.py          # non-destructive deployment record (host/CI)
├── runner.py                    # CLI entry point
├── selftest.py                  # zero-dependency runner for the deterministic tests
├── scenarios/                   # production-shape inputs (sample_org.json drives the live chain)
├── tests/                       # deterministic tests (verifiers, report, config redaction)
└── audit/                       # read-only instrumentation: what actually runs (see below)
```

## `audit/` — establishing which code actually runs

The rest of the harness answers *"is the output correct?"*. `audit/` answers a different question:
**"which code was executed at all?"** — the one static analysis cannot answer, because a module can be
imported and still never run.

Nothing in the application imports it; every module is a standalone `python -m` entry point. It uses
only the standard library (`sys.setprofile`), so it runs inside the production image unchanged.

| Module | What it does |
|---|---|
| `exec_probe.py` | records every Python function entered, keyed `file:line:qualname`. Also wraps any module: `python -m harness.audit.exec_probe -m harness.runner chain` |
| `inventory.py` | every function/method in the tree, from the AST — the denominator |
| `surface_drive.py` | drives the real ASGI app **in-process** over every route incl. the WebSocket, under the probe |
| `report_coverage.py` | joins inventory × probe → per-file "never executed" table |
| `langfuse_recon.py` | read-only dump of the hosted prompt registry, traces and observations |

```bash
# inside the running api container (or any container built from this image)
docker exec workshops-ai-api python -m harness.audit.inventory --base /app --out /app/logs/audit/inventory.json
docker exec workshops-ai-api python -m harness.audit.surface_drive --no-llm --out /app/logs/audit/exec.json   # free
docker exec workshops-ai-api python -m harness.audit.surface_drive          --out /app/logs/audit/exec.json   # live, ~$0.04
docker exec workshops-ai-api python -m harness.audit.report_coverage \
    --inventory /app/logs/audit/inventory.json --exec /app/logs/audit/exec.json
docker exec workshops-ai-api python -m harness.audit.langfuse_recon
```

Artifacts land in `/app/logs/audit/`, which is bind-mounted, so they survive a redeploy.

**Why `surface_drive` runs in-process rather than over HTTP:** the probe has to live in the same
interpreter as the code it records. Driving the app through `starlette.testclient` exercises the genuine
middleware stack, dependency graph, routers and WebSocket handler while keeping everything in one
process the probe can see.

**Two traps worth knowing before reusing this:**

- The probe→inventory join must be on **qualname, not line number**. For a decorated function CPython
  reports `co_firstlineno` at the first decorator while the AST reports the `def`; joining on line
  silently marks every decorated endpoint as never-run.
- The probe is **function-level**. A 2,000-line file with 7 methods yields 7 data points, so it cannot
  distinguish branches inside a large method.

`surface_drive` also covers a gap the rest of the harness has by design: the harness drives the
generators **in-process**, so it never exercises the API's request validation, middleware, auth or
session layers. (That gap is why nobody had noticed that `scenarios/sample_org.json`'s
`organization_profile` carries a `sector` key, which `api/schema/organization.py:Organization` —
`extra="forbid"` — rejects with 422 when the same profile is POSTed to the wizard REST API.)

## Prerequisites

- The application image exists (`docker compose build api` — the normal stack build).
- `.env` is configured with a working provider and catalog (the same file the
  service uses). The harness inherits it.

## Running

Every command uses the three compose files (base + override + harness overlay).
The harness starts `redis` and `mock-tazilla` automatically as dependencies:

```bash
# Full Appendix-D evidence suite (chain + agent + timeout fault + token-limit fault + report)
docker compose -f docker-compose.yaml \
  -f harness/docker-compose.harness.yml run --rm harness
```

The default command is `python -m harness.runner evidence --reps 1`. Override it
by appending your own command:

```bash
# Full wizard chain only, verified, 2 reps
docker compose ... run --rm harness python -m harness.runner chain --reps 2

# Drive the agent; asserts it PROPOSES changes and calls only known non-writing tools
docker compose ... run --rm harness python -m harness.runner agent \
  --message "Create an asset for the customer database and set confidentiality to PROTECTED."

# Induce a real fault
docker compose ... run --rm harness python -m harness.runner fault --kind timeout
docker compose ... run --rm harness python -m harness.runner fault --kind token_limit

# Deterministic tests (no provider / tokens)
docker compose ... run --rm harness python -m harness.selftest

# Re-render a report from a saved run
docker compose ... run --rm harness python -m harness.runner report \
  --records harness/runs/<run-id>/records.json
```

Deployment record (run on the **host** or in CI, where git + the docker CLI exist):

```bash
python -m harness.release_evidence --base-url http://localhost:8000
```

Run artifacts land on the host under `./harness/runs/` (bind mount).

## Configuration

The harness takes **no configuration of its own** — it reads the application
settings, so it always tests whatever the service is configured to do. The
settings that shape a run (all from `.env` / the compose environment):

| Setting | Effect on the harness |
|---|---|
| `WORKSHOPS_AI_LLM__PROVIDER` | `azure` or `openai` — which provider every call uses |
| provider credentials (`WORKSHOPS_AI_LLM__API_KEY`, `…__AZURE_ENDPOINT`, `…__AZURE_API_VERSION`, or `OPENAI_API_KEY`) | authenticate the live calls |
| `WORKSHOPS_AI_WIZARD__DEFAULT_MODEL` | model for the wizard steps |
| `WORKSHOPS_AI_LLM__MODEL` | model for the agent |
| `WORKSHOPS_AI_WIZARD__REASONING_EFFORT` | default reasoning effort (overridable per run) |
| `WORKSHOPS_AI_CATALOG__MODE` (`http` / `file`) | catalog source; `http` uses `…__ENDPOINT` (e.g. `http://mock-tazilla:9000`), `file` uses the bundled CSVs |
| `WORKSHOPS_AI_REDIS__URL` | session store; the harness service points this at `redis://redis:6379/0` |
| `WORKSHOPS_AI_OPENAI_THROTTLE__*` | rate limits applied during runs |

Per-run CLI flags (see `--help` on any subcommand):

| Flag | Default | Applies to |
|---|---|---|
| `--scenario` | `harness/scenarios/sample_org.json` | chain, fault, evidence |
| `--reasoning` | `medium` | all live runs |
| `--model` | configured default | all live runs |
| `--language` | `en` | chain, evidence |
| `--max-services` / `--max-assets` / `--max-controls` | 3 / 5 / 5 | chain, evidence |
| `--reps` | 1 | chain, evidence |
| `--message` | a create request | agent |
| `--no-expect-proposal` | a proposal is required | agent: for a message that asks no change |
| `--prompts-dir` | — (prompts from Langfuse) | all live runs: serve each prompt with a file `DIR/<name>.json` from that file (`prompt_override.py`) |
| `--kind` | — (`timeout` \| `token_limit`) | fault |
| `--records` | — | report |

## Evidence coverage

Each category of the testing/deployment appendix, and how the harness produces its evidence:

| Appendix category | How the harness produces it |
|---|---|
| Normal operation | `chain` / `evidence` return schema-valid output; `verify.py` asserts structure, CIA values, risk scale, and controls selected-from-and-enriched-by the catalog |
| Malformed output | guarded by structured outputs; the representative differential (a fixed production incident) is the project `CHANGELOG.md` 67%→0% result — reproducing malformed output live is non-deterministic and is not forced |
| Timeout / provider error | `fault --kind timeout` induces a real 1s-timeout failure and asserts it surfaces cleanly |
| Retry / fallback | `fault --kind token_limit` triggers a real `RequestTooLargeError` → asserts the recursive batch-split recovers |
| User validation | the agent run asserts it returns **proposals** (generated assets / deletions; a reply without one fails) and uses only known non-writing tools. That it has **no** write path to the Tazilla system of record is a property of its tools (`workshops_ai/graph/assets/tools/`), not a runtime check |
| Release to production | `release_evidence.py` captures source revision, container health, image identity, and `/health` (status / version / environment) |

## Output

Each run writes to `harness/runs/<timestamp>_<kind>_<id>/`:

- `records.json` — the full structured run (config, per-step results, checks, faults)
- `report.md` — the rendered, redaction-safe report (paste into the appendix)
- `results.jsonl` — one line per notable record (step / agent / fault)

`harness/runs/` is git-ignored; copy the specific representative runs you want to
cite into the report rather than committing every run.

## Tests

The deterministic parts (verifiers, report rendering, config redaction, scenario
loading) run without a provider. The application image does not ship `pytest`, so
use the zero-dependency runner — or `pytest` if you have it:

```bash
docker compose ... run --rm harness python -m harness.selftest   # no deps
pytest harness/tests/                                             # if pytest installed
```

The end-to-end live runs (`chain` / `agent` / `fault` / `evidence`) are triggered
explicitly — they cost provider tokens and are non-deterministic.
