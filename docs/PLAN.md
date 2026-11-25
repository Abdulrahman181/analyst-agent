# Project plan

## Goal

Build an analyst agent that answers Arabic and English questions about a SQLite database, cannot
modify the data, cannot return a number it did not retrieve, and comes with a harness that turns any
model into a comparable score in both languages.

## Requirements

Functional

- Accept a question in Arabic or English and answer it from a SQLite database through a fixed tool
  set (schema inspection, read-only SQL, final answer).
- Validate every query before it runs and refuse anything that is not a read-only `SELECT`.
- Recover from SQL errors and from answers that fail verification.
- Return the SQL, a grounding verdict and a confidence signal with every answer.
- Persist every run and step. Expose the agent through a CLI and an HTTP API.
- Evaluate execution accuracy, valid-SQL rate, correct refusals and grounded-answer rate per
  language, and compare agent mode with a single-shot baseline.

Non-functional

- No agent framework. Plain HTTP to any OpenAI-compatible endpoint.
- Runs on a CPU-only laptop with 8 GB of RAM. No GPU and no Docker required.
- Works from VS Code on Windows and on Linux.
- Every claim in the documentation is backed by a test or a command that can be re-run.

## Architecture

See `ARCHITECTURE.md`. A loop calls a model through an LLM client, executes tool calls through a
registry, runs SQL through a guard and a read-only engine, and verifies the final answer against the
retrieved rows. An evaluation harness drives the same code over a benchmark.

## Technology choices

| Choice | Why | Trade-off |
|---|---|---|
| httpx against `/chat/completions` | One client for Groq, Ollama and OpenAI, testable with a mock transport and a stub server. | No streaming and no provider-specific structured-output modes. |
| sqlglot for the guard | A real SQL parser gives an AST to inspect instead of fragile string matching. | It is not SQLite's own parser, so the engine layers back it up. |
| sqlite3 with `mode=ro`, authorizer and progress handler | Enforces read-only access and time limits inside the engine itself. | SQLite specific. |
| pydantic for tool arguments | Validation errors become readable feedback and tool specs come from the same models. | Schemas avoid `Optional` because some providers reject `anyOf`. |
| FastAPI | Typed request and response models and generated docs. | A static API key only, see limitations. |
| YAML glossary | Domain knowledge a non-programmer can edit. | Hand-maintained, no automatic discovery. |
| SQLite for traces | Zero setup and one file. | Single writer, guarded by a lock. |

## Folder structure

```
src/analystagent/
  config.py            settings and budgets
  text.py              digit and Arabic normalisation, language detection
  database.py          database handle with schema
  db/                  schema introspection, SQL guard, read-only connection
  semantic/            glossary loader and the bundled glossary
  grounding.py         answer verifier
  llm/                 client protocol, OpenAI-compatible client, scripted client for tests
  agent/               prompts, tools, decision loop
  store.py             SQLite traces
  api.py               FastAPI application
  eval/                dataset, scoring, harness, metrics, report, baselines, self-check
  cli.py               command line interface
data/                  Chinook database and its license
benchmarks/            questions.json with golden SQL
tests/                 unit, integration and end-to-end tests
docs/                  this folder
```

## Data design

`runs(id, created_at, finished_at, question, language, mode, provider, model, status, stop_reason,
answer, cited_sql, grounded, confidence, steps, prompt_tokens, completion_tokens, security_blocks,
query_errors, grounding_rejections, wall_s, meta_json)`

`steps(id, run_id, idx, kind, tool, args_json, result_preview, ok, security_block, latency_s,
prompt_tokens, completion_tokens)` with an index on `(run_id, idx)`.

## API design

| Endpoint | Purpose |
|---|---|
| `POST /v1/ask` | Answer a question. Body: `question` (1 to 1000 characters), optional `language`, `mode` |
| `GET /v1/runs/{id}` | Stored run and steps |
| `GET /healthz` | Liveness, table count, whether an LLM and an API key are configured |

Errors: 401 bad key, 404 unknown run, 422 invalid body, 503 no model configured, 500 with a generic
message.

## Roadmap and milestones

| Milestone | Content | State |
|---|---|---|
| M1 | Database layer: schema, guard, read-only engine | done |
| M2 | Agent: tools, loop, glossary, verifier | done |
| M3 | Service: store, API, CLI | done |
| M4 | Evaluation: benchmark, scoring, harness, baselines, self-check | done |
| M5 | Live evaluation with a real model | **open, needs an API key** |
| M6 | Windows run and CI run | **open** |

Complexity was highest in the guard (parser behaviour and bypass attempts) and the verifier (number
formats in two languages). Main risks: parser differences between sqlglot and SQLite, models that
call tools unreliably, and benchmark contamination.

## Testing strategy

- Unit tests per module, with adversarial cases for the guard, engine and verifier.
- Loop tests with a scripted model for every stop reason and recovery path.
- API tests with the FastAPI test client, including authentication and concurrency.
- An end-to-end test that runs the CLI evaluation against a stub OpenAI-compatible server over real
  sockets.
- A harness self-check that must pass before any model result is trusted.

## Deployment plan

Local: `pip install -e .`, `python -m analystagent serve`. Behind a gateway for any wider exposure.
A container image is not provided because no Docker daemon was available to build and test one.
