# Architecture

## Components

```
              +-------------------+        +--------------------+
 question --> |   AnalystAgent    | <----> |  LLMClient         |  OpenAI-compatible HTTP
              |   (agent/loop.py) |        |  (Groq/Ollama/...) |  or ScriptedClient in tests
              +---------+---------+        +--------------------+
                        |  tool call / result
                        v
              +-------------------+     +---------------------+
              |   ToolRegistry    | --> |  Glossary + schema  |  prompt context
              |   5 tools         |     +---------------------+
              +---------+---------+
                        |
         run_sql        |         final_answer
            +-----------+-----------+
            v                       v
   +-----------------+     +-------------------+
   |  SQL guard      |     |  Grounding        |  numbers vs retrieved rows
   |  (sqlglot AST)  |     |  verifier         |
   +--------+--------+     +-------------------+
            v
   +-----------------+     +-------------------+
   |  Read-only      | --> |  RunStore (SQLite)|
   |  SQLite engine  |     +-------------------+
   +-----------------+
```

The model never writes to the database and never sees more than the first 30 rows of a result. The
verifier only trusts rows the model was shown.

## Decision loop

Each iteration:

1. Check the budgets (steps, tokens, wall clock). Stop if one is exhausted.
2. Call the model with the conversation and the tool specifications.
3. Recoverable errors:
   - A malformed tool call from the provider (`tool_use_failed`) or a reply with no tool call adds a
     short corrective message. Three in a row end the run with `no_tool_calls`.
   - Any other client error ends the run with `llm_error`.
4. Execute at most three tool calls from the reply. Each goes through argument validation and the
   tool handler. Failures are returned to the model as text, not raised.
5. Track repeated identical calls. The second repeat appends a hint and the third stops the run
   with `stalled`.
6. Stop on a verified `final_answer`, on three security blocks, or continue.

Stop reasons: `finished`, `ungrounded`, `max_steps`, `max_tokens`, `timeout`, `stalled`,
`no_tool_calls`, `llm_error`, `policy_violations`.

Run status is one of `answered` (verified answer, or a clean refusal with no SQL), `ungrounded`
(an answer was returned but could not be verified) or `failed` (no answer).

## Tools

| Tool | Purpose |
|---|---|
| `list_tables` | Tables with columns, keys and row counts |
| `describe_table` | One table in detail |
| `sample_rows` | Up to five example rows |
| `run_sql` | One guarded read-only `SELECT` |
| `final_answer(answer, sql)` | Finish. `sql` must be a query that ran successfully in this run, or empty for a refusal |

Tool specifications are generated from pydantic models with titles stripped to save tokens. Query
results are prefixed with a line stating that they are data, not instructions.

## Layers between a model and the data

| Layer | File | Stops |
|---|---|---|
| Statement check | `db/guard.py` | Anything that is not one read-only `SELECT`, table-valued functions, qualified tables, unknown tables and columns, denied functions |
| Connection | `db/connection.py` | Writes (`mode=ro`, `query_only`), reads outside the allowed tables and denied functions (authorizer), oversized values, expression depth |
| Execution | `db/connection.py` | Runaway queries (progress handler time limit) and large results (row cap at fetch) |
| Answer | `grounding.py`, `agent/tools.py` | Numbers that were not retrieved, citing SQL that did not run |

The guard produces a security flag for attack-shaped input (writes, probes of system tables, denied
functions). Ordinary mistakes such as an unknown column are recoverable and do not count towards the
security limit.

## Semantic layer

`semantic/glossary.yaml` holds notes and entries. An entry has an id, a kind (metric, entity,
dimension or value), English and Arabic terms and a meaning. For a question, the loader normalises
Arabic (diacritics, alef and ya variants, the definite article and common prefixes) and matches
terms on word boundaries. Matching meanings are added to the user message, so the model sees for
example that "ألمانيا" is the stored value `'Germany'` and that revenue is `SUM(Invoice.Total)`.

## Answer verification

`final_answer` is accepted only when:

1. the cited SQL, compared after normalising formatting and identifier case, matches a query that
   executed successfully in this run (or the SQL is empty), and
2. every number in the answer is grounded.

A number is grounded when, after normalising Arabic-Indic digits and separators, it equals a value in
a result the model saw (first 30 rows of each), equals a row count, appears in the question, or is
such a value rounded to the number of decimals written. Dates must appear in a result or in the
question. List markers, result column labels, and cell text quoted verbatim are ignored.

A rejection returns the offending numbers to the model. After two rejections the third ends the run
with the answer marked ungrounded, so a failure is visible instead of hidden.

## Confidence

`high`: a verified answer, no SQL errors, no blocks, no rejections, and a result that was not
truncated. `medium`: verified, but the run had errors, blocks or rejections, or the result was
truncated. `low`: refusals, unverified answers and failed runs. This describes how clean the process
was. It is not a probability of correctness; the evaluation report shows accuracy per band.

## Stored data

See the schema in `PLAN.md`. Large arguments and results are clipped when stored (2000 and 500
characters). Nothing from the environment is stored. Writes are serialised with a lock so the API
can serve concurrent requests.

## Modes

`agent` is the loop above. `single-shot` is the evaluation baseline: one model call with the schema
and glossary hints, no tools, one attempt, SQL extracted from a fenced block (or `DECLINE`), run
through the same guard and engine. Its answer text is a template over the returned rows, so it has no
model-written numbers to verify.
