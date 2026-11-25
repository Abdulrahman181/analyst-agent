# Evaluation protocol

## What is measured

| Metric | Definition |
|---|---|
| Execution accuracy | Share of answerable items where the result of the SQL the run cited matches the golden result. Extra or reordered columns are tolerated. Headline number. |
| Execution accuracy, strict | Same, but the column count must also match. |
| Valid-SQL rate | Share of answerable items where the cited SQL executed successfully. |
| Correct decline rate | Share of must-decline items (unanswerable or unsafe requests) where the run ended with a refusal and no SQL. |
| Overall accuracy | Correct items over all items, answerable and must-decline. |
| Grounded-answer rate | Share of model-written answers that passed verification. Agent mode only. |
| Grounding rejection rate | Share of agent runs where at least one final answer was rejected. |
| Security block rate | Share of runs with a guard security block. Should be close to 0 on benign questions. |
| Language gap | Execution accuracy in English minus Arabic. |
| Calibration | Accuracy per confidence band (`high`, `medium`, `low`). |
| Turns, tokens, wall time | Cost per item. Tokens come from the provider's `usage` field and are zero if the provider omits it. |

The report also breaks accuracy down by category and difficulty, lists stop reasons, and gives a
Wilson 95% interval for execution accuracy.

## How results are compared with the golden answer

Both results are normalised cell by cell: integers and integer-valued floats are equal, other floats
are rounded to two decimals, and strings that look like numbers are compared as numbers. Rows are
compared as a multiset, or as a sequence when the intent is marked `ordered`. Strict comparison
needs the same columns. The tolerant comparison finds an assignment of golden columns to candidate
columns that makes the projected rows equal.

## How to run

```
python -m analystagent selfcheck
python -m analystagent eval --mode single-shot
python -m analystagent eval --mode agent
python -m analystagent eval --mode agent --repeats 3
python -m analystagent compare reports/old.json reports/new.json --max-drop 0.05
```

`selfcheck` needs no model. `eval` needs `LLM_MODEL` and, for hosted providers, an API key.
Run `single-shot` and `agent` with the same model to measure what the loop adds.

## Reading the numbers honestly

- There are 44 answerable items from 22 intents, each asked twice (once per language). The two
  languages of an intent share a golden query, so they are not independent. A result of 36 of 44 is
  82% with a Wilson interval of about 68% to 90%. Differences of a few points between two models are
  noise.
- Repeats of the same item are not independent either. The reported interval treats items as
  independent and is therefore too narrow. Use it as a rough guide.
- The language gap is computed on paired questions, but with 22 pairs a gap of a few points is
  noise.
- The database is Chinook, a small and well-known sample. Models may have seen it and similar
  questions in training. Contamination cannot be ruled out, and results do not transfer to a messy
  production schema.
- Temperature defaults to 0, but hosted models are still not perfectly deterministic.
- A grounded answer is not necessarily a correct one. Read execution accuracy and the grounded rate
  together.
- The single-shot baseline answers with a template, so only its SQL is scored.
- Confidence calibration with a few dozen items is a rough check, not a calibration curve.

## Regression gate

`compare` fails when the candidate's overall accuracy drops by more than `--max-drop` and lists the
items that got worse. It does not test significance. With 50 items, treat a failed gate as a prompt
to look at the traces.

## What the self-check proves

- Every golden query passes the guard, runs, returns rows, is deterministic, and ranking questions
  declare an order.
- A scripted agent that runs the golden SQL and quotes the rows through the real tools resolves
  every answerable item, and refuses every must-decline item, so the guard, tools, loop, verifier
  and scoring work together on all 50 items.
- A scripted single-shot run with the golden SQL scores 44 of 44.
- A null agent that always refuses scores 0 of 44 on answerable items.
- An agent that quotes an invented number is marked ungrounded for every answerable item.
- An agent that tries `DELETE`, `DROP` and a read of `sqlite_master` is stopped after three blocks,
  and the database file hash is unchanged.

It does **not** say anything about how well a real model performs.

## Adding a question

Add an intent to `benchmarks/questions.json` with `id`, `category`, `difficulty`, both questions,
`golden_sql` and `ordered`, or `"expect": "decline"` without SQL. Run `selfcheck`. It rejects golden
SQL that the guard refuses, that returns no rows or a non-deterministic result, or that declares an
order without `ORDER BY`. Pin the output columns in the wording when several are reasonable, and give
ranking queries a tie-breaker.

## Results

No model results are published here. Run `eval` and paste the generated Markdown report into this
section with the date, provider and model.
