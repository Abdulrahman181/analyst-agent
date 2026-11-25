# Threat model

## What this protects

The goals are an unmodified database, answers whose numbers trace to real query results, and an
evaluation number that can be trusted. The adversary is a model that is careless, steered by text in
the question or in the data, or asked to do something destructive. It is **not** a determined
attacker with a SQLite or parser exploit.

## Controls and how each is tested

| Threat | Control | Test |
|---|---|---|
| Destructive statements (`DELETE`, `DROP`, `UPDATE`, `INSERT`, `CREATE`, `ALTER`) | Guard accepts only one `SELECT`. Engine opened read-only. | `test_guard.py::test_rejected_queries`, `test_connection.py::test_engine_refuses_writes_even_without_the_guard`, `test_connection.py::test_database_file_is_never_modified` |
| Stacked statements | Single statement only, parsed with a real parser | `test_guard.py::test_rejected_queries` (`SELECT 1; DROP TABLE ...`) |
| `PRAGMA`, `ATTACH`, `VACUUM`, transactions | Rejected as non-SELECT. Denied again by the authorizer. | `test_guard.py`, `test_connection.py` |
| Reading system tables | Unknown-table check, plus authorizer denies `sqlite_*` | `test_guard.py::test_cte_cannot_hide_a_forbidden_table`, `test_subquery_cannot_hide_a_forbidden_table`, `test_connection.py::test_system_tables_are_not_readable` |
| Table-valued functions (`pragma_table_info`, `json_each`) | Rejected in `FROM` | `test_guard.py::test_rejected_queries` |
| Dangerous functions (`load_extension`, `randomblob`, `zeroblob`, `random`) | Denied by name in the guard, found inside subqueries, and denied by the authorizer | `test_guard.py::test_forbidden_function_inside_subquery_is_found`, `test_connection.py::test_denied_functions_are_refused_by_the_engine` |
| Schema-qualified tables (`main.x`, `temp.x`) | Rejected | `test_guard.py::test_rejected_queries` |
| Runaway queries (infinite recursion) | Time limit from a progress handler. The connection stays usable. | `test_connection.py::test_infinite_recursion_hits_the_time_limit`, `test_connection_is_usable_after_a_timeout` |
| Huge results | Row cap at fetch time, model sees at most 30 rows | `test_connection.py::test_cartesian_product_is_bounded_by_row_cap`, `test_tools.py::test_large_results_are_truncated_for_the_model` |
| Memory blow-up from huge values | `SQLITE_LIMIT_LENGTH` of 1 MB | `test_connection.py::test_oversized_values_are_refused` |
| Wrong results from double-quoted unknown names | Unknown-column check with a hint to use single quotes | `test_guard.py::test_double_quoted_string_literal_is_not_silently_accepted` |
| Invented numbers in the answer | Verifier rejects ungrounded figures, with Arabic-Indic digit support | `test_grounding.py`, `test_loop.py::test_persistently_ungrounded_answer_is_flagged_not_hidden` |
| Citing SQL that never ran | Cited SQL must match a successful execution in this run | `test_tools.py::test_final_answer_rejects_unexecuted_sql` |
| Prompt injection inside data | Results are labelled as data, and an injected number still fails verification | `test_loop.py::test_injection_in_data_cannot_force_an_answer`, `test_grounding.py::test_injected_number_in_a_cell_cannot_justify_a_claim` |
| Prompt injection in the question | The guard and engine do not depend on the model complying | `test_loop.py::test_destructive_statements_are_blocked_and_end_the_run`, `test_api.py::test_destructive_question_never_modifies_the_database` |
| Repeated attack attempts | Three security blocks end the run | `test_loop.py::test_destructive_statements_are_blocked_and_end_the_run` |
| Endless agent loops | Step, token and wall-clock budgets, stall detection | `test_loop.py` |
| API key in error messages or traces | Key only placed in the request header. Error text limited to status and a body excerpt. | `test_llm_client.py::test_client_errors_are_not_retried_and_do_not_leak_key`, `test_http_end_to_end.py` |
| Leaking internals through the API | Generic 500 body, details only in server logs | `test_api.py::test_internal_errors_do_not_leak_details` |
| Unauthenticated API use | Optional `X-API-Key`, compared in constant time | `test_api.py::test_api_key_is_enforced_when_configured` |
| SQL injection into the trace store | Parameterised statements | `test_store.py::test_sql_injection_in_question_is_stored_as_text` |

## Known limitations

These are real and not hidden by the tests.

1. **Two parsers.** The guard parses with sqlglot, SQLite parses with its own grammar. Unusual input
   might be read differently by each. The read-only connection, the authorizer and the limits are
   the backstop, and they do not depend on the guard.
2. **Confidentiality is not addressed.** The agent can read every table in the database. A
   deployment with sensitive columns needs a table and column allow-list or a restricted view, and
   it should not send those rows to a hosted model.
3. **Data leaves the machine.** Query results shown to the model are sent to the provider. Do not
   point this at private data with a hosted provider without permission.
4. **Numbers only.** The verifier does not check numbers written in words, names, or whether a
   correct number is attached to the right label.
5. **Authentication is minimal.** A single shared key, no per-user identity, no rate limit, no
   request quota. Use a gateway.
6. **Resource limits are per query.** Many concurrent requests are not throttled.
7. **Windows paths were not executed.** The read-only URI is built with `Path.as_uri()` and the
   tests cover spaces and Arabic characters on Linux. Windows itself, and the CI job for it, had not
   been run when this was written.
8. **Model-side risks are out of scope.** Provider data handling and prompt content are the
   operator's responsibility.
