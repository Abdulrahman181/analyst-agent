from __future__ import annotations

from analystagent.database import Database
from analystagent.db.connection import execute_query
from analystagent.eval.dataset import EvalItem
from analystagent.llm.scripted import ScriptedClient, tool_call_response
from analystagent.render import render_rows


def oracle_client(item: EvalItem, db: Database) -> ScriptedClient:
    if item.expect == "decline" or not item.golden_sql:
        return null_client(item.language)
    conn = db.connect()
    try:
        golden = execute_query(conn, item.golden_sql, 1000, 10.0)
    finally:
        conn.close()
    answer = render_rows(golden.columns, golden.rows, item.language)
    return ScriptedClient(
        [
            tool_call_response("run_sql", {"sql": item.golden_sql}),
            tool_call_response("final_answer", {"answer": answer, "sql": item.golden_sql}),
        ]
    )


def null_client(language: str = "en") -> ScriptedClient:
    message = (
        "لا يمكنني الإجابة على هذا السؤال."
        if language == "ar"
        else "I cannot answer this question."
    )
    return ScriptedClient([tool_call_response("final_answer", {"answer": message, "sql": ""})])


def cheater_client() -> ScriptedClient:
    return ScriptedClient(
        [
            tool_call_response("run_sql", {"sql": "DELETE FROM Customer"}),
            tool_call_response("run_sql", {"sql": "DROP TABLE Customer"}),
            tool_call_response("run_sql", {"sql": "SELECT name FROM sqlite_master"}),
            tool_call_response("final_answer", {"answer": "done", "sql": ""}),
        ]
    )


def hallucinator_client(item: EvalItem) -> ScriptedClient:
    sql = item.golden_sql or "SELECT COUNT(*) FROM Customer"
    wrong = {"answer": "The figure is 987654.321.", "sql": sql}
    return ScriptedClient(
        [
            tool_call_response("run_sql", {"sql": sql}),
            tool_call_response("final_answer", wrong, call_id="call_1"),
            tool_call_response("final_answer", wrong, call_id="call_2"),
            tool_call_response("final_answer", wrong, call_id="call_3"),
        ]
    )
