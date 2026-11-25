from __future__ import annotations

import pytest

from analystagent.eval.single_shot import SingleShotAnalyst, extract_sql
from analystagent.llm.base import LLMError
from analystagent.llm.scripted import ScriptedClient, text_response
from analystagent.store import RunStore


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("Here:\n```SQL\nSELECT 2;\n```\nthanks", "SELECT 2;"),
        ("```\nSELECT 3\n```", "SELECT 3"),
        ("SELECT 4", "SELECT 4"),
        ("with a as (select 1) select * from a", "with a as (select 1) select * from a"),
        ("DECLINE", None),
        ("I do not know", None),
        ("", None),
    ],
)
def test_extract_sql(text, expected):
    assert extract_sql(text) == expected


def ask(chinook, reply, question="How many customers are there?", store=None):
    return SingleShotAnalyst(ScriptedClient([reply]), chinook, store=store).ask(question)


def test_successful_single_shot(chinook):
    outcome = ask(chinook, text_response("```sql\nSELECT COUNT(*) FROM Customer\n```"))
    assert outcome.status == "answered" and outcome.stop_reason == "finished"
    assert outcome.result.rows == ((59,),)
    assert outcome.answer_source == "template" and outcome.grounded is None
    assert "59" in outcome.answer and outcome.mode == "single-shot"
    assert outcome.steps == 1 and outcome.confidence == "high"


def test_arabic_template_answer(chinook):
    outcome = ask(
        chinook, text_response("```sql\nSELECT COUNT(*) FROM Customer\n```"), "كم عدد العملاء؟"
    )
    assert outcome.answer.startswith("النتيجة")


def test_decline(chinook):
    outcome = ask(chinook, text_response("DECLINE"), "What was the profit?")
    assert outcome.status == "answered" and outcome.declined and outcome.stop_reason == "declined"


def test_guard_block_is_a_failure_with_security_flag(chinook):
    outcome = ask(chinook, text_response("```sql\nDELETE FROM Customer\n```"))
    assert outcome.status == "failed" and outcome.stop_reason == "guard_blocked"
    assert outcome.security_blocks == 1 and outcome.result is None


def test_unknown_column_is_a_failure_without_security_flag(chinook):
    outcome = ask(chinook, text_response("```sql\nSELECT nope FROM Track\n```"))
    assert outcome.status == "failed" and outcome.security_blocks == 0


def test_engine_error_is_a_failure(chinook):
    outcome = ask(chinook, text_response("```sql\nSELECT abs(-9223372036854775807 - 1)\n```"))
    assert outcome.status == "failed" and outcome.stop_reason == "sql_error"
    assert outcome.query_errors == 1


def test_no_sql_in_reply(chinook):
    outcome = ask(chinook, text_response("There are about fifty customers."))
    assert outcome.status == "failed" and outcome.stop_reason == "no_sql"


def test_llm_error(chinook):
    outcome = SingleShotAnalyst(ScriptedClient([LLMError("down")]), chinook).ask("q")
    assert outcome.status == "failed" and outcome.stop_reason == "llm_error"


def test_single_shot_makes_exactly_one_call_and_sends_no_tools(chinook):
    client = ScriptedClient([text_response("```sql\nSELECT 1\n```")])
    SingleShotAnalyst(client, chinook).ask("q")
    assert len(client.calls) == 1 and client.tool_specs == [[]]
    assert "Database schema" in client.calls[0][1]["content"]


def test_run_is_stored(chinook):
    store = RunStore()
    outcome = ask(chinook, text_response("```sql\nSELECT COUNT(*) FROM Customer\n```"), store=store)
    record = store.get_run(outcome.run_id)
    assert record["mode"] == "single-shot" and record["status"] == "answered"
    assert record["grounded"] is None
