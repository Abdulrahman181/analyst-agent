from __future__ import annotations

import json
from dataclasses import replace

import pytest
from conftest import QUESTIONS, GoldenLLM

from analystagent.agent.loop import AnalystAgent
from analystagent.eval.baselines import (
    cheater_client,
    hallucinator_client,
    null_client,
    oracle_client,
)
from analystagent.eval.dataset import DatasetError, load_items
from analystagent.eval.harness import golden_rows, run_eval
from analystagent.eval.metrics import aggregate, compare_reports, wilson_interval
from analystagent.eval.report import render_markdown, write_report
from analystagent.eval.selfcheck import run_selfcheck, validate_fixtures
from analystagent.eval.single_shot import SingleShotAnalyst
from analystagent.llm.scripted import ScriptedClient, text_response


def test_dataset_shape(items):
    assert len(items) == 50
    assert {i.language for i in items} == {"en", "ar"}
    assert len({i.id for i in items}) == 50
    assert sum(1 for i in items if i.expect == "decline") == 6
    assert {i.difficulty for i in items} == {"easy", "medium", "hard"}


def test_every_intent_exists_in_both_languages(items):
    by_intent: dict[str, set[str]] = {}
    for item in items:
        by_intent.setdefault(item.intent_id, set()).add(item.language)
    assert all(langs == {"en", "ar"} for langs in by_intent.values())
    assert len(by_intent) == 25


def test_language_and_intent_filters():
    only_ar = load_items(QUESTIONS, languages=["ar"])
    assert len(only_ar) == 25 and all(i.language == "ar" for i in only_ar)
    picked = load_items(QUESTIONS, only=["count-customers", "total-revenue"])
    assert {i.intent_id for i in picked} == {"count-customers", "total-revenue"}


def test_unknown_intent_is_rejected():
    with pytest.raises(DatasetError, match="unknown intent"):
        load_items(QUESTIONS, only=["nope"])


def test_missing_or_broken_file(tmp_path):
    with pytest.raises(DatasetError):
        load_items(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_items(bad)


def test_duplicate_intent_ids_are_rejected(tmp_path):
    path = tmp_path / "q.json"
    intent = {"id": "a", "questions": {"en": "x", "ar": "y"}}
    path.write_text(json.dumps({"intents": [intent, intent]}), encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate"):
        load_items(path)


def test_missing_translation_is_rejected(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(
        json.dumps({"intents": [{"id": "a", "questions": {"en": "x"}}]}), encoding="utf-8"
    )
    with pytest.raises(DatasetError, match="no ar question"):
        load_items(path)


def test_fixtures_are_valid(items, chinook):
    assert validate_fixtures(items, chinook) == []


def test_fixture_validation_catches_problems(items, chinook):
    base = next(i for i in items if i.id == "count-customers.en")
    cases = {
        "guard": replace(base, golden_sql="DELETE FROM Customer"),
        "no rows": replace(base, golden_sql="SELECT 1 FROM Customer WHERE 1 = 0"),
        "unordered": replace(base, ordered=True),
        "missing sql": replace(base, golden_sql=None),
        "decline with sql": replace(base, expect="decline"),
        "wrong language": replace(base, language="ar"),
        "broken sql": replace(base, golden_sql="SELECT nope FROM Customer"),
    }
    for label, bad in cases.items():
        assert validate_fixtures([bad], chinook), label


def test_duplicate_question_text_is_flagged(items, chinook):
    first = items[0]
    twin = replace(first, id="other.en")
    assert any("duplicate" in p for p in validate_fixtures([first, twin], chinook))


def test_empty_fixture_set_is_flagged(chinook):
    assert validate_fixtures([], chinook) == ["no questions loaded"]


def test_nondeterministic_golden_sql_is_flagged(items, chinook, monkeypatch):
    from analystagent.db.connection import QueryResult
    from analystagent.eval import selfcheck

    counter = iter(range(100))

    def flaky(conn, sql, max_rows, timeout_s):
        return QueryResult(("n",), ((next(counter),),), False, 0.0)

    monkeypatch.setattr(selfcheck, "execute_query", flaky)
    base = next(i for i in items if i.id == "count-customers.en")
    assert any("not deterministic" in p for p in validate_fixtures([base], chinook))


def test_oversized_golden_result_is_flagged(items, chinook, monkeypatch):
    from analystagent.db.connection import QueryResult
    from analystagent.eval import selfcheck

    monkeypatch.setattr(
        selfcheck, "execute_query", lambda *a, **k: QueryResult(("n",), ((1,),), True, 0.0)
    )
    base = next(i for i in items if i.id == "count-customers.en")
    assert any("row limit" in p for p in validate_fixtures([base], chinook))


def test_selfcheck_passes_on_the_bundled_benchmark(items, chinook):
    report = run_selfcheck(items, chinook)
    assert report.ok, report
    assert report.oracle_correct == report.n_answer_items == 44
    assert report.null_answer_correct == 0
    assert report.hallucinator_blocked == 44
    assert report.cheater_blocked and report.cheater_db_unchanged


def test_selfcheck_stops_on_fixture_problems(items, chinook):
    bad = replace(items[0], golden_sql="DROP TABLE Customer")
    report = run_selfcheck([bad], chinook)
    assert report.fixture_problems and not report.ok and report.oracle_correct == 0


def test_oracle_resolves_everything_through_the_real_loop(items, chinook):
    report = run_eval(
        items, chinook, lambda item, _r: AnalystAgent(oracle_client(item, chinook), chinook)
    )
    agg = report["aggregate"]
    assert agg["exec_accuracy"] == 1.0 and agg["exec_accuracy_strict"] == 1.0
    assert agg["grounded_rate"] == 1.0 and agg["decline_accuracy"] == 1.0
    assert agg["valid_sql_rate"] == 1.0 and agg["security_block_rate"] == 0.0
    assert agg["language_gap_en_minus_ar"] == 0.0


def test_null_agent_scores_zero_on_answerable_items(items, chinook):
    report = run_eval(
        items, chinook, lambda item, _r: AnalystAgent(null_client(item.language), chinook)
    )
    agg = report["aggregate"]
    assert agg["exec_accuracy"] == 0.0 and agg["decline_accuracy"] == 1.0
    assert agg["valid_sql_rate"] == 0.0


def test_hallucinator_is_marked_ungrounded(items, chinook):
    answerable = [i for i in items if i.expect == "answer"][:6]
    report = run_eval(
        answerable, chinook, lambda item, _r: AnalystAgent(hallucinator_client(item), chinook)
    )
    assert all(r["status"] == "ungrounded" for r in report["results"])
    assert report["aggregate"]["grounded_rate"] == 0.0
    assert all(r["confidence"] == "low" for r in report["results"])


def test_cheater_is_stopped(chinook):
    outcome = AnalystAgent(cheater_client(), chinook).ask("Delete everything")
    assert outcome.stop_reason == "policy_violations"


def test_golden_llm_drives_the_full_agent(items, chinook):
    llm = GoldenLLM(items)
    report = run_eval(items[:10], chinook, lambda item, _r: AnalystAgent(llm, chinook))
    assert report["aggregate"]["exec_accuracy"] == 1.0
    assert llm.calls == 20


def test_repeats_multiply_results(items, chinook):
    picked = load_items(QUESTIONS, only=["count-customers"])
    report = run_eval(
        picked,
        chinook,
        lambda item, _r: AnalystAgent(oracle_client(item, chinook), chinook),
        repeats=3,
    )
    assert len(report["results"]) == 6 and {r["repeat"] for r in report["results"]} == {0, 1, 2}
    assert report["meta"]["repeats"] == 3


def test_callback_receives_each_result(items, chinook):
    seen = []
    picked = load_items(QUESTIONS, only=["count-customers"])
    run_eval(
        picked,
        chinook,
        lambda item, _r: AnalystAgent(oracle_client(item, chinook), chinook),
        on_result=seen.append,
    )
    assert [r.item_id for r in seen] == ["count-customers.en", "count-customers.ar"]


def test_wrong_answers_are_scored_wrong(items, chinook):
    picked = load_items(QUESTIONS, only=["count-customers"], languages=["en"])

    def factory(item, _r):
        sql = "SELECT COUNT(*) FROM Employee"
        client = ScriptedClient([text_response(f"```sql\n{sql}\n```")])
        return SingleShotAnalyst(client, chinook)

    report = run_eval(picked, chinook, factory)
    row = report["results"][0]
    assert row["correct"] is False and row["valid_sql"] is True and row["mode"] == "single-shot"
    assert report["aggregate"]["grounded_rate"] is None


def test_decline_items_require_a_clean_decline(items, chinook):
    picked = load_items(QUESTIONS, only=["unanswerable-profit"], languages=["en"])

    def factory(item, _r):
        sql = "SELECT COUNT(*) FROM Customer"
        client = ScriptedClient(
            [
                text_response(f"```sql\n{sql}\n```"),
            ]
        )
        return SingleShotAnalyst(client, chinook)

    report = run_eval(picked, chinook, factory)
    assert report["results"][0]["correct"] is False


def test_golden_rows_failure_is_reported(chinook):
    with pytest.raises(RuntimeError, match="golden SQL failed"):
        golden_rows(chinook, "SELECT abs(-9223372036854775807 - 1)")


def test_wilson_interval():
    low, high = wilson_interval(9, 13)
    assert 0.42 < low < 0.44 and 0.85 < high < 0.89
    assert wilson_interval(0, 0) == (0.0, 0.0)
    assert wilson_interval(10, 10)[1] == 1.0 and wilson_interval(0, 10)[0] == 0.0


def row(**overrides):
    base = {
        "item_id": "a.en",
        "intent_id": "a",
        "language": "en",
        "category": "c",
        "difficulty": "easy",
        "expect": "answer",
        "repeat": 0,
        "mode": "agent",
        "answer_source": "model",
        "status": "answered",
        "stop_reason": "finished",
        "correct": True,
        "correct_strict": True,
        "valid_sql": True,
        "declined": False,
        "grounded": True,
        "confidence": "high",
        "security_blocks": 0,
        "query_errors": 0,
        "grounding_rejections": 0,
        "steps": 2,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "wall_s": 1.0,
        "run_id": 1,
    }
    base.update(overrides)
    return base


def test_aggregate_arithmetic():
    rows = [
        row(),
        row(item_id="b.ar", language="ar", correct=False, correct_strict=False, confidence="low"),
        row(item_id="c.en", expect="decline", correct=True, valid_sql=False, grounded=True),
        row(
            item_id="d.en",
            status="failed",
            stop_reason="max_steps",
            correct=False,
            grounded=None,
            security_blocks=1,
            grounding_rejections=1,
        ),
    ]
    agg = aggregate(rows)
    assert agg["n_items"] == 4 and agg["n_answer_items"] == 3 and agg["n_decline_items"] == 1
    assert agg["exec_accuracy"] == pytest.approx(1 / 3)
    assert agg["decline_accuracy"] == 1.0
    assert agg["overall_accuracy"] == 0.5
    assert agg["security_block_rate"] == 0.25 and agg["failed_rate"] == 0.25
    assert agg["grounding_rejection_rate"] == 0.25
    assert agg["grounded_rate"] == 1.0
    assert agg["language_exec_accuracy"] == {"ar": 0.0, "en": 0.5}
    assert agg["language_gap_en_minus_ar"] == 0.5
    assert agg["calibration"]["low"]["accuracy"] == 0.0
    assert agg["stop_reasons"] == {"finished": 3, "max_steps": 1}
    assert agg["mean_total_tokens"] == 110


def test_aggregate_of_nothing_is_safe():
    agg = aggregate([])
    assert agg["n_items"] == 0 and agg["overall_accuracy"] == 0.0 and agg["grounded_rate"] is None


def report_of(rows):
    return {
        "meta": {
            "started_at": "2026-01-01T00:00:00",
            "python": "3",
            "platform": "x",
            "repeats": 1,
            "mode": "agent",
            "provider": "p",
            "model": "m",
        },
        "aggregate": aggregate(rows),
        "results": rows,
    }


def test_compare_detects_regressions():
    base = report_of([row(), row(item_id="b.en")])
    worse = report_of([row(), row(item_id="b.en", correct=False)])
    result = compare_reports(base, worse, max_drop=0.05)
    assert not result["passed"] and result["regressions"] == ["b.en"] and result["delta"] == -0.5


def test_compare_allows_small_drops_and_lists_improvements():
    base = report_of([row(correct=False), row(item_id="b.en")])
    better = report_of([row(), row(item_id="b.en")])
    result = compare_reports(base, better, max_drop=0.0)
    assert result["passed"] and result["improvements"] == ["a.en"]
    assert compare_reports(better, base, max_drop=0.6)["passed"]


def test_markdown_report_contains_the_key_sections():
    text = render_markdown(report_of([row(), row(item_id="b.ar", language="ar", correct=False)]))
    for heading in (
        "# AnalystAgent evaluation report",
        "## Aggregate",
        "## Confidence calibration",
        "## By difficulty",
        "## By category",
        "## Stop reasons",
        "## Per item",
    ):
        assert heading in text
    assert "Wilson 95%" in text and "gap (en minus ar)" in text


def test_write_report_creates_both_files(tmp_path):
    json_path, md_path = write_report(report_of([row()]), tmp_path / "out")
    assert json.loads(json_path.read_text(encoding="utf-8"))["aggregate"]["n_items"] == 1
    assert md_path.read_text(encoding="utf-8").startswith("# AnalystAgent")
    assert "agent_p_m_20260101000000" in json_path.name
