from __future__ import annotations

import json

import pytest
from conftest import CHINOOK, QUESTIONS, GoldenLLM, final, run_sql
from typer.testing import CliRunner

from analystagent import cli
from analystagent.llm.scripted import ScriptedClient, text_response

runner = CliRunner()
COUNT = "SELECT COUNT(*) FROM Customer"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_PROVIDER", "LLM_MODEL", "GROQ_API_KEY", "LLM_API_KEY", "ANALYST_DB"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "test-model")


def invoke(*args):
    return runner.invoke(cli.app, list(args))


def test_help_lists_every_command():
    result = invoke("--help")
    assert result.exit_code == 0
    for command in ("ask", "serve", "questions", "selfcheck", "eval", "compare", "trace"):
        assert command in result.output


def test_questions_lists_the_benchmark():
    result = invoke("questions", "--questions", str(QUESTIONS), "--language", "ar")
    assert result.exit_code == 0 and "كم عدد العملاء؟" in result.output
    assert len(result.output.strip().splitlines()) == 25


def test_questions_with_a_missing_file():
    result = invoke("questions", "--questions", "nope.json")
    assert result.exit_code == 2 and "could not read questions file" in result.output


def test_selfcheck_passes():
    result = invoke("selfcheck", "--db", str(CHINOOK), "--questions", str(QUESTIONS))
    assert result.exit_code == 0, result.output
    assert "SELFCHECK PASSED" in result.output and "44/44" in result.output


def test_selfcheck_with_a_missing_database():
    result = invoke("selfcheck", "--db", "missing.sqlite", "--questions", str(QUESTIONS))
    assert result.exit_code == 2 and "cannot open database" in result.output


def test_ask_without_configuration_explains_what_is_missing():
    result = invoke("ask", "How many customers?", "--db", str(CHINOOK))
    assert result.exit_code == 2 and "LLM_MODEL" in result.output


def test_ask_requires_a_key_for_hosted_providers(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "m")
    result = invoke("ask", "q", "--db", str(CHINOOK))
    assert result.exit_code == 2 and "GROQ_API_KEY" in result.output


def test_ask_rejects_an_unknown_mode(configured):
    result = invoke("ask", "q", "--mode", "magic", "--db", str(CHINOOK))
    assert result.exit_code == 2


def test_ask_prints_a_grounded_answer(configured, monkeypatch):
    monkeypatch.setattr(
        cli,
        "OpenAICompatClient",
        lambda settings: ScriptedClient([run_sql(COUNT), final("59 customers", COUNT)]),
    )
    result = invoke(
        "ask", "How many customers are there?", "--db", str(CHINOOK), "--store", "runs/t.db"
    )
    assert result.exit_code == 0, result.output
    assert "grounded   : True" in result.output and COUNT in result.output


def test_ask_json_output_with_arabic(configured, monkeypatch):
    monkeypatch.setattr(
        cli,
        "OpenAICompatClient",
        lambda settings: ScriptedClient([run_sql(COUNT), final("يوجد ٥٩ عميلاً", COUNT)]),
    )
    result = invoke(
        "ask", "كم عدد العملاء؟", "--json", "--db", str(CHINOOK), "--store", "runs/t.db"
    )
    payload = json.loads(result.output)
    assert payload["status"] == "answered" and payload["answer"] == "يوجد ٥٩ عميلاً"


def test_ask_single_shot(configured, monkeypatch):
    monkeypatch.setattr(
        cli,
        "OpenAICompatClient",
        lambda settings: ScriptedClient([text_response(f"```sql\n{COUNT}\n```")]),
    )
    result = invoke(
        "ask",
        "How many customers?",
        "--mode",
        "single-shot",
        "--json",
        "--db",
        str(CHINOOK),
        "--store",
        "runs/t.db",
    )
    assert json.loads(result.output)["sql"] == COUNT


def test_ask_then_trace(configured, monkeypatch):
    monkeypatch.setattr(
        cli,
        "OpenAICompatClient",
        lambda settings: ScriptedClient([run_sql("DROP TABLE x"), final("none")]),
    )
    result = invoke("ask", "Drop it", "--json", "--db", str(CHINOOK), "--store", "runs/t.db")
    run_id = json.loads(result.output)["run_id"]
    shown = invoke("trace", str(run_id), "--store", "runs/t.db")
    assert shown.exit_code == 0 and "BLOCKED" in shown.output and "run_sql" in shown.output


def test_trace_errors():
    assert invoke("trace", "1", "--store", "nope.db").exit_code == 2


def test_trace_unknown_run(configured, monkeypatch, tmp_path):
    from analystagent.store import RunStore

    RunStore(tmp_path / "t.db").close()
    assert invoke("trace", "42", "--store", str(tmp_path / "t.db")).exit_code == 2


def test_eval_end_to_end_with_a_golden_model(configured, monkeypatch, items):
    llm = GoldenLLM(items)
    monkeypatch.setattr(cli, "OpenAICompatClient", lambda settings: llm)
    result = invoke(
        "eval",
        "--db",
        str(CHINOOK),
        "--questions",
        str(QUESTIONS),
        "--intent",
        "count-customers",
        "--intent",
        "unanswerable-profit",
        "--out",
        "reports",
        "--store",
        "runs/e.db",
    )
    assert result.exit_code == 0, result.output
    assert "execution accuracy 100.0%" in result.output and "[PASS]" in result.output
    reports = sorted((cli.Path("reports")).glob("*.json"))
    assert len(reports) == 1
    data = json.loads(reports[0].read_text(encoding="utf-8"))
    assert data["meta"]["model"] == "test-model" and data["aggregate"]["n_items"] == 4


def test_eval_validates_options(configured):
    assert invoke("eval", "--mode", "x", "--db", str(CHINOOK)).exit_code == 2
    assert invoke("eval", "--language", "fr", "--db", str(CHINOOK)).exit_code == 2
    unknown = invoke(
        "eval", "--intent", "nope", "--db", str(CHINOOK), "--questions", str(QUESTIONS)
    )
    assert unknown.exit_code == 2 and "unknown intent" in unknown.output


def test_eval_without_configuration():
    assert "LLM_MODEL" in invoke("eval", "--db", str(CHINOOK)).output


def test_compare_gate(tmp_path):
    def report(rate_flags):
        rows = [{"item_id": f"i{n}.en", "correct": flag} for n, flag in enumerate(rate_flags)]
        overall = sum(rate_flags) / len(rate_flags)
        return {"aggregate": {"overall_accuracy": overall}, "results": rows}

    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(report([True, True, True, True])), encoding="utf-8")
    bad.write_text(json.dumps(report([True, False, False, True])), encoding="utf-8")
    passed = invoke("compare", str(good), str(good))
    failed = invoke("compare", str(good), str(bad))
    assert passed.exit_code == 0 and "GATE PASSED" in passed.output
    assert failed.exit_code == 1 and "GATE FAILED" in failed.output and "i1.en" in failed.output
    assert invoke("compare", str(good), str(tmp_path / "missing.json")).exit_code == 2


def test_env_file_is_loaded(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("LLM_PROVIDER=ollama\nLLM_MODEL=from-file\n", encoding="utf-8")
    seen = {}

    def fake_client(settings):
        seen["model"] = settings.model
        return ScriptedClient([final("none")])

    monkeypatch.setattr(cli, "OpenAICompatClient", fake_client)
    result = invoke("ask", "q", "--db", str(CHINOOK), "--store", "runs/t.db")
    assert result.exit_code == 0 and seen["model"] == "from-file"
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)


def test_serve_builds_the_app_and_hands_it_to_uvicorn(configured, monkeypatch):
    import uvicorn
    from fastapi.testclient import TestClient

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw))
    monkeypatch.setenv("ANALYST_API_KEY", "k-123")
    result = invoke("serve", "--port", "9123", "--db", str(CHINOOK), "--store", "runs/s.db")
    assert result.exit_code == 0, result.output
    assert captured["port"] == 9123 and captured["host"] == "127.0.0.1"
    health = TestClient(captured["app"]).get("/healthz").json()
    assert health["auth_required"] is True and health["llm_configured"] is True


def test_serve_starts_even_without_llm_configuration(monkeypatch):
    import uvicorn
    from fastapi.testclient import TestClient

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app))
    result = invoke("serve", "--db", str(CHINOOK), "--store", "runs/s.db")
    assert result.exit_code == 0
    client = TestClient(captured["app"])
    assert client.get("/healthz").json()["llm_configured"] is False
    assert client.post("/v1/ask", json={"question": "q"}).status_code == 503


def test_module_entry_point_runs():
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "analystagent", "--help"], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 0 and "selfcheck" in done.stdout
