from __future__ import annotations

import pytest

from analystagent.config import Budget, ConfigError, LLMSettings, load_env_file
from analystagent.paths import default_db_path, default_questions_path, repo_root


def test_budget_defaults_are_conservative():
    budget = Budget()
    assert budget.max_steps == 10 and budget.max_rows == 200 and budget.sql_timeout_s == 5.0
    assert budget.max_guard_blocks == 3 and budget.max_grounding_rejections == 2


def test_budget_is_immutable():
    with pytest.raises(AttributeError):
        Budget().max_steps = 99


def test_default_paths_point_at_the_bundled_files(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANALYST_DB", raising=False)
    assert default_db_path() == repo_root() / "data" / "chinook.sqlite"
    assert default_db_path().is_file()
    assert default_questions_path() == repo_root() / "benchmarks" / "questions.json"
    assert default_questions_path().is_file()


def test_db_path_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("ANALYST_DB", str(tmp_path / "other.sqlite"))
    assert default_db_path() == tmp_path / "other.sqlite"


def test_working_directory_data_wins(monkeypatch, tmp_path):
    monkeypatch.delenv("ANALYST_DB", raising=False)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "chinook.sqlite").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    assert default_db_path() == tmp_path / "data" / "chinook.sqlite"


def test_settings_defaults_come_from_presets():
    settings = LLMSettings.from_env(provider="groq", model="m", environ={"GROQ_API_KEY": "k"})
    assert settings.base_url == "https://api.groq.com/openai/v1"
    assert settings.temperature == 0.0 and settings.max_retries == 4


def test_missing_model_message_points_to_the_docs():
    with pytest.raises(ConfigError, match="console.groq.com/docs/models"):
        LLMSettings.from_env(environ={"GROQ_API_KEY": "k"})


def test_env_file_parsing_is_utf8(tmp_path):
    path = tmp_path / ".env"
    path.write_text("LLM_MODEL=ن\n", encoding="utf-8")
    target: dict[str, str] = {}
    assert load_env_file(path, target) == 1 and target["LLM_MODEL"] == "ن"
