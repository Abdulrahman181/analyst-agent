from __future__ import annotations

import sqlite3
import threading

import pytest

from analystagent.store import RunStore


@pytest.fixture
def store():
    with RunStore() as instance:
        yield instance


def test_create_and_finish_run(store):
    run_id = store.create_run("q", "en", "agent", {"provider": "p", "model": "m"})
    store.finish_run(run_id, status="answered", grounded=True, steps=3, wall_s=1.5)
    record = store.get_run(run_id)
    assert record["question"] == "q" and record["language"] == "en"
    assert record["provider"] == "p" and record["model"] == "m"
    assert record["status"] == "answered" and record["grounded"] == 1
    assert record["steps"] == 3 and record["finished_at"]


def test_unknown_run_fields_are_rejected(store):
    run_id = store.create_run("q", "en", "agent")
    with pytest.raises(ValueError, match="unknown run fields"):
        store.finish_run(run_id, status="x", drop_table="y")


def test_steps_are_ordered_and_clipped(store):
    run_id = store.create_run("q", "en", "agent")
    store.add_step(run_id, "llm", result="a" * 2000)
    store.add_step(run_id, "tool", tool="run_sql", args={"sql": "SELECT 1"}, result="ok", ok=True)
    store.add_step(run_id, "tool", tool="run_sql", args={"sql": "x" * 5000}, ok=False)
    steps = store.steps_for(run_id)
    assert [s["idx"] for s in steps] == [0, 1, 2]
    assert (
        steps[0]["result_preview"].endswith("[clipped]") and len(steps[0]["result_preview"]) < 600
    )
    assert steps[1]["ok"] == 1 and steps[2]["ok"] == 0
    assert len(steps[2]["args_json"]) < 2100


def test_arabic_text_round_trips(store):
    run_id = store.create_run("كم عدد العملاء؟", "ar", "agent")
    store.add_step(run_id, "tool", tool="run_sql", args={"sql": "SELECT 'ع'"})
    assert store.get_run(run_id)["question"] == "كم عدد العملاء؟"
    assert "ع" in store.steps_for(run_id)[0]["args_json"]


def test_missing_run(store):
    assert store.get_run(999) is None
    assert store.steps_for(999) == []


def test_list_runs_newest_first(store):
    first = store.create_run("a", "en", "agent")
    second = store.create_run("b", "en", "agent")
    assert [r["id"] for r in store.list_runs()] == [second, first]
    assert len(store.list_runs(limit=1)) == 1


def test_file_store_persists_and_creates_directories(tmp_path):
    path = tmp_path / "nested" / "runs.db"
    with RunStore(path) as first:
        run_id = first.create_run("q", "en", "agent")
    with RunStore(path) as second:
        assert second.get_run(run_id)["question"] == "q"


def test_security_block_flag_is_stored(store):
    run_id = store.create_run("q", "en", "agent")
    store.add_step(run_id, "tool", tool="run_sql", ok=False, security_block=True)
    assert store.steps_for(run_id)[0]["security_block"] == 1


def test_concurrent_writes_from_threads(store):
    run_ids = []

    def worker(n):
        rid = store.create_run(f"q{n}", "en", "agent")
        run_ids.append(rid)
        for _ in range(10):
            store.add_step(rid, "llm", result="x")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(run_ids)) == 8
    assert all(len(store.steps_for(rid)) == 10 for rid in run_ids)


def test_closed_store_cannot_be_used():
    store = RunStore()
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.create_run("q", "en", "agent")


def test_sql_injection_in_question_is_stored_as_text(store):
    run_id = store.create_run("'); DROP TABLE runs; --", "en", "agent")
    assert store.get_run(run_id)["question"] == "'); DROP TABLE runs; --"
