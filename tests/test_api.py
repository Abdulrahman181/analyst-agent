from __future__ import annotations

import pytest
from conftest import final, run_sql
from fastapi.testclient import TestClient

from analystagent.api import create_app
from analystagent.config import ConfigError
from analystagent.llm.scripted import ScriptedClient, text_response
from analystagent.store import RunStore

COUNT = "SELECT COUNT(*) FROM Customer"


def make_client(chinook, script=None, api_key=None, factory=None):
    store = RunStore()
    holder = {"script": script or []}

    def default_factory():
        return ScriptedClient(list(holder["script"]))

    app = create_app(chinook, factory or default_factory, store, api_key=api_key)
    return TestClient(app), store


def test_healthz(chinook):
    client, _ = make_client(chinook)
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["tables"] == 11
    assert body["llm_configured"] is True and body["auth_required"] is False


def test_healthz_reports_missing_llm_configuration(chinook):
    def broken():
        raise ConfigError("no key")

    client, _ = make_client(chinook, factory=broken)
    assert client.get("/healthz").json()["llm_configured"] is False


def test_ask_returns_a_grounded_answer(chinook):
    client, _ = make_client(chinook, [run_sql(COUNT), final("There are 59 customers.", COUNT)])
    response = client.post("/v1/ask", json={"question": "How many customers are there?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered" and body["grounded"] is True
    assert body["sql"] == COUNT and body["confidence"] == "high" and body["language"] == "en"
    assert body["result"] == {"columns": ["COUNT(*)"], "rows": [[59]], "truncated": False}
    assert body["run_id"] >= 1


def test_ask_in_arabic_with_explicit_language(chinook):
    client, _ = make_client(chinook, [run_sql(COUNT), final("يوجد ٥٩ عميلاً", COUNT)])
    body = client.post("/v1/ask", json={"question": "كم عدد العملاء؟", "language": "ar"}).json()
    assert body["language"] == "ar" and body["grounded"] is True


def test_ungrounded_result_is_visible_to_the_caller(chinook):
    wrong = final("There are 60 customers.", COUNT)
    client, _ = make_client(chinook, [run_sql(COUNT), wrong, wrong, wrong])
    body = client.post("/v1/ask", json={"question": "How many customers?"}).json()
    assert body["status"] == "ungrounded" and body["grounded"] is False
    assert body["ungrounded_numbers"] == ["60"] and body["confidence"] == "low"


def test_failed_run_returns_null_answer(chinook):
    client, _ = make_client(chinook, [text_response("hmm")] * 3)
    body = client.post("/v1/ask", json={"question": "q"}).json()
    assert body["status"] == "failed" and body["answer"] is None and body["result"] is None


def test_result_rows_are_limited_in_the_response(chinook):
    sql = "SELECT TrackId FROM Track"
    client, _ = make_client(chinook, [run_sql(sql), final("Returned 200 rows.", sql)])
    body = client.post("/v1/ask", json={"question": "List tracks"}).json()
    assert len(body["result"]["rows"]) == 20 and body["result"]["truncated"] is True


def test_single_shot_mode(chinook):
    client, _ = make_client(chinook, [text_response(f"```sql\n{COUNT}\n```")])
    body = client.post(
        "/v1/ask", json={"question": "How many customers?", "mode": "single-shot"}
    ).json()
    assert body["status"] == "answered" and body["steps"] == 1 and body["grounded"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "x" * 1001},
        {"question": "q", "language": "fr"},
        {"question": "q", "mode": "magic"},
        {"question": 5},
    ],
)
def test_invalid_requests_are_rejected(chinook, payload):
    client, _ = make_client(chinook)
    assert client.post("/v1/ask", json=payload).status_code == 422


def test_missing_llm_configuration_is_a_503(chinook):
    def broken():
        raise ConfigError("No API key found. Set GROQ_API_KEY.")

    client, _ = make_client(chinook, factory=broken)
    response = client.post("/v1/ask", json={"question": "q"})
    assert response.status_code == 503 and "GROQ_API_KEY" in response.json()["detail"]


def test_internal_errors_do_not_leak_details(chinook):
    store = RunStore()

    class Boom:
        def chat(self, *args, **kwargs):
            raise RuntimeError("secret-internal-path /etc/passwd")

    app = create_app(chinook, lambda: Boom(), store)
    response = TestClient(app).post("/v1/ask", json={"question": "q"})
    assert response.status_code == 500
    assert "secret" not in response.text and "passwd" not in response.text


def test_runs_can_be_fetched(chinook):
    client, _ = make_client(chinook, [run_sql(COUNT), final("59", COUNT)])
    run_id = client.post("/v1/ask", json={"question": "How many customers?"}).json()["run_id"]
    body = client.get(f"/v1/runs/{run_id}").json()
    assert body["run"]["status"] == "answered"
    assert [s["tool"] for s in body["steps"] if s["kind"] == "tool"] == ["run_sql", "final_answer"]


def test_unknown_run_is_404(chinook):
    client, _ = make_client(chinook)
    assert client.get("/v1/runs/12345").status_code == 404
    assert client.get("/v1/runs/abc").status_code == 422


def test_api_key_is_enforced_when_configured(chinook):
    client, _ = make_client(chinook, [run_sql(COUNT), final("59", COUNT)], api_key="s3cret-key")
    body = {"question": "How many customers?"}
    assert client.get("/healthz").json()["auth_required"] is True
    assert client.post("/v1/ask", json=body).status_code == 401
    assert client.post("/v1/ask", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/v1/runs/1").status_code == 401
    ok = client.post("/v1/ask", json=body, headers={"X-API-Key": "s3cret-key"})
    assert ok.status_code == 200
    assert client.get("/v1/runs/1", headers={"X-API-Key": "s3cret-key"}).status_code == 200


def test_destructive_question_never_modifies_the_database(chinook):
    before = chinook.path.read_bytes()
    client, _ = make_client(
        chinook,
        [
            run_sql("DELETE FROM Customer"),
            run_sql("DROP TABLE Customer"),
            run_sql("PRAGMA writable_schema=1"),
        ],
    )
    body = client.post("/v1/ask", json={"question": "Delete all customers"}).json()
    assert body["status"] == "failed" and body["stop_reason"] == "policy_violations"
    assert chinook.path.read_bytes() == before


def test_concurrent_requests_do_not_interfere(chinook):
    from concurrent.futures import ThreadPoolExecutor

    script = [run_sql(COUNT), final("59", COUNT)]
    client, _ = make_client(chinook, script)

    def call(_):
        return client.post("/v1/ask", json={"question": "How many customers?"}).json()

    with ThreadPoolExecutor(max_workers=6) as pool:
        bodies = list(pool.map(call, range(12)))
    assert all(b["status"] == "answered" for b in bodies)
    assert len({b["run_id"] for b in bodies}) == 12
