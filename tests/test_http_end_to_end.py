from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from conftest import CHINOOK, QUESTIONS
from typer.testing import CliRunner

from analystagent import cli

runner = CliRunner()


def make_handler(items):
    class Handler(BaseHTTPRequestHandler):
        requests_seen: list[dict] = []

        def log_message(self, *args):
            return

        def do_POST(self):
            length = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(length))
            Handler.requests_seen.append(body)
            messages = body["messages"]
            first_user = next(m["content"] for m in messages if m["role"] == "user")
            item = next(i for i in items if i.question in first_user)
            if messages[-1]["role"] == "tool":
                call = ("final_answer", {"answer": "Done.", "sql": item.golden_sql or ""})
            elif item.expect == "decline":
                call = ("final_answer", {"answer": "The database cannot answer this.", "sql": ""})
            else:
                call = ("run_sql", {"sql": item.golden_sql})
            payload = {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_http",
                                    "type": "function",
                                    "function": {"name": call[0], "arguments": json.dumps(call[1])},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


@pytest.fixture
def stub_server(items):
    handler = make_handler(items)
    handler.requests_seen = []
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", handler
    server.shutdown()
    server.server_close()


def test_cli_eval_over_real_http_sockets(stub_server, monkeypatch, tmp_path):
    base_url, handler = stub_server
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "stub-model")
    monkeypatch.setenv("LLM_BASE_URL", base_url)
    result = runner.invoke(
        cli.app,
        [
            "eval",
            "--db",
            str(CHINOOK),
            "--questions",
            str(QUESTIONS),
            "--intent",
            "count-customers",
            "--intent",
            "top-countries-revenue",
            "--intent",
            "revenue-by-year",
            "--intent",
            "unanswerable-salary",
            "--intent",
            "injection-drop-table",
            "--out",
            "reports",
            "--store",
            "runs/http.db",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "execution accuracy 100.0%" in result.output
    report = json.loads(next((tmp_path / "reports").glob("*.json")).read_text(encoding="utf-8"))
    assert report["aggregate"]["n_items"] == 10
    assert report["aggregate"]["decline_accuracy"] == 1.0
    assert report["aggregate"]["mean_total_tokens"] > 0
    assert report["meta"]["model"] == "stub-model"
    assert len(handler.requests_seen) == 18


def test_requests_carry_tools_and_system_prompt(stub_server, monkeypatch, tmp_path):
    base_url, handler = stub_server
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "stub-model")
    monkeypatch.setenv("LLM_BASE_URL", base_url)
    result = runner.invoke(
        cli.app,
        ["ask", "How many customers are there?", "--db", str(CHINOOK), "--store", "runs/a.db"],
    )
    assert result.exit_code == 0, result.output
    first = handler.requests_seen[0]
    assert first["model"] == "stub-model" and first["temperature"] == 0.0
    assert {t["function"]["name"] for t in first["tools"]} == {
        "list_tables",
        "describe_table",
        "sample_rows",
        "run_sql",
        "final_answer",
    }
    assert first["messages"][0]["role"] == "system"
    assert "Authorization" not in json.dumps(first)
