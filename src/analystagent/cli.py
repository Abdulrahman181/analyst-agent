from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from analystagent.agent.loop import AnalystAgent
from analystagent.config import Budget, ConfigError, LLMSettings, load_env_file
from analystagent.database import Database
from analystagent.eval.dataset import DatasetError, EvalItem, load_items
from analystagent.eval.harness import ItemResult, run_eval
from analystagent.eval.metrics import compare_reports
from analystagent.eval.report import write_report
from analystagent.eval.selfcheck import run_selfcheck
from analystagent.eval.single_shot import SingleShotAnalyst
from analystagent.llm.openai_compat import OpenAICompatClient
from analystagent.paths import default_db_path, default_questions_path
from analystagent.semantic.glossary import GlossaryError, load_glossary
from analystagent.store import RunStore

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Arabic/English data analyst agent."
)

DbOption = Annotated[Path | None, typer.Option("--db", help="SQLite database to query.")]
QuestionsOption = Annotated[
    Path | None, typer.Option("--questions", help="Benchmark questions JSON.")
]
ProviderOption = Annotated[str | None, typer.Option("--provider", help="groq, ollama or openai.")]
ModelOption = Annotated[str | None, typer.Option("--model", help="Model id for the provider.")]
GlossaryOption = Annotated[Path | None, typer.Option("--glossary", help="Custom glossary YAML.")]
StoreOption = Annotated[Path, typer.Option("--store", help="SQLite file for run traces.")]


@app.callback()
def _main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _fail(message: str, code: int = 2) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def _open_db(path: Path | None) -> Database:
    target = path or default_db_path()
    try:
        return Database.open(target)
    except Exception as exc:
        _fail(f"cannot open database {target}: {exc}")
    raise typer.Exit(2)


def _settings(provider: str | None, model: str | None) -> LLMSettings:
    load_env_file(".env")
    try:
        return LLMSettings.from_env(provider=provider, model=model)
    except ConfigError as exc:
        _fail(str(exc))
    raise typer.Exit(2)


def _glossary(path: Path | None):
    try:
        return load_glossary(path)
    except GlossaryError as exc:
        _fail(str(exc))
    raise typer.Exit(2)


def _items(questions: Path | None, languages: list[str], only: list[str] | None) -> list[EvalItem]:
    try:
        return load_items(questions or default_questions_path(), languages, only)
    except DatasetError as exc:
        _fail(str(exc))
    return []


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question in English or Arabic.")],
    mode: Annotated[str, typer.Option("--mode", help="agent or single-shot.")] = "agent",
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON.")] = False,
    provider: ProviderOption = None,
    model: ModelOption = None,
    db: DbOption = None,
    glossary: GlossaryOption = None,
    store_path: StoreOption = Path("runs/analystagent.db"),
) -> None:
    if mode not in ("agent", "single-shot"):
        _fail("mode must be agent or single-shot")
    settings = _settings(provider, model)
    database = _open_db(db)
    meta = {"provider": settings.provider, "model": settings.model}
    store = RunStore(store_path)
    llm = OpenAICompatClient(settings)
    analyst: AnalystAgent | SingleShotAnalyst
    if mode == "agent":
        analyst = AnalystAgent(llm, database, _glossary(glossary), store=store, run_meta=meta)
    else:
        analyst = SingleShotAnalyst(llm, database, _glossary(glossary), store=store, run_meta=meta)
    outcome = analyst.ask(question)
    store.close()
    if as_json:
        payload = {
            "run_id": outcome.run_id,
            "status": outcome.status,
            "stop_reason": outcome.stop_reason,
            "answer": outcome.answer,
            "sql": outcome.sql,
            "grounded": outcome.grounded,
            "ungrounded_numbers": list(outcome.ungrounded_numbers),
            "confidence": outcome.confidence,
            "steps": outcome.steps,
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(f"status     : {outcome.status} ({outcome.stop_reason})")
    typer.echo(f"answer     : {outcome.answer or '-'}")
    typer.echo(f"sql        : {outcome.sql or '-'}")
    typer.echo(f"grounded   : {outcome.grounded}")
    typer.echo(f"confidence : {outcome.confidence}")
    typer.echo(f"run id     : {outcome.run_id}")


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    provider: ProviderOption = None,
    model: ModelOption = None,
    db: DbOption = None,
    glossary: GlossaryOption = None,
    store_path: StoreOption = Path("runs/analystagent.db"),
) -> None:
    import uvicorn

    from analystagent.api import create_app

    load_env_file(".env")
    database = _open_db(db)

    def llm_factory() -> OpenAICompatClient:
        return OpenAICompatClient(LLMSettings.from_env(provider=provider, model=model))

    application = create_app(
        database,
        llm_factory,
        RunStore(store_path),
        glossary=_glossary(glossary),
        api_key=os.environ.get("ANALYST_API_KEY") or None,
    )
    uvicorn.run(application, host=host, port=port, log_level="info")


@app.command()
def questions(
    questions_file: QuestionsOption = None,
    language: Annotated[str, typer.Option("--language", help="en, ar or all.")] = "all",
) -> None:
    langs = ["en", "ar"] if language == "all" else [language]
    for item in _items(questions_file, langs, None):
        typer.echo(f"{item.id:<36} {item.difficulty:<7} {item.category:<13} {item.question}")


@app.command()
def selfcheck(
    questions_file: QuestionsOption = None,
    db: DbOption = None,
) -> None:
    database = _open_db(db)
    items = _items(questions_file, ["en", "ar"], None)
    typer.echo(f"Validating {len(items)} questions and the harness (no LLM involved)...")
    report = run_selfcheck(items, database)
    for problem in report.fixture_problems:
        typer.echo(f"FIXTURE PROBLEM {problem}", err=True)
    if report.fixture_problems:
        raise typer.Exit(1)
    n, d = report.n_answer_items, report.n_decline_items
    typer.echo(f"oracle agent correct        : {report.oracle_correct}/{n} (expected all)")
    typer.echo(f"oracle answers grounded     : {report.oracle_grounded}/{n} (expected all)")
    typer.echo(f"oracle declines correct     : {report.oracle_declines_ok}/{d} (expected all)")
    typer.echo(
        f"single-shot oracle correct  : {report.single_shot_oracle_correct}/{n} (expected all)"
    )
    typer.echo(f"null agent correct          : {report.null_answer_correct}/{n} (expected 0)")
    typer.echo(f"hallucinated numbers blocked: {report.hallucinator_blocked}/{n} (expected all)")
    typer.echo(
        f"write attempts blocked      : {report.cheater_blocked}, database unchanged: "
        f"{report.cheater_db_unchanged}"
    )
    typer.echo("SELFCHECK PASSED" if report.ok else "SELFCHECK FAILED")
    raise typer.Exit(0 if report.ok else 1)


@app.command("eval")
def eval_cmd(
    mode: Annotated[str, typer.Option("--mode", help="agent or single-shot.")] = "agent",
    language: Annotated[str, typer.Option("--language", help="en, ar or all.")] = "all",
    intent: Annotated[
        list[str] | None, typer.Option("--intent", help="Intent id, repeatable. Default: all.")
    ] = None,
    repeats: Annotated[int, typer.Option("--repeats", min=1)] = 1,
    out: Annotated[Path, typer.Option("--out", help="Report output directory.")] = Path("reports"),
    max_steps: Annotated[int, typer.Option("--max-steps")] = 10,
    provider: ProviderOption = None,
    model: ModelOption = None,
    db: DbOption = None,
    questions_file: QuestionsOption = None,
    glossary: GlossaryOption = None,
    store_path: StoreOption = Path("runs/analystagent.db"),
) -> None:
    if mode not in ("agent", "single-shot"):
        _fail("mode must be agent or single-shot")
    if language not in ("en", "ar", "all"):
        _fail("language must be en, ar or all")
    settings = _settings(provider, model)
    database = _open_db(db)
    shared_glossary = _glossary(glossary)
    items = _items(questions_file, ["en", "ar"] if language == "all" else [language], intent)
    store = RunStore(store_path)
    budget = Budget(max_steps=max_steps)
    meta = {"provider": settings.provider, "model": settings.model}

    def factory(_item: EvalItem, _repeat: int):
        llm = OpenAICompatClient(settings)
        if mode == "agent":
            return AnalystAgent(
                llm, database, shared_glossary, budget=budget, store=store, run_meta=dict(meta)
            )
        return SingleShotAnalyst(
            llm, database, shared_glossary, budget=budget, store=store, run_meta=dict(meta)
        )

    def show(record: ItemResult) -> None:
        mark = "PASS" if record.correct else "FAIL"
        typer.echo(
            f"[{mark}] {record.item_id} repeat={record.repeat} status={record.status} "
            f"stop={record.stop_reason} steps={record.steps}"
        )

    report = run_eval(
        items, database, factory, repeats, show, {**meta, "mode": mode, "max_steps": max_steps}
    )
    store.close()
    json_path, md_path = write_report(report, out)
    agg = report["aggregate"]
    typer.echo(
        f"execution accuracy {agg['exec_accuracy'] * 100:.1f}% on "
        f"{agg['n_answer_items']} answerable items, "
        f"declines correct {agg['decline_accuracy'] * 100:.1f}%"
    )
    typer.echo(f"report: {md_path}")
    typer.echo(f"data  : {json_path}")


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="Baseline report JSON.")],
    candidate: Annotated[Path, typer.Argument(help="Candidate report JSON.")],
    max_drop: Annotated[float, typer.Option("--max-drop", min=0.0, max=1.0)] = 0.05,
) -> None:
    try:
        base = json.loads(baseline.read_text(encoding="utf-8"))
        cand = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(f"could not read reports: {exc}")
    result = compare_reports(base, cand, max_drop)
    typer.echo(
        f"baseline {result['baseline_rate'] * 100:.1f}% -> candidate "
        f"{result['candidate_rate'] * 100:.1f}% (delta {result['delta'] * 100:+.1f} points, "
        f"allowed drop {max_drop * 100:.1f})"
    )
    typer.echo(f"regressions : {', '.join(result['regressions']) or '-'}")
    typer.echo(f"improvements: {', '.join(result['improvements']) or '-'}")
    typer.echo("GATE PASSED" if result["passed"] else "GATE FAILED")
    raise typer.Exit(0 if result["passed"] else 1)


@app.command()
def trace(
    run_id: Annotated[int, typer.Argument(help="Run id printed by 'ask' or stored in reports.")],
    store_path: StoreOption = Path("runs/analystagent.db"),
) -> None:
    if not store_path.is_file():
        _fail(f"database not found: {store_path}")
    store = RunStore(store_path)
    record = store.get_run(run_id)
    if record is None:
        _fail(f"run {run_id} not found")
    assert record is not None
    typer.echo(
        f"run {record['id']} mode={record['mode']} model={record['model']} "
        f"status={record['status']} stop={record['stop_reason']} grounded={record['grounded']}"
    )
    typer.echo(f"question: {record['question']}")
    for step in store.steps_for(run_id):
        label = step["tool"] or step["kind"]
        flag = " BLOCKED" if step["security_block"] else ""
        preview = (step["result_preview"] or "").replace("\n", " ")[:100]
        typer.echo(f"{step['idx']:>3} {label:<16}{flag} {preview}")
    store.close()


if __name__ == "__main__":
    app()
