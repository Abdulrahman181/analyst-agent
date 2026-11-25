from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from analystagent.agent.loop import AnalystAgent
from analystagent.config import Budget
from analystagent.database import Database
from analystagent.db.connection import QueryError, execute_query
from analystagent.db.guard import check_sql
from analystagent.eval.baselines import (
    cheater_client,
    hallucinator_client,
    null_client,
    oracle_client,
)
from analystagent.eval.dataset import EvalItem
from analystagent.eval.harness import run_eval
from analystagent.eval.single_shot import SingleShotAnalyst
from analystagent.llm.scripted import ScriptedClient, text_response
from analystagent.semantic.glossary import GlossaryError, load_glossary
from analystagent.text import detect_language


@dataclass
class SelfCheckReport:
    n_items: int = 0
    n_answer_items: int = 0
    fixture_problems: list[str] = field(default_factory=list)
    oracle_correct: int = 0
    oracle_grounded: int = 0
    oracle_declines_ok: int = 0
    n_decline_items: int = 0
    single_shot_oracle_correct: int = 0
    null_answer_correct: int = 0
    null_declines_ok: int = 0
    hallucinator_blocked: int = 0
    cheater_blocked: bool = False
    cheater_db_unchanged: bool = False

    @property
    def ok(self) -> bool:
        return (
            not self.fixture_problems
            and self.oracle_correct == self.n_answer_items
            and self.oracle_grounded == self.n_answer_items
            and self.oracle_declines_ok == self.n_decline_items
            and self.single_shot_oracle_correct == self.n_answer_items
            and self.null_answer_correct == 0
            and self.null_declines_ok == self.n_decline_items
            and self.hallucinator_blocked == self.n_answer_items
            and self.cheater_blocked
            and self.cheater_db_unchanged
        )


def validate_fixtures(items: list[EvalItem], db: Database) -> list[str]:
    problems: list[str] = []
    if not items:
        return ["no questions loaded"]
    try:
        load_glossary()
    except GlossaryError as exc:
        problems.append(f"glossary: {exc}")
    seen_questions: set[str] = set()
    for item in items:
        if item.question in seen_questions:
            problems.append(f"{item.id}: duplicate question text")
        seen_questions.add(item.question)
        if detect_language(item.question) != item.language:
            problems.append(f"{item.id}: question language does not match its label")
        if item.expect == "decline":
            if item.golden_sql:
                problems.append(f"{item.id}: decline item must not have golden SQL")
            continue
        if not item.golden_sql:
            problems.append(f"{item.id}: answer item needs golden SQL")
            continue
        verdict = check_sql(item.golden_sql, db.schema)
        if not verdict.ok:
            problems.append(f"{item.id}: golden SQL rejected by guard: {verdict.message()}")
            continue
        if item.ordered and not re.search(r"(?i)\border\s+by\b", item.golden_sql):
            problems.append(f"{item.id}: ordered item needs ORDER BY in golden SQL")
        conn = db.connect()
        try:
            first = execute_query(conn, item.golden_sql, 1000, 10.0)
            second = execute_query(conn, item.golden_sql, 1000, 10.0)
        except QueryError as exc:
            problems.append(f"{item.id}: golden SQL failed: {exc}")
            continue
        finally:
            conn.close()
        if not first.rows:
            problems.append(f"{item.id}: golden SQL returned no rows")
        if first.rows != second.rows:
            problems.append(f"{item.id}: golden SQL is not deterministic")
        if first.truncated:
            problems.append(f"{item.id}: golden result exceeds the row limit")
    return problems


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_selfcheck(items: list[EvalItem], db: Database) -> SelfCheckReport:
    report = SelfCheckReport(
        n_items=len(items),
        n_answer_items=sum(1 for i in items if i.expect == "answer"),
        n_decline_items=sum(1 for i in items if i.expect == "decline"),
    )
    report.fixture_problems = validate_fixtures(items, db)
    if report.fixture_problems:
        return report

    oracle = run_eval(items, db, lambda item, _r: AnalystAgent(oracle_client(item, db), db))
    for row in oracle["results"]:
        if row["expect"] == "answer":
            report.oracle_correct += int(row["correct"])
            report.oracle_grounded += int(bool(row["grounded"]))
        else:
            report.oracle_declines_ok += int(row["correct"])

    by_id = {item.id: item for item in items}

    def single_shot_factory(item: EvalItem, _repeat: int) -> SingleShotAnalyst:
        sql = item.golden_sql
        reply = f"```sql\n{sql}\n```" if sql else "DECLINE"
        return SingleShotAnalyst(ScriptedClient([text_response(reply)]), db)

    single = run_eval(items, db, single_shot_factory)
    for row in single["results"]:
        if by_id[row["item_id"]].expect == "answer":
            report.single_shot_oracle_correct += int(row["correct"])

    null = run_eval(items, db, lambda item, _r: AnalystAgent(null_client(item.language), db))
    for row in null["results"]:
        if row["expect"] == "answer":
            report.null_answer_correct += int(row["correct"])
        else:
            report.null_declines_ok += int(row["correct"])

    answer_items = [i for i in items if i.expect == "answer"]
    halluc = run_eval(
        answer_items, db, lambda item, _r: AnalystAgent(hallucinator_client(item), db)
    )
    for row in halluc["results"]:
        report.hallucinator_blocked += int(row["status"] == "ungrounded")

    before = _file_hash(db.path)
    outcome = AnalystAgent(cheater_client(), db, budget=Budget()).ask("Delete everything", "en")
    report.cheater_blocked = (
        outcome.stop_reason == "policy_violations" and outcome.security_blocks == 3
    )
    report.cheater_db_unchanged = _file_hash(db.path) == before
    return report
