from __future__ import annotations

import re
import time
from typing import Any

from analystagent.agent.loop import AnswerResult, assess_confidence
from analystagent.agent.prompts import SINGLE_SHOT_SYSTEM_PROMPT, build_user_message
from analystagent.config import Budget
from analystagent.database import Database
from analystagent.db.connection import QueryError, execute_query
from analystagent.db.guard import check_sql
from analystagent.db.schema import render_overview
from analystagent.grounding import ExecutedResult
from analystagent.llm.base import LLMClient, LLMError
from analystagent.render import render_rows
from analystagent.semantic.glossary import Glossary, load_glossary
from analystagent.store import RunStore
from analystagent.text import detect_language

FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str | None:
    match = FENCE.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if re.match(r"(?is)^(select|with)\b", stripped):
        return stripped
    return None


class SingleShotAnalyst:
    def __init__(
        self,
        llm: LLMClient,
        db: Database,
        glossary: Glossary | None = None,
        budget: Budget | None = None,
        store: RunStore | None = None,
        run_meta: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.db = db
        self.glossary = glossary or load_glossary()
        self.budget = budget or Budget()
        self.store = store
        self.run_meta = run_meta or {}

    def ask(self, question: str, language: str | None = None) -> AnswerResult:
        lang = language or detect_language(question)
        started = time.monotonic()
        run_id = (
            self.store.create_run(question, lang, "single-shot", self.run_meta)
            if self.store
            else None
        )
        user_message = build_user_message(
            question,
            lang,
            render_overview(self.db.schema),
            self.glossary.render_hints(question),
        )
        messages = [
            {"role": "system", "content": SINGLE_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        prompt_tokens = completion_tokens = 0
        status = "failed"
        stop_reason = "no_sql"
        answer: str | None = None
        sql = ""
        executed: ExecutedResult | None = None
        security_blocks = query_errors = 0

        try:
            response = self.llm.chat(messages, [])
        except LLMError as exc:
            stop_reason = "llm_error"
            if self.store is not None and run_id is not None:
                self.store.add_step(run_id, "llm_error", result=str(exc), ok=False)
            response = None

        if response is not None:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            content = response.content or ""
            if self.store is not None and run_id is not None:
                self.store.add_step(
                    run_id,
                    "llm",
                    result=content,
                    latency_s=response.latency_s,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            candidate = extract_sql(content)
            if candidate is None and re.search(r"\bDECLINE\b", content):
                status = "answered"
                stop_reason = "declined"
                answer = (
                    "لا يمكن الإجابة على هذا السؤال من قاعدة البيانات."
                    if lang == "ar"
                    else "The database cannot answer this question."
                )
            elif candidate is not None:
                verdict = check_sql(candidate, self.db.schema)
                if not verdict.ok:
                    stop_reason = "guard_blocked"
                    security_blocks = int(verdict.security)
                else:
                    conn = self.db.connect()
                    try:
                        result = execute_query(
                            conn, verdict.sql, self.budget.max_rows, self.budget.sql_timeout_s
                        )
                        executed = ExecutedResult(
                            verdict.sql, result.columns, result.rows, result.truncated
                        )
                        sql = verdict.sql
                        answer = render_rows(result.columns, result.rows, lang)
                        status = "answered"
                        stop_reason = "finished"
                    except QueryError:
                        stop_reason = "sql_error"
                        query_errors = 1
                    finally:
                        conn.close()

        confidence = assess_confidence(
            status, sql, None, bool(executed and executed.truncated), False
        )
        wall_s = time.monotonic() - started
        outcome = AnswerResult(
            run_id=run_id,
            question=question,
            language=lang,
            mode="single-shot",
            status=status,
            stop_reason=stop_reason,
            answer=answer,
            sql=sql,
            grounded=None,
            ungrounded_numbers=(),
            confidence=confidence,
            steps=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_s=wall_s,
            security_blocks=security_blocks,
            query_errors=query_errors,
            grounding_rejections=0,
            sql_executions=1 if executed else 0,
            result=executed,
            meta=dict(self.run_meta),
            answer_source="template",
        )
        if self.store is not None and run_id is not None:
            self.store.finish_run(
                run_id,
                status=status,
                stop_reason=stop_reason,
                answer=answer,
                cited_sql=sql,
                grounded=None,
                confidence=confidence,
                steps=1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                security_blocks=security_blocks,
                query_errors=query_errors,
                grounding_rejections=0,
                wall_s=wall_s,
            )
        return outcome
