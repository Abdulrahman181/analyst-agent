from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from analystagent.agent.prompts import (
    FORMAT_NUDGE,
    NO_TOOL_NUDGE,
    REPEAT_NOTE,
    SYSTEM_PROMPT,
    build_user_message,
)
from analystagent.agent.tools import RunState, ToolContext, ToolRegistry
from analystagent.config import Budget
from analystagent.database import Database
from analystagent.db.schema import render_overview
from analystagent.grounding import ExecutedResult
from analystagent.llm.base import LLMClient, LLMError, LLMToolFormatError
from analystagent.semantic.glossary import Glossary, load_glossary
from analystagent.store import RunStore
from analystagent.text import detect_language

MAX_CALLS_PER_TURN = 3
NO_TOOL_LIMIT = 3
MAX_TOOL_CHARS = 5000


@dataclass
class AnswerResult:
    run_id: int | None
    question: str
    language: str
    mode: str
    status: str
    stop_reason: str
    answer: str | None
    sql: str
    grounded: bool | None
    ungrounded_numbers: tuple[str, ...]
    confidence: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    security_blocks: int
    query_errors: int
    grounding_rejections: int
    sql_executions: int
    result: ExecutedResult | None
    meta: dict[str, Any] = field(default_factory=dict)
    answer_source: str = "model"

    @property
    def declined(self) -> bool:
        return self.status == "answered" and not self.sql


def assess_confidence(
    status: str, sql: str, grounded: bool | None, truncated: bool, had_trouble: bool
) -> str:
    if status != "answered" or not sql or grounded is False:
        return "low"
    if truncated or had_trouble:
        return "medium"
    return "high"


def _clip(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


class AnalystAgent:
    def __init__(
        self,
        llm: LLMClient,
        db: Database,
        glossary: Glossary | None = None,
        registry: ToolRegistry | None = None,
        budget: Budget | None = None,
        store: RunStore | None = None,
        run_meta: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.llm = llm
        self.db = db
        self.glossary = glossary or load_glossary()
        self.registry = registry or ToolRegistry()
        self.budget = budget or Budget()
        self.store = store
        self.run_meta = run_meta or {}
        self.clock = clock

    def _log(self, run_id: int | None, kind: str, **fields: Any) -> None:
        if self.store is not None and run_id is not None:
            self.store.add_step(run_id, kind, **fields)

    def ask(self, question: str, language: str | None = None) -> AnswerResult:
        lang = language or detect_language(question)
        conn = self.db.connect()
        try:
            return self._run(question, lang, conn)
        finally:
            conn.close()

    def _run(self, question: str, lang: str, conn: Any) -> AnswerResult:
        budget = self.budget
        state = RunState()
        ctx = ToolContext(self.db, conn, budget, state, question)
        run_id = (
            self.store.create_run(question, lang, "agent", self.run_meta) if self.store else None
        )
        user_message = build_user_message(
            question,
            lang,
            render_overview(self.db.schema),
            self.glossary.render_hints(question),
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        specs = self.registry.specs()
        started = self.clock()

        prompt_tokens = completion_tokens = steps = no_tool_turns = 0
        stop_reason = "max_steps"
        final: dict[str, Any] | None = None
        seen: dict[str, int] = {}

        while True:
            if steps >= budget.max_steps:
                stop_reason = "max_steps"
                break
            if prompt_tokens + completion_tokens >= budget.max_total_tokens:
                stop_reason = "max_tokens"
                break
            if self.clock() - started >= budget.max_wall_s:
                stop_reason = "timeout"
                break
            steps += 1

            try:
                response = self.llm.chat(messages, specs)
            except LLMToolFormatError as exc:
                self._log(run_id, "llm_error", result=str(exc), ok=False)
                no_tool_turns += 1
                if no_tool_turns >= NO_TOOL_LIMIT:
                    stop_reason = "no_tool_calls"
                    break
                messages.append({"role": "user", "content": FORMAT_NUDGE})
                continue
            except LLMError as exc:
                self._log(run_id, "llm_error", result=str(exc), ok=False)
                stop_reason = "llm_error"
                break

            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            self._log(
                run_id,
                "llm",
                result=response.content,
                latency_s=response.latency_s,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )

            calls = response.tool_calls[:MAX_CALLS_PER_TURN]
            if not calls:
                messages.append({"role": "assistant", "content": response.content or ""})
                no_tool_turns += 1
                if no_tool_turns >= NO_TOOL_LIMIT:
                    stop_reason = "no_tool_calls"
                    break
                messages.append({"role": "user", "content": NO_TOOL_NUDGE})
                continue
            no_tool_turns = 0

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.raw_arguments or "{}",
                            },
                        }
                        for call in calls
                    ],
                }
            )

            stalled = False
            for call in calls:
                key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
                seen[key] = seen.get(key, 0) + 1
                tool_started = time.perf_counter()
                outcome = self.registry.execute(call, ctx)
                text = outcome.text
                if seen[key] == budget.max_repeat_actions - 1:
                    text += REPEAT_NOTE
                if seen[key] >= budget.max_repeat_actions:
                    stalled = True
                self._log(
                    run_id,
                    "tool",
                    tool=call.name,
                    args=call.arguments,
                    result=text,
                    ok=outcome.ok,
                    security_block=outcome.security_block,
                    latency_s=time.perf_counter() - tool_started,
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": _clip(text)})
                if outcome.terminal:
                    final = outcome.data
                    break

            if final is not None:
                stop_reason = "finished" if final.get("grounded") else "ungrounded"
                break
            if state.security_blocks >= budget.max_guard_blocks:
                stop_reason = "policy_violations"
                break
            if stalled:
                stop_reason = "stalled"
                break

        wall_s = self.clock() - started
        result = self._build_result(
            run_id,
            question,
            lang,
            state,
            final,
            stop_reason,
            steps,
            prompt_tokens,
            completion_tokens,
            wall_s,
        )
        if self.store is not None and run_id is not None:
            self.store.finish_run(
                run_id,
                status=result.status,
                stop_reason=result.stop_reason,
                answer=result.answer,
                cited_sql=result.sql,
                grounded=result.grounded,
                confidence=result.confidence,
                steps=steps,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                security_blocks=state.security_blocks,
                query_errors=state.query_errors,
                grounding_rejections=state.grounding_rejections,
                wall_s=wall_s,
            )
        return result

    def _build_result(
        self,
        run_id: int | None,
        question: str,
        lang: str,
        state: RunState,
        final: dict[str, Any] | None,
        stop_reason: str,
        steps: int,
        prompt_tokens: int,
        completion_tokens: int,
        wall_s: float,
    ) -> AnswerResult:
        answer = None
        sql = ""
        grounded: bool | None = None
        ungrounded: tuple[str, ...] = ()
        cited: ExecutedResult | None = None
        status = "failed"
        if final is not None:
            answer = str(final.get("answer", ""))
            sql = str(final.get("sql", ""))
            grounded = bool(final.get("grounded"))
            cited = final.get("result")
            status = "answered" if grounded else "ungrounded"
            if not grounded and state.last_grounding is not None:
                ungrounded = tuple(state.last_grounding.ungrounded)
        truncated = bool(cited and cited.truncated)
        confidence = assess_confidence(status, sql, grounded, truncated, state.had_trouble)
        return AnswerResult(
            run_id=run_id,
            question=question,
            language=lang,
            mode="agent",
            status=status,
            stop_reason=stop_reason,
            answer=answer,
            sql=sql,
            grounded=grounded,
            ungrounded_numbers=ungrounded,
            confidence=confidence,
            steps=steps,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_s=wall_s,
            security_blocks=state.security_blocks,
            query_errors=state.query_errors,
            grounding_rejections=state.grounding_rejections,
            sql_executions=len(state.executed),
            result=cited,
            meta=dict(self.run_meta),
        )
