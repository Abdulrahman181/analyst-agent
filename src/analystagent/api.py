from __future__ import annotations

import hmac
import logging
from collections.abc import Callable
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from analystagent import __version__
from analystagent.agent.loop import AnalystAgent, AnswerResult
from analystagent.config import Budget, ConfigError
from analystagent.database import Database
from analystagent.eval.single_shot import SingleShotAnalyst
from analystagent.llm.base import LLMClient
from analystagent.semantic.glossary import Glossary, load_glossary
from analystagent.store import RunStore

LOGGER = logging.getLogger("analystagent.api")
RESULT_ROWS_IN_RESPONSE = 20

LLMFactory = Callable[[], LLMClient]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    language: Literal["ar", "en"] | None = None
    mode: Literal["agent", "single-shot"] = "agent"


class ResultTable(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    truncated: bool


class AskResponse(BaseModel):
    run_id: int | None
    status: str
    stop_reason: str
    language: str
    answer: str | None
    sql: str
    grounded: bool | None
    ungrounded_numbers: list[str]
    confidence: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    wall_s: float
    result: ResultTable | None


def _to_response(outcome: AnswerResult) -> AskResponse:
    table = None
    if outcome.result is not None:
        cited = outcome.result
        shown = cited.rows[:RESULT_ROWS_IN_RESPONSE]
        table = ResultTable(
            columns=list(cited.columns),
            rows=[list(row) for row in shown],
            truncated=cited.truncated or len(cited.rows) > len(shown),
        )
    return AskResponse(
        run_id=outcome.run_id,
        status=outcome.status,
        stop_reason=outcome.stop_reason,
        language=outcome.language,
        answer=outcome.answer,
        sql=outcome.sql,
        grounded=outcome.grounded,
        ungrounded_numbers=list(outcome.ungrounded_numbers),
        confidence=outcome.confidence,
        steps=outcome.steps,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        wall_s=round(outcome.wall_s, 3),
        result=table,
    )


def create_app(
    db: Database,
    llm_factory: LLMFactory,
    store: RunStore,
    glossary: Glossary | None = None,
    budget: Budget | None = None,
    api_key: str | None = None,
    run_meta: dict[str, Any] | None = None,
) -> FastAPI:
    app = FastAPI(title="AnalystAgent", version=__version__)
    shared_glossary = glossary or load_glossary()
    shared_budget = budget or Budget()
    meta = run_meta or {}

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if api_key and not (x_api_key and hmac.compare_digest(x_api_key, api_key)):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        try:
            llm_factory()
            llm_ready = True
        except ConfigError:
            llm_ready = False
        return {
            "status": "ok",
            "version": __version__,
            "tables": len(db.schema.tables),
            "llm_configured": llm_ready,
            "auth_required": bool(api_key),
        }

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    def ask(request: AskRequest) -> AskResponse:
        try:
            llm = llm_factory()
        except ConfigError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            if request.mode == "single-shot":
                analyst: AnalystAgent | SingleShotAnalyst = SingleShotAnalyst(
                    llm, db, shared_glossary, shared_budget, store, meta
                )
            else:
                analyst = AnalystAgent(llm, db, shared_glossary, None, shared_budget, store, meta)
            outcome = analyst.ask(request.question, request.language)
        except Exception:
            LOGGER.exception("ask failed")
            raise HTTPException(status_code=500, detail="internal error while answering") from None
        return _to_response(outcome)

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(require_key)])
    def get_run(run_id: int) -> dict[str, Any]:
        record = store.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run": record, "steps": store.steps_for(run_id)}

    return app
