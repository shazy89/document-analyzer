from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from document_analyzer.models.chat import (
    DesignerResumeRequest,
    DesignerResponse,
    DesignerRunRequest,
)
from general_agent.agent import DesignerAgent

logger = logging.getLogger(__name__)

designer_router = APIRouter(prefix="/api/v1/designer", tags=["designer"])

# Single agent instance shared across requests (MemorySaver is in-process).
_agent: DesignerAgent | None = None


def _get_agent() -> DesignerAgent:
    global _agent
    if _agent is None:
        _agent = DesignerAgent.from_env()
    return _agent


@designer_router.post("/run", response_model=DesignerResponse)
def designer_run(body: DesignerRunRequest) -> DesignerResponse:
    """Start a designer agent session. May return questions or a final result."""
    agent = _get_agent()
    try:
        result = agent.run(
            message=body.message,
            thread_id=body.thread_id,
            user_id=body.user_id,
            profile_id=body.profile_id,
            is_new_session=body.is_new_session,
        )
    except Exception as exc:
        logger.exception("Designer agent run failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return DesignerResponse(**result)


@designer_router.post("/resume", response_model=DesignerResponse)
def designer_resume(body: DesignerResumeRequest) -> DesignerResponse:
    """Resume a paused session by providing answers to discovery questions."""
    agent = _get_agent()
    try:
        result = agent.resume(
            thread_id=body.thread_id,
            user_answers=body.user_answers,
        )
    except Exception as exc:
        logger.exception("Designer agent resume failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return DesignerResponse(**result)
