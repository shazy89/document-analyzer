from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request

from document_analyzer.api.router import router
from document_analyzer.core.config import get_settings
from document_analyzer.core.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title=settings.app_name, version=settings.app_version)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        logger.info("→ %s %s", request.method, request.url.path)
        response = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        logger.info("← %s %s %.1fms", response.status_code, request.url.path, ms)
        return response

    app.include_router(router)
    return app


app = create_app()
