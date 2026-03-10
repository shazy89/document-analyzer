from __future__ import annotations

from fastapi import FastAPI

from document_analyzer.api.router import router
from document_analyzer.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.include_router(router)
    return app


app = create_app()
