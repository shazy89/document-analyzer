from __future__ import annotations

import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logging for the application.

    Call once at startup (in ``create_app``).  ``force=True`` overrides any
    prior config applied by uvicorn or third-party imports.
    """
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
