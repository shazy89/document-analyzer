from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, tool

from general_agent.config import GeneralAgentConfig

logger = logging.getLogger(__name__)


class GeneralAgentTools:
    """Central registry for tools used by the general agent."""

    def __init__(self, config: GeneralAgentConfig) -> None:
        self.config = config
        self.tools: dict[str, BaseTool] = {}

    def register_tool(self, func: callable) -> BaseTool:
        """Wrap a plain function as a LangChain tool and register it."""
        wrapped = func if isinstance(func, BaseTool) else tool(func)
        self.tools[wrapped.name] = wrapped
        logger.info("Registered tool: %s", wrapped.name)
        return wrapped

    # ------------------------------------------------------------------
    # Example placeholder tool — replace or extend as needed
    # ------------------------------------------------------------------

    def echo(self, text: str) -> str:
        """Return the input text unchanged. Useful for smoke-testing the tool loop."""
        return text
