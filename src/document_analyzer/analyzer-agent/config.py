from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MAX_REVISIONS = 2

@dataclass
class DocumentAnalyzerConfig:
    model_name: str = "openai/gpt-oss-120b"
    api_key: str = field(default_factory=lambda: os.getenv("TOGETHER_API_KEY", ""))
    api_base: str = "https://api.together.xyz/v1"
    temperature: float = 0.0
    max_revisions: int = DEFAULT_MAX_REVISIONS
    search_results_per_query: int = 2
    interrupt_before: list[str] = field(default_factory=list)
    
    @classmethod
    def from_env(cls) -> DocumentAnalyzerConfig:
        return cls(
            model_name=os.getenv("TOGETHER_MODEL", "openai/gpt-oss-120b"),
            api_key=os.getenv("TOGETHER_API_KEY", ""),
            api_base=os.getenv("TOGETHER_API_BASE", "https://api.together.xyz/v1"),
            temperature=float(os.getenv("TOGETHER_TEMPERATURE", "0.0")),
            max_revisions=int(os.getenv("TOGETHER_MAX_REVISIONS", str(DEFAULT_MAX_REVISIONS))),
            search_results_per_query=int(os.getenv("TOGETHER_SEARCH_RESULTS_PER_QUERY", "2")),
            interrupt_before=os.getenv("TOGETHER_INTERRUPT_BEFORE", "").split(",") if os.getenv("TOGETHER_INTERRUPT_BEFORE") else [],
        )