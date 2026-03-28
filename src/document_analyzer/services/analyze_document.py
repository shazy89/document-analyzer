from __future__ import annotations

from fastapi import Depends

from document_analyzer.services.prompt_builder import PromptBuilder


class AnalyzeDocumentService:
    def __init__(self, prompt_builder: PromptBuilder = Depends(PromptBuilder)) -> None:
        self.prompt_builder = prompt_builder

    def analyze(self, prompt: str) -> str:
        return self.prompt_builder.rewrite_query_only(prompt)