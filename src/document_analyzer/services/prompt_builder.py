from __future__ import annotations

from document_analyzer.services.together_client import TogetherChatService


class PromptBuilder:
    """Rewrites raw user input into a knowledge-base-friendly query via the LLM."""

    _REWRITE_SYSTEM_PROMPT = """\
You are a query reformulation assistant for a document analysis system.

Your task is to transform the user's raw input into a search-friendly query.

Steps:
1. Normalize
   - lowercase the text
   - trim extra whitespace
   - remove filler / noise words that do not change meaning
2. Detect intent (one of: question, summary, comparison, lookup)
3. Reformulate the input into a concise knowledge-base search query
4. Expand with relevant synonyms and domain keywords

Return **only** valid JSON in this exact schema – no markdown, no explanation:
{
  "normalized_text": "...",
  "intent": "question | summary | comparison | lookup",
  "reformulated_query": "...",
  "expanded_keywords": ["...", "..."]
}

Example
-------
Input:  What does this document say about revenue?
Output:
{
  "normalized_text": "what does this document say about revenue",
  "intent": "question",
  "reformulated_query": "financial performance revenue growth earnings results",
  "expanded_keywords": ["revenue", "earnings", "income", "financial results", "growth"]
}
"""

    def __init__(self, service: TogetherChatService) -> None:
        self.service = service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(self, user_prompt: str) -> str:
        """Send the raw user prompt through the LLM and return the full
        JSON reformulation (normalized text, intent, query, keywords)."""
        cleaned = self._pre_clean(user_prompt)
        response = self.service.ask(
            prompt=cleaned,
            system_prompt=self._REWRITE_SYSTEM_PROMPT,
        )
        return response.answer

    def rewrite_query_only(self, user_prompt: str) -> str:
        """Return **only** the reformulated query string (no JSON)."""
        import json

        raw = self.rewrite(user_prompt)
        try:
            data = json.loads(self._extract_json_payload(raw))
            print("DEBUG: Parsed LLM response:", data)
            return data.get("reformulated_query", raw)
        except (json.JSONDecodeError, TypeError):
            # LLM did not return valid JSON – fall back to raw response
            return raw

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _pre_clean(text: str) -> str:
        import re

        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _extract_json_payload(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped