from __future__ import annotations

import logging
import re
from document_analyzer.services.together_client import TogetherChatService

logger = logging.getLogger(__name__)


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
4. Expand with only obvious synonyms or closely related search terms.
Do not add facts, entities, dates, names, or claims not present in the user's input.

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

    _Base_Prompt = """You are a query reformulation assistant for a document analysis system.
    
    
    
    """

    def __init__(self, service: TogetherChatService) -> None:
        self.service = service
        self._tokenizer = None
        self._tokenizer_name = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(self, user_prompt: str) -> str:
        cleaned = self._pre_clean(user_prompt)
        wrapped = f"### USER INPUT ###\n{cleaned}\n###"
        response = self.service.ask(
        prompt=wrapped,
        system_prompt=self._REWRITE_SYSTEM_PROMPT,
        )
        return response.answer

    def rewrite_query_only(self, user_prompt: str) -> str:
        """Return **only** the reformulated query string (no JSON)."""
        import json

        raw = self.rewrite(user_prompt)
        try:
            data = json.loads(self._extract_json_payload(raw))
            
            return data.get("reformulated_query", raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                 "LLM returned non-JSON response (possible injection or model error). "
                 "Falling back to raw response. First 200 chars: %.200s", raw
             )
            return raw
    
    def context_builder(self, user_prompt: str, search_response: dict) -> dict[str, str]:
        """Build a context string for the LLM by combining the reformulated query
        with the original user prompt."""
        rewritten = self.rewrite_query_only(user_prompt)
        context = f"User Prompt: {user_prompt}\nReformulated Query: {rewritten}"

        
        if search_response:
            context += "\nSearch Results:\n"
            for i, result in enumerate(search_response.get("results", []), 1):
                context += f"""Source {i}.\n
                id: {result.get('id', '')}\n
                metadata: {result.get('metadata', {})}\n
                score: {result.get('score', 0.0)}\n 
                context: {result.get('document', '')}\n\n
                """
                
        tokens = self.token_count(context)
        logger.info(f"Built context with {tokens} tokens:\n{context}")
        return {"context": context}
    
    def system_prompt_builder(self, instructions: str | None) -> str:
        base_prompt = """You are an assistant for a document analysis system. \n
        Answer the user's question based on the provided context from relevant documents.
        If the context does not contain enough information, say you don't know rather than making up an answer. \n
        Always use all available context to provide the best possible answer."""
        if instructions:
            return f"{base_prompt}\n\nAdditional instructions: {instructions}"
        return base_prompt 
    

    
    def token_count(self, text: str) -> int:
        """Best-effort token count without requiring network access.

        Together model IDs are not always valid Hugging Face repo IDs, so this
        method falls back to an approximate count when no local tokenizer exists.
        """
        tokenizer = self._get_local_tokenizer()
        if tokenizer is None:
            # Approximation used when tokenizer files are unavailable locally.
            # Typical English text is roughly 4 chars/token for BPE tokenizers.
            return max(1, (len(text) + 3) // 4)
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _get_local_tokenizer(self):  # noqa: ANN202
        from transformers import AutoTokenizer

        model_name = getattr(self.service, "_default_model", "")
        if self._tokenizer is not None and model_name == self._tokenizer_name:
            return self._tokenizer

        for candidate in (model_name, "gpt2"):
            if not candidate:
                continue
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    candidate,
                    local_files_only=True,
                )
                self._tokenizer = tokenizer
                self._tokenizer_name = model_name
                return tokenizer
            except Exception:
                continue

        logger.warning(
            "No local tokenizer found for model '%s'; using approximate token count.",
            model_name,
        )
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _pre_clean(text: str) -> str:
        
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        injection_pattern = re.compile(
            r"(ignore|disregard|forget|override)\s+(all\s+)?(previous|prior|above|system)?\s*(instructions?|prompt|context|rules?)",
            re.IGNORECASE,
        )
        if injection_pattern.search(text):
            logger.warning("Possible prompt injection attempt detected in input: %.100s", text)
            text = injection_pattern.sub("[removed]", text)
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