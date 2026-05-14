from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_TRACE_FILE = Path("eval_traces.jsonl")


class EvaluationService:
    """Logs RAG pipeline traces to a JSONL file for offline evaluation."""

    def __init__(self, trace_file: Path = _DEFAULT_TRACE_FILE) -> None:
        self.trace_file = trace_file

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_trace(
        self,
        *,
        question: str,
        rewritten_query: str,
        contexts: list[str],
        answer: str,
        model: str,
        ground_truth: str | None = None,
    ) -> None:
        """Append a single RAG trace to the JSONL file.

        Each line is a self-contained JSON object with all fields needed
        to run RAGAS metrics (faithfulness, answer relevance, context
        precision, context recall).
        """
        trace = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "question": question,
            "rewritten_query": rewritten_query,
            "contexts": contexts,
            "answer": answer,
            "model": model,
        }
        if ground_truth is not None:
            trace["ground_truth"] = ground_truth

        try:
            with self.trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace) + "\n")
        except OSError:
            logger.exception("Failed to write eval trace to %s", self.trace_file)

    def load_traces(self) -> list[dict]:
        """Read all traces from the JSONL file."""
        if not self.trace_file.exists():
            return []
        traces = []
        for line in self.trace_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed trace line: %.80s", line)
        return traces
