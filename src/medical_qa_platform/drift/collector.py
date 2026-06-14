"""Collect per-request rows for offline drift analysis."""

import json

from ..api.schemas import PredictRequest, PredictResponse


class DriftCollector:
    def __init__(self, path: str | None):
        """If path is None, rows are computed but not persisted."""
        self.path = path

    def record(
        self,
        request: PredictRequest,
        response: PredictResponse,
        n_evidence: int,
        tool_call_count: int,
    ) -> dict:
        if tool_call_count == 0:
            outcome = "not_called"
        elif n_evidence == 0:
            outcome = "empty"
        else:
            outcome = "hit"
        row = {
            "q_token_len": len(request.question.split()),
            "answer": response.answer if response.answer is not None else "none",
            "tool_call_count": tool_call_count,
            "tool_called": tool_call_count > 0,
            "n_evidence": n_evidence,
            "tool_outcome": outcome,
            "latency_ms": response.latency_ms,
        }
        if self.path is not None:
            try:
                with open(self.path, "a") as handle:
                    handle.write(json.dumps(row) + "\n")
            except OSError:
                # Drift logging is best-effort observability; a write failure
                # (read-only/unwritable path) must never break the prediction.
                pass
        return row
