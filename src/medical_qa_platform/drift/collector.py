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
    ) -> dict:
        option_lengths = [len(value) for value in request.options.values()]
        row = {
            "q_token_len": len(request.question.split()),
            "n_options": len(request.options),
            "mean_option_len": sum(option_lengths) / len(option_lengths),
            "answer": response.answer if response.answer is not None else "none",
            "no_result": n_evidence == 0,
            "latency_ms": response.latency_ms,
        }
        if self.path is not None:
            with open(self.path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
        return row
