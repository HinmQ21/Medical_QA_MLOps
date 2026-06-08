"""Run a deterministic smoke evaluation and write metrics/artifacts."""

import argparse
import json
import time
from pathlib import Path

from medical_qa_platform.api.parser import parse_answer
from medical_qa_platform.api.prompt import build_prompt
from medical_qa_platform.inference.mock_backend import MockBackend
from medical_qa_platform.retrieval.backends import FixtureRetrieval

from .profiles import PipelineProfile, load_profile


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def evaluate_smoke(profile: PipelineProfile) -> dict:
    data_path = profile.data_dir / "medical_mcq.jsonl"
    examples = _read_jsonl(data_path)
    fixture = json.loads(profile.retrieval_fixture_path.read_text())
    retrieval = FixtureRetrieval(fixture)

    predictions = []
    correct = 0
    no_result = 0
    latencies_ms = []

    for example in examples:
        started = time.perf_counter()
        evidence = retrieval.search(example["retrieval_query"], profile.top_k)
        backend = MockBackend(answer=example["mock_answer"])
        messages = build_prompt(example["question"], evidence)
        raw = backend.generate(messages)
        predicted = parse_answer(raw)
        latency_ms = (time.perf_counter() - started) * 1000.0
        is_correct = predicted == example["answer"]
        correct += int(is_correct)
        no_result += int(len(evidence) == 0)
        latencies_ms.append(latency_ms)
        predictions.append(
            {
                "id": example["id"],
                "question": example["question"],
                "expected_answer": example["answer"],
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "evidence": evidence,
                "latency_ms": latency_ms,
                "model_version": profile.model_version,
            }
        )

    profile.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    profile.predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions)
    )

    n_examples = len(examples)
    metrics = {
        "profile": profile.name,
        "model_backend": profile.model_backend,
        "model_version": profile.model_version,
        "n_examples": n_examples,
        "accuracy": correct / n_examples if n_examples else 0.0,
        "retrieval_no_result_rate": no_result / n_examples if n_examples else 0.0,
        "latency_ms_avg": sum(latencies_ms) / n_examples if n_examples else 0.0,
    }
    _write_json(profile.metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke")
    args = parser.parse_args()
    evaluate_smoke(load_profile(args.profile))


if __name__ == "__main__":
    main()
