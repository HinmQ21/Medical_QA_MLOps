"""Build chat messages for the medical MCQ task.

NOTE: SYSTEM_PROMPT must be reconciled with the canonical eval prompt in
baseline/scripts/benchmark/grpo_eval/grpo_eval.py before the real-model demo
(plan 4). For mock/local testing the exact wording is not asserted.
"""

SYSTEM_PROMPT = (
    "You are a medical question-answering assistant. Read the question and "
    "options, reason step by step inside <think></think>, then give the single "
    "best option letter inside <answer></answer>."
)


def build_prompt(
    question: str,
    options: dict[str, str],
    evidence: list[str],
) -> list[dict]:
    """Return OpenAI-style chat messages for the given MCQ."""
    lines = []
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"- {item}" for item in evidence)
        lines.append("")
    lines.append(f"Question: {question}")
    for letter in sorted(options):
        lines.append(f"{letter}. {options[letter]}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
