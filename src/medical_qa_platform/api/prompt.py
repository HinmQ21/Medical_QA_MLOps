"""Build chat messages for the medical MCQ task.

SYSTEM_PROMPT is kept verbatim in sync with the canonical training/eval prompt
(the reference data-prep SYSTEM_PROMPT; see docs/specs/2026-06-03-retrieval-contract-sync-design.md)
so the model sees at serving time the exact system message it was trained
against. The structural difference between single-shot evidence injection here
and the agentic tool round-trip used in training is tracked as a separate
sub-project.
"""

SYSTEM_PROMPT = (
    "You are a medical reasoning assistant with access to a "
    "search_medical_knowledge tool.\n\n"
    "Structure your response in this order:\n"
    "1. <think>Your initial reasoning about the question</think>\n"
    "2. (Optional) If you need to verify a medical fact, call "
    "search_medical_knowledge, then add another <think>...</think> "
    "incorporating the result.\n"
    "3. <answer>Your final answer</answer>\n\n"
    "IMPORTANT: In <answer> tags, write ONLY the option letter (e.g. A) "
    "or a short answer, NOT an explanation."
)


def build_prompt(
    question: str,
    evidence: list[str],
) -> list[dict]:
    """Return OpenAI-style chat messages for the given question.

    The question is free-text and already contains any answer options inline,
    so no separate options block is rendered.
    """
    lines = []
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"- {item}" for item in evidence)
        lines.append("")
    lines.append(f"Question: {question}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]
