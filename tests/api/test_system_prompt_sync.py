from medical_qa_platform.api.prompt import SYSTEM_PROMPT

CANONICAL = (
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


def test_system_prompt_matches_canonical_baseline():
    assert SYSTEM_PROMPT == CANONICAL


def test_build_prompt_renders_evidence_via_contract_format():
    """Lock the api's evidence rendering to the Layer-2 contract formatter."""
    from medical_qa_platform.api.prompt import build_prompt
    from medical_qa_platform.retrieval.contract import format_evidence

    msgs = build_prompt("Q?", ["fact one", "fact two"])
    user = msgs[1]["content"]
    assert format_evidence(["fact one", "fact two"]) in user
