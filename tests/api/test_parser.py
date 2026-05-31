from medical_qa_platform.api.parser import parse_answer


def test_parses_answer_tag():
    assert parse_answer("<think>x</think><answer>B</answer>") == "B"


def test_parses_lowercase_and_strips():
    assert parse_answer("blah <answer> c </answer> blah") == "C"


def test_fallback_answer_is_phrase():
    assert parse_answer("The answer is D.") == "D"


def test_fallback_answer_colon():
    assert parse_answer("Answer: A") == "A"


def test_returns_none_when_absent():
    assert parse_answer("no letter here") is None


def test_respects_valid_letters():
    assert parse_answer("<answer>F</answer>", valid_letters={"A", "B"}) is None


def test_prefers_tag_over_fallback_phrase():
    assert parse_answer("The answer is A <answer>B</answer>") == "B"
