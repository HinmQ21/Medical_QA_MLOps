from medical_qa_platform.api.prompt import SYSTEM_PROMPT, build_prompt


def test_returns_system_and_user_messages():
    msgs = build_prompt("Q?", {"A": "opt a", "B": "opt b"}, [])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1]["role"] == "user"
    assert len(msgs) == 2


def test_user_message_contains_question_and_options():
    msgs = build_prompt("What is X?", {"A": "alpha", "B": "beta"}, [])
    user = msgs[1]["content"]
    assert "What is X?" in user
    assert "A. alpha" in user
    assert "B. beta" in user


def test_evidence_block_included_when_present():
    msgs = build_prompt("Q?", {"A": "a", "B": "b"}, ["fact one", "fact two"])
    user = msgs[1]["content"]
    assert "fact one" in user
    assert "fact two" in user


def test_no_evidence_block_when_empty():
    msgs = build_prompt("Q?", {"A": "a", "B": "b"}, [])
    assert "Evidence" not in msgs[1]["content"]


def test_options_sorted_by_letter():
    msgs = build_prompt("Q?", {"B": "beta", "A": "alpha"}, [])
    user = msgs[1]["content"]
    assert user.index("A. alpha") < user.index("B. beta")
