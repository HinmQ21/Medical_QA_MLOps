from medical_qa_platform.retrieval import contract


def test_format_evidence_empty_returns_sentinel():
    assert contract.format_evidence([]) == "No relevant knowledge found."


def test_format_evidence_joins_with_dash_prefix():
    assert contract.format_evidence(["a fact", "b fact"]) == "- a fact\n- b fact"


def test_contract_version_is_v1_medembed_large():
    assert contract.RETRIEVAL_CONTRACT_VERSION == "v1-medembed-large"
