from medical_qa_platform.retrieval.backends import FixtureRetrieval, RetrievalBackend


def test_fixture_is_a_retrieval_backend():
    assert isinstance(FixtureRetrieval({}), RetrievalBackend)


def test_fixture_returns_canned_results():
    fx = FixtureRetrieval({"diabetes": ["metformin treats diabetes"]})
    assert fx.search("diabetes", top_k=5) == ["metformin treats diabetes"]


def test_fixture_truncates_to_top_k():
    fx = FixtureRetrieval({"q": ["a", "b", "c"]})
    assert fx.search("q", top_k=2) == ["a", "b"]


def test_fixture_unknown_query_returns_empty():
    assert FixtureRetrieval({"q": ["a"]}).search("other", top_k=5) == []
