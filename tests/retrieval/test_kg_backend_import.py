def test_kg_backend_symbol_exists():
    from medical_qa_platform.retrieval import kg_backend

    assert hasattr(kg_backend, "KGRetrieval")
