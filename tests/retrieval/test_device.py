from medical_qa_platform.retrieval.device import resolve_device


def test_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_DEVICE", raising=False)
    assert resolve_device() == "cpu"


def test_reads_env(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DEVICE", "cuda")
    assert resolve_device() == "cuda"


def test_explicit_arg_wins(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_DEVICE", "cuda")
    assert resolve_device("cpu") == "cpu"
