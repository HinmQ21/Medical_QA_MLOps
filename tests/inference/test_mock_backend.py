import pytest

from medical_qa_platform.inference import get_backend
from medical_qa_platform.inference.base import ModelBackend
from medical_qa_platform.inference.mock_backend import MockBackend


def test_mock_is_a_model_backend():
    assert isinstance(MockBackend(), ModelBackend)


def test_mock_name():
    assert MockBackend().name == "mock"


def test_mock_generate_is_deterministic():
    backend = MockBackend(answer="C")
    out1 = backend.generate([{"role": "user", "content": "x"}])
    out2 = backend.generate([{"role": "user", "content": "x"}])
    assert out1 == out2
    assert "<answer>C</answer>" in out1


def test_factory_returns_mock_by_name():
    assert isinstance(get_backend("mock"), MockBackend)


def test_factory_reads_env(monkeypatch):
    monkeypatch.setenv("MODEL_BACKEND", "mock")
    assert isinstance(get_backend(), MockBackend)


def test_factory_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("nope")


def test_factory_returns_vllm_by_name():
    from medical_qa_platform.inference.vllm_backend import VllmBackend

    assert isinstance(get_backend("vllm"), VllmBackend)


def test_factory_no_longer_knows_runpod():
    with pytest.raises(ValueError):
        get_backend("runpod")


def test_factory_no_longer_knows_kserve():
    with pytest.raises(ValueError):
        get_backend("kserve")
