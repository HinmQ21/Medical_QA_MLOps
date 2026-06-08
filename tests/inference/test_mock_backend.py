import pytest

from medical_qa_platform.inference import get_backend
from medical_qa_platform.inference.base import ChatTurn, ModelBackend
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


def test_factory_returns_llm_by_name():
    from medical_qa_platform.inference.llm_backend import LLMBackend

    assert isinstance(get_backend("llm"), LLMBackend)


def test_factory_vllm_alias_maps_to_llm():
    from medical_qa_platform.inference.llm_backend import LLMBackend

    # "vllm" is kept as a back-compat alias for the generic LLM/OpenAI backend.
    assert isinstance(get_backend("vllm"), LLMBackend)


def test_factory_no_longer_knows_runpod():
    with pytest.raises(ValueError):
        get_backend("runpod")


def test_factory_no_longer_knows_kserve():
    with pytest.raises(ValueError):
        get_backend("kserve")


def test_mock_chat_returns_chatturn():
    turn = MockBackend(answer="C").chat([{"role": "user", "content": "x"}])
    assert isinstance(turn, ChatTurn)
    assert "<answer>C</answer>" in (turn.content or "")
    assert turn.tool_calls == []
    assert turn.finish_reason == "stop"
