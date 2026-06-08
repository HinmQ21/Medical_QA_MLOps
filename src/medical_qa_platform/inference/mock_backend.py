"""Deterministic backend for tests and CI (no GPU, no network)."""

from .base import ChatTurn, ModelBackend


class MockBackend(ModelBackend):
    name = "mock"

    def __init__(self, answer: str = "A"):
        self._answer = answer.upper()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> ChatTurn:
        return ChatTurn(
            content=(
                f"<think>Mock reasoning for {len(messages)} messages.</think>"
                f"<answer>{self._answer}</answer>"
            ),
            tool_calls=[],
            finish_reason="stop",
        )
