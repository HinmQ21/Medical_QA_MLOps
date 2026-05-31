"""Deterministic backend for tests and CI (no GPU, no network)."""

from .base import ModelBackend


class MockBackend(ModelBackend):
    name = "mock"

    def __init__(self, answer: str = "A"):
        self._answer = answer.upper()

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        return (
            f"<think>Mock reasoning for {len(messages)} messages.</think>"
            f"<answer>{self._answer}</answer>"
        )
