"""Model backend interface."""

from abc import ABC, abstractmethod


class ModelBackend(ABC):
    """A backend that turns chat messages into raw model text."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Return the raw text completion for the given chat messages."""
        raise NotImplementedError
