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

    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready to serve.

        Default: assume healthy (mock/in-process backends never go down).
        Network backends override this with a real probe.
        """
        return True
