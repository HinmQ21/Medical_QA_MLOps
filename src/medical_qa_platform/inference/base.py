"""Model backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChatTurn:
    """One assistant turn from a chat-completions call.

    ``tool_calls`` holds the raw OpenAI tool-call objects
    (``{"id", "type", "function": {"name", "arguments"}}``) so the agentic loop
    can echo them back verbatim in the next request.
    """

    content: str | None
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"


class ModelBackend(ABC):
    """A backend that turns chat messages into an assistant turn."""

    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> ChatTurn:
        """Return the assistant turn (content + any tool calls) for these messages."""
        raise NotImplementedError

    def generate(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Convenience wrapper: return just the assistant text (no tools)."""
        return (
            self.chat(messages, max_tokens=max_tokens, temperature=temperature).content
            or ""
        )

    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready to serve.

        Default: assume healthy (mock/in-process backends never go down).
        Network backends override this with a real probe.
        """
        return True
