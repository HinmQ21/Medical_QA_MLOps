"""Retrieval backend interface + a deterministic fixture for tests."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsSearch(Protocol):
    """Anything the API can call to fetch evidence."""

    def search(self, query: str, top_k: int) -> list[str]: ...


class RetrievalBackend(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int) -> list[str]:
        raise NotImplementedError


class FixtureRetrieval(RetrievalBackend):
    """In-memory query to results map."""

    def __init__(self, data: dict[str, list[str]]):
        self._data = data

    def search(self, query: str, top_k: int) -> list[str]:
        return list(self._data.get(query, []))[:top_k]
