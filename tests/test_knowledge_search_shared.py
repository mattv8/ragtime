"""Shared test fixtures for knowledge search tests."""


class FakeDoc:
    """Fake document for testing FAISS search."""

    def __init__(self, page_content: str, source: str) -> None:
        self.page_content = page_content
        self.metadata = {"source": source}


class FakeFaissDb:
    """Fake FAISS database for testing search operations."""

    def __init__(self, docs: list[FakeDoc]) -> None:
        self._docs = docs
        self.similarity_search_by_vector_calls: list[tuple[list[float], int]] = []
        self.max_marginal_relevance_search_by_vector_calls: list[tuple[list[float], int, int, float]] = []

    def similarity_search(self, query: str, k: int):
        return self._docs[:k]

    def max_marginal_relevance_search(self, query: str, k: int, fetch_k: int, lambda_mult: float):
        return self._docs[:k]

    def similarity_search_by_vector(self, embedding: list[float], k: int):
        self.similarity_search_by_vector_calls.append((embedding, k))
        return self._docs[:k]

    def max_marginal_relevance_search_by_vector(self, embedding: list[float], k: int, fetch_k: int, lambda_mult: float):
        self.max_marginal_relevance_search_by_vector_calls.append((embedding, k, fetch_k, lambda_mult))
        return self._docs[:k]


# Export with underscore variants for backward compatibility if needed
_FakeDoc = FakeDoc
_FakeFaissDb = FakeFaissDb

__all__ = ["FakeDoc", "FakeFaissDb", "_FakeDoc", "_FakeFaissDb"]
