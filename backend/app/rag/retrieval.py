"""
RAG retrieval service — unified interface for the retrieval pipeline.

Combines vector store retrieval with optional re-ranking and context
formatting for downstream agent consumption.
"""

from typing import Any

from app.core.logging import get_logger
from app.rag.vector_store import VectorStoreService

logger = get_logger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service.

    Wraps the vector store with higher-level retrieval strategies:
    - Symptom-based retrieval
    - Disease-specific retrieval
    - Multi-query expansion
    - Context deduplication and ranking
    """

    def __init__(self, vector_store: VectorStoreService | None = None) -> None:
        self._vector_store = vector_store or VectorStoreService()

    @property
    def vector_store(self) -> VectorStoreService:
        """Access the underlying vector store."""
        return self._vector_store

    async def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Run a semantic search and return formatted results.

        This is the method consumed by the DiagnosisAgent and other agents.

        Args:
            query_text: Free-text query.
            top_k: Number of results.

        Returns:
            List of result dicts with ``content``, ``source``, etc.
        """
        return await self._vector_store.query(query_text, top_k=top_k)

    def _format_context(self, results: list[dict[str, Any]]) -> str:
        """Format retrieval results into a single context string.

        Args:
            results: List of retrieval result dicts.

        Returns:
            Formatted multi-section context string.
        """
        if not results:
            return "No relevant medical context found."

        sections: list[str] = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            source = result.get("source", "unknown")
            doc_type = result.get("type", "")

            sections.append(
                f"--- Context {i} [{source}] ({doc_type}) ---\n{content}"
            )

        return "\n\n".join(sections)
