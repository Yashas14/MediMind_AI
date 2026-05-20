"""
ChromaDB vector store service.

Provides an async-compatible wrapper around ChromaDB for storing and
retrieving medical knowledge embeddings. Supports both local (persistent)
and remote (HTTP client) modes.
"""

from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

# Module-level client and collection (lazy init)
_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Get or create the ChromaDB client.

    Uses HTTP mode in production (connects to ChromaDB container)
    and persistent local mode for development.

    Returns:
        ChromaDB client instance.
    """
    global _chroma_client  # noqa: PLW0603
    if _chroma_client is None:
        try:
            _chroma_client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            # Verify connection
            _chroma_client.heartbeat()
            logger.info(
                "Connected to ChromaDB at %s:%s",
                settings.chroma_host,
                settings.chroma_port,
            )
        except Exception as exc:
            logger.warning(
                "Cannot connect to ChromaDB server (%s). "
                "Falling back to local persistent mode: %s",
                exc, exc,
            )
            _chroma_client = chromadb.PersistentClient(
                path="./chroma_data",
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info("Using local persistent ChromaDB at ./chroma_data")

    return _chroma_client


def get_collection() -> chromadb.Collection:
    """Get or create the medical knowledge collection.

    Returns:
        ChromaDB collection for medical knowledge.
    """
    global _collection  # noqa: PLW0603
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={
                "description": "Medical knowledge base for healthcare AI platform",
                "hnsw:space": "cosine",  # Use cosine similarity
            },
        )
        logger.info(
            "Collection '%s' ready (%d documents)",
            settings.chroma_collection,
            _collection.count(),
        )
    return _collection


class VectorStoreService:
    """High-level interface for medical knowledge vector operations.

    Provides methods for:
    - Ingesting structured documents into ChromaDB
    - Querying for relevant medical context
    - Managing the vector store lifecycle
    """

    def __init__(self) -> None:
        self._collection: chromadb.Collection | None = None

    @property
    def collection(self) -> chromadb.Collection:
        """Lazy-initialise and return the ChromaDB collection."""
        if self._collection is None:
            self._collection = get_collection()
        return self._collection

    def ingest_documents(
        self,
        documents: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Ingest a list of documents into the vector store.

        Documents are upserted (insert or update) to avoid duplicates.

        Args:
            documents: List of dicts with ``id``, ``text``, and ``metadata``.
            batch_size: Number of documents to upsert per batch.

        Returns:
            Total number of documents ingested.
        """
        if not documents:
            logger.warning("No documents to ingest")
            return 0

        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            metadatas = [doc.get("metadata", {}) for doc in batch]

            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.debug("Ingested batch %d–%d", i, i + len(batch))

        logger.info("Ingested %d documents into '%s'", total, self.collection.name)
        return total

    async def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector store for relevant documents.

        ChromaDB's query is synchronous but fast — we wrap it for
        consistency with the async API layer.

        Args:
            query_text: Natural language query string.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter (e.g., ``{"type": "disease_description"}``).

        Returns:
            List of result dicts with ``content``, ``source``, ``distance``, ``metadata``.
        """
        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query_text],
                "n_results": top_k,
            }
            if filter_metadata:
                kwargs["where"] = filter_metadata

            results = self.collection.query(**kwargs)

            if not results or not results.get("documents"):
                return []

            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(documents)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(documents)

            return [
                {
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "disease": meta.get("disease", ""),
                    "type": meta.get("type", ""),
                    "distance": dist,
                    "metadata": meta,
                }
                for doc, meta, dist in zip(documents, metadatas, distances)
            ]

        except Exception as exc:
            logger.error("Vector store query failed: %s", exc)
            return []

    async def query_by_symptoms(
        self,
        symptoms: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query using symptom names for disease-related context.

        Args:
            symptoms: List of canonical symptom names.
            top_k: Number of results.

        Returns:
            Relevant medical context documents.
        """
        query = " ".join(s.replace("_", " ") for s in symptoms)
        return await self.query(query, top_k=top_k)

    async def query_by_disease(
        self,
        disease: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Query for information about a specific disease.

        Args:
            disease: Disease name.
            top_k: Number of results.

        Returns:
            Documents about the disease.
        """
        return await self.query(
            f"Disease: {disease}",
            top_k=top_k,
            filter_metadata={"type": "disease_description"},
        )

    def get_stats(self) -> dict[str, Any]:
        """Return collection statistics.

        Returns:
            Dict with document count and collection metadata.
        """
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "metadata": self.collection.metadata,
        }

    def clear(self) -> None:
        """Delete all documents from the collection.

        Use with caution — this is irreversible.
        """
        client = get_chroma_client()
        client.delete_collection(self.collection.name)
        self._collection = None
        logger.warning("Cleared collection '%s'", settings.chroma_collection)
