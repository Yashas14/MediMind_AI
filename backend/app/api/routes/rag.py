"""
RAG pipeline admin endpoints.

Provides endpoints for managing the medical knowledge base:
- Ingest CSV data into the vector store
- Query the knowledge base
- Get vector store statistics
"""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import AdminUser, CurrentUser
from app.core.logging import get_logger
from app.rag.retrieval import RAGService
from app.rag.vector_store import VectorStoreService
from app.services.data_loader import ingest_to_vector_store

router = APIRouter(prefix="/rag", tags=["RAG Knowledge Base"])
logger = get_logger(__name__)


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query."""

    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    filter_type: str | None = Field(
        default=None,
        description="Filter by document type (disease_description, precautions, etc.)",
    )


class RAGQueryResult(BaseModel):
    """Schema for a single retrieval result."""

    content: str
    source: str
    disease: str
    doc_type: str
    distance: float


class RAGQueryResponse(BaseModel):
    """Response schema for RAG query."""

    results: list[RAGQueryResult]
    total: int
    query: str


class RAGStatsResponse(BaseModel):
    """Response schema for vector store stats."""

    collection_name: str
    document_count: int
    pipeline: str


# ── Endpoints ───────────────────────────────────────────────────────

@router.post("/ingest", summary="Ingest medical data into vector store")
async def ingest_data(admin: AdminUser) -> dict[str, Any]:
    """Trigger ingestion of all medical CSV data into ChromaDB.

    Reads Training.csv, symptom descriptions, precautions, and severity
    data, then upserts them as vector embeddings.
    """
    logger.info("Manual RAG ingestion triggered")
    try:
        result = ingest_to_vector_store()
        return result
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Query the medical knowledge base",
)
async def query_knowledge_base(
    body: RAGQueryRequest,
    user: CurrentUser,
) -> RAGQueryResponse:
    """Search the medical knowledge vector store.

    Returns the most similar documents to the query text.
    """
    rag = RAGService(VectorStoreService())

    filter_meta = None
    if body.filter_type:
        filter_meta = {"type": body.filter_type}

    results = await rag.vector_store.query(
        body.query,
        top_k=body.top_k,
        filter_metadata=filter_meta,
    )

    return RAGQueryResponse(
        results=[
            RAGQueryResult(
                content=r.get("content", "")[:500],
                source=r.get("source", "unknown"),
                disease=r.get("disease", ""),
                doc_type=r.get("type", ""),
                distance=r.get("distance", 0.0),
            )
            for r in results
        ],
        total=len(results),
        query=body.query,
    )


@router.get(
    "/stats",
    response_model=RAGStatsResponse,
    summary="Get vector store statistics",
)
async def get_rag_stats(user: CurrentUser) -> RAGStatsResponse:
    """Return statistics about the medical knowledge vector store."""
    try:
        vs = VectorStoreService()
        stats = vs.get_stats()
        return RAGStatsResponse(
            collection_name=stats["collection_name"],
            document_count=stats["document_count"],
            pipeline="ChromaDB + cosine similarity",
        )
    except Exception as exc:
        logger.error("Failed to get RAG stats: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {exc}",
        )
