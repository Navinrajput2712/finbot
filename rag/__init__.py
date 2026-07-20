"""
FinBot — rag package
====================
RAG pipeline components:
- ingest: PDF loading, chunking, ChromaDB storage
- retriever: ChromaDB retrieval + cross-encoder reranking
- pipeline: Full RAG chain with NVIDIA NIM LLM
"""

from rag.ingest import load_documents, split_documents, create_vectorstore
from rag.retriever import load_vectorstore, retrieve_and_rerank, get_reranker
from rag.pipeline import ask_finbot, is_market_query

__all__ = [
    "load_documents",
    "split_documents",
    "create_vectorstore",
    "load_vectorstore",
    "retrieve_and_rerank",
    "get_reranker",
    "ask_finbot",
    "is_market_query",
]
