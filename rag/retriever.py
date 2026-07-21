"""
FinBot — rag/retriever.py
=========================
Loads ChromaDB vectorstore and provides retrieval with
cross-encoder reranking for improved result quality.

Supports merging results from the global knowledge base with
a session-scoped collection (uploaded documents).

Usage:
    from rag.retriever import load_vectorstore, retrieve_and_rerank
"""

import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "finbot_knowledge")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cached cross-encoder reranker instance (loaded once, reused across requests)
_reranker = None


def get_reranker():
    """
    Return the cached CrossEncoder reranker, loading it on first call.

    Returns:
        CrossEncoder instance
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder reranker model: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Cross-encoder reranker loaded and cached")
    return _reranker


# ============================================================
# LOAD VECTORSTORE
# ============================================================

def load_vectorstore():
    """
    Load existing ChromaDB vectorstore from disk.
    Must run rag/ingest.py first to populate the database.

    Returns:
        Chroma vectorstore object

    Raises:
        RuntimeError: If ChromaDB is empty or not found
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    import chromadb

    # Check ChromaDB exists
    if not Path(CHROMA_DB_PATH).exists():
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_DB_PATH}\n"
            f"Run: python rag/ingest.py"
        )

    logger.info("Loading ChromaDB from: %s", CHROMA_DB_PATH)

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Load vectorstore
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    # Verify it has data
    count = vectorstore._collection.count()
    if count == 0:
        raise RuntimeError(
            f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' is empty!\n"
            f"Run: python rag/ingest.py"
        )

    logger.info("ChromaDB loaded — %d chunks available", count)
    return vectorstore


def _get_session_vectorstore(session_id: str):
    """
    Load a session-scoped Chroma collection if it exists.

    Args:
        session_id: The session identifier

    Returns:
        Chroma vectorstore or None if no session collection exists
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    import chromadb

    collection_name = f"session_{session_id}"
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        col = client.get_collection(collection_name)
        if col.count() == 0:
            return None
    except Exception:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )


# ============================================================
# BASIC RETRIEVER
# ============================================================

def get_retriever(vectorstore, k: int = 6):
    """
    Get a basic ChromaDB retriever with top-k similarity search.

    Args:
        vectorstore: Chroma vectorstore object
        k: Number of documents to retrieve

    Returns:
        LangChain retriever object
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever


# ============================================================
# CROSS-ENCODER RERANKER
# ============================================================

def rerank_documents(
    query: str,
    documents: list,
    top_n: int = 4
) -> List[Tuple]:
    """
    Rerank retrieved documents using cross-encoder model.
    Cross-encoder reads query + document together for better scoring.

    Args:
        query: User's financial question
        documents: List of retrieved Document objects
        top_n: Number of top documents to return after reranking

    Returns:
        List of (Document, score) tuples sorted by relevance
    """
    logger.info("Reranking %d documents with cross-encoder...", len(documents))

    # Use cached cross-encoder model
    reranker = get_reranker()

    # Create query-document pairs for scoring
    pairs = [[query, doc.page_content] for doc in documents]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Combine documents with scores
    scored_docs = list(zip(documents, scores))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Take top_n
    top_docs = scored_docs[:top_n]

    # Add rerank score to metadata
    for doc, score in top_docs:
        doc.metadata["rerank_score"] = float(score)

    logger.info("Reranking complete — top %d docs selected", top_n)
    for i, (doc, score) in enumerate(top_docs):
        logger.info(
            "  Rank %d: score=%.3f | %s p.%s",
            i + 1, score,
            doc.metadata.get("file_name", "?"),
            doc.metadata.get("page_number", "?"),
        )

    return top_docs


# ============================================================
# FULL RETRIEVE + RERANK PIPELINE
# ============================================================

def retrieve_and_rerank(
    query: str,
    vectorstore,
    k: int = 6,
    top_n: int = 4,
    session_id: Optional[str] = None,
) -> list:
    """
    Full retrieval pipeline:
    Step 1 — Retrieve top-k chunks from global + session ChromaDB
    Step 2 — Rerank with cross-encoder
    Step 3 — Return top_n most relevant documents

    Args:
        query: User's financial question
        vectorstore: Global Chroma vectorstore object
        k: Initial retrieval count per collection
        top_n: Final number of documents after reranking
        session_id: Optional session ID to also search session collection

    Returns:
        List of top Document objects with rerank_score in metadata
    """
    logger.info("Retrieving documents for: '%s'...", query[:80])

    # Step 1: Retrieve from global collection
    retriever = get_retriever(vectorstore, k=k)
    initial_docs = retriever.invoke(query)
    logger.info("Retrieved %d documents from global collection", len(initial_docs))

    # Also retrieve from session collection if it exists
    if session_id:
        session_vs = _get_session_vectorstore(session_id)
        if session_vs:
            session_retriever = get_retriever(session_vs, k=k)
            session_docs = session_retriever.invoke(query)
            logger.info(
                "Retrieved %d documents from session collection '%s'",
                len(session_docs), session_id,
            )
            # Mark session docs for source tracking
            for doc in session_docs:
                doc.metadata["source"] = "uploaded_document"
            initial_docs.extend(session_docs)

    if not initial_docs:
        logger.warning("No documents retrieved from ChromaDB!")
        return []

    # Step 2: Rerank all documents together
    ranked_docs = rerank_documents(query, initial_docs, top_n=top_n)

    # Return just the documents (without scores)
    return [doc for doc, score in ranked_docs]


# ============================================================
# ENTRY POINT — Quick test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print("\n=== Testing FinBot Retriever ===\n")

    # Load vectorstore
    vs = load_vectorstore()

    # Test query
    test_query = "What is the 80C deduction limit for income tax?"
    print(f"Query: {test_query}\n")

    # Retrieve and rerank
    results = retrieve_and_rerank(test_query, vs, k=6, top_n=4)

    print(f"\nTop {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"File    : {doc.metadata.get('file_name', 'unknown')}")
        print(f"Page    : {doc.metadata.get('page_number', '?')}")
        print(f"Score   : {doc.metadata.get('rerank_score', 0):.3f}")
        print(f"Preview : {doc.page_content[:200]}...")
