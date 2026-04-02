"""
FinBot — rag/ingest.py
======================
RAG Ingestion Pipeline — loads PDFs, chunks them, generates embeddings,
and stores everything in ChromaDB for retrieval.

Usage:
    python rag/ingest.py
    python rag/ingest.py --reset   # wipe and re-ingest
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION FROM .env
# ============================================================
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_base")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "finbot_knowledge")
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================
# STEP 1 — LOAD PDF DOCUMENTS
# ============================================================

def load_documents(folder_path: str) -> list:
    """
    Load all PDF files from the knowledge base folder.
    Uses PyMuPDF (fitz) to extract text and metadata from each page.

    Args:
        folder_path: Path to folder containing PDF files

    Returns:
        List of LangChain Document objects with content and metadata
    """
    import fitz  # PyMuPDF
    from langchain_core.documents import Document

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {folder_path}")

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {folder_path}\n"
            f"Please add financial PDFs to {folder_path} first!"
        )

    logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")

    all_documents = []
    total_pages = 0

    for pdf_path in pdf_files:
        try:
            logger.info(f"Loading: {pdf_path.name}")
            doc = fitz.open(str(pdf_path))
            pages_loaded = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # Skip empty pages
                if not text.strip():
                    continue

                # Clean text
                text = " ".join(text.split())

                # Create LangChain Document
                document = Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "file_name": pdf_path.name,
                        "page_number": page_num + 1,
                        "total_pages": len(doc),
                    }
                )
                all_documents.append(document)
                pages_loaded += 1

            total_pages += pages_loaded
            logger.info(f"  ✅ {pdf_path.name} — {pages_loaded} pages loaded")
            doc.close()

        except Exception as e:
            logger.error(f"  ❌ Failed to load {pdf_path.name}: {str(e)}")
            continue

    logger.info(f"Total: {len(pdf_files)} PDFs, {total_pages} pages loaded")
    return all_documents


# ============================================================
# STEP 2 — SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

def split_documents(documents: list) -> list:
    """
    Split loaded documents into smaller chunks for embedding.
    Uses RecursiveCharacterTextSplitter with financial-friendly settings.

    Args:
        documents: List of LangChain Document objects

    Returns:
        List of chunked Document objects
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    logger.info(f"Splitting {len(documents)} pages into chunks...")
    logger.info(f"Settings: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Split on these separators in order
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"✅ Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks


# ============================================================
# STEP 3 — CREATE EMBEDDINGS + STORE IN CHROMADB
# ============================================================

def create_vectorstore(chunks: list, reset: bool = False):
    """
    Generate embeddings using BAAI/bge-base-en-v1.5 and store in ChromaDB.

    Args:
        chunks: List of chunked Document objects
        reset: If True, wipe existing collection before storing

    Returns:
        Chroma vectorstore object
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    import chromadb

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    logger.info("⏳ First run downloads ~500MB model — please wait...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("✅ Embedding model loaded")

    # Handle reset
    if reset:
        logger.warning("⚠️  Reset flag set — wiping existing ChromaDB collection...")
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            chroma_client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info(f"✅ Collection '{CHROMA_COLLECTION_NAME}' deleted")
        except Exception as e:
            logger.info(f"No existing collection to delete: {e}")

    # Store in ChromaDB
    logger.info(f"Storing {len(chunks)} chunks in ChromaDB...")
    logger.info(f"Collection: {CHROMA_COLLECTION_NAME}")
    logger.info(f"Path: {CHROMA_DB_PATH}")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
    )

    # Verify storage
    count = vectorstore._collection.count()
    logger.info(f"✅ {count} chunks stored in ChromaDB successfully")

    return vectorstore


# ============================================================
# STEP 4 — SAVE INGESTION SUMMARY
# ============================================================

def save_ingestion_summary(
    pdf_count: int,
    page_count: int,
    chunk_count: int,
    elapsed_time: float
) -> None:
    """
    Save ingestion statistics to ingestion_summary.json.

    Args:
        pdf_count: Number of PDFs processed
        page_count: Total pages loaded
        chunk_count: Total chunks stored
        elapsed_time: Time taken in seconds
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "pdfs_processed": pdf_count,
        "pages_loaded": page_count,
        "chunks_stored": chunk_count,
        "embedding_model": EMBEDDING_MODEL,
        "collection_name": CHROMA_COLLECTION_NAME,
        "chroma_db_path": CHROMA_DB_PATH,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "elapsed_seconds": round(elapsed_time, 2),
    }

    summary_path = Path("ingestion_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✅ Summary saved to {summary_path}")


# ============================================================
# MAIN — ORCHESTRATE FULL PIPELINE
# ============================================================

def main(reset: bool = False) -> None:
    """
    Run the full ingestion pipeline:
    Load PDFs → Split chunks → Embed → Store in ChromaDB

    Args:
        reset: If True, wipe ChromaDB before re-ingesting
    """
    print("\n" + "="*60)
    print("   FINBOT — RAG INGESTION PIPELINE")
    print("="*60 + "\n")

    start_time = time.time()

    try:
        # Step 1: Load documents
        logger.info("STEP 1/3 — Loading PDF documents...")
        documents = load_documents(KNOWLEDGE_BASE_PATH)
        pdf_count = len(set(d.metadata["file_name"] for d in documents))
        page_count = len(documents)

        # Step 2: Split into chunks
        logger.info("\nSTEP 2/3 — Splitting documents into chunks...")
        chunks = split_documents(documents)
        chunk_count = len(chunks)

        # Step 3: Create vectorstore
        logger.info("\nSTEP 3/3 — Creating embeddings and storing in ChromaDB...")
        vectorstore = create_vectorstore(chunks, reset=reset)

        # Save summary
        elapsed_time = time.time() - start_time
        save_ingestion_summary(pdf_count, page_count, chunk_count, elapsed_time)

        # Final summary
        print("\n" + "="*60)
        print("   INGESTION COMPLETE ✅")
        print("="*60)
        print(f"  PDFs processed  : {pdf_count}")
        print(f"  Pages loaded    : {page_count}")
        print(f"  Chunks stored   : {chunk_count}")
        print(f"  Collection      : {CHROMA_COLLECTION_NAME}")
        print(f"  ChromaDB path   : {CHROMA_DB_PATH}")
        print(f"  Time taken      : {elapsed_time:.1f} seconds")
        print("="*60)
        print("\n  👉 Next: Start Hour 3 (RAG Pipeline)\n")

    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.error(f"   Add PDF files to: {KNOWLEDGE_BASE_PATH}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Ingestion failed: {str(e)}")
        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinBot RAG Ingestion Pipeline")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing ChromaDB collection and re-ingest all PDFs"
    )
    args = parser.parse_args()
    main(reset=args.reset)
