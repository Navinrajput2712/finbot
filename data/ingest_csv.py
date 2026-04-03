"""
FinBot — data/ingest_csv.py
============================
Loads 68,913 rows financial CSV dataset into ChromaDB.
Combines instruction + input + output columns into rich text chunks.

Usage:
    python data/ingest_csv.py --csv data/financial_dataset.csv
    python data/ingest_csv.py --csv data/financial_dataset.csv --reset
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

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
CHROMA_DB_PATH       = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION    = os.getenv("CHROMA_COLLECTION_NAME", "finbot_knowledge")
EMBEDDING_MODEL      = "BAAI/bge-base-en-v1.5"
BATCH_SIZE           = 500   # Process 500 rows at a time
CHUNK_SIZE           = 1000  # Max chars per chunk


# ============================================================
# STEP 1 — LOAD CSV
# ============================================================

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the financial CSV dataset.

    Args:
        csv_path: Path to CSV file

    Returns:
        Pandas DataFrame with cleaned data
    """
    logger.info(f"Loading CSV: {csv_path}")

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # Check required columns
    required = ["instruction", "output"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # Fill NaN with empty string
    df = df.fillna("")

    # Remove rows where both instruction and output are empty
    df = df[
        (df["instruction"].str.strip() != "") |
        (df["output"].str.strip() != "")
    ]

    logger.info(f"After cleaning: {len(df):,} valid rows")
    return df


# ============================================================
# STEP 2 — CONVERT ROWS TO DOCUMENTS
# ============================================================

def rows_to_documents(df: pd.DataFrame) -> list:
    """
    Convert CSV rows to LangChain Document objects.
    Combines instruction + input + output into rich text.

    Format:
        Question: {instruction}
        Context: {input}      ← only if input exists
        Answer: {output}

    Args:
        df: Pandas DataFrame with financial Q&A data

    Returns:
        List of LangChain Document objects
    """
    from langchain_core.documents import Document

    logger.info(f"Converting {len(df):,} rows to documents...")
    documents = []

    for idx, row in df.iterrows():
        instruction = str(row.get("instruction", "")).strip()
        input_text  = str(row.get("input", "")).strip()
        output      = str(row.get("output", "")).strip()

        # Skip if no meaningful content
        if not instruction and not output:
            continue

        # Combine into rich text
        if input_text:
            combined_text = (
                f"Question: {instruction}\n"
                f"Context: {input_text}\n"
                f"Answer: {output}"
            )
        else:
            combined_text = (
                f"Question: {instruction}\n"
                f"Answer: {output}"
            )

        # Truncate if too long
        if len(combined_text) > CHUNK_SIZE * 2:
            combined_text = combined_text[:CHUNK_SIZE * 2]

        # Create Document with metadata
        doc = Document(
            page_content=combined_text,
            metadata={
                "source"      : "financial_dataset_csv",
                "row_index"   : int(idx),
                "has_context" : bool(input_text),
                "file_name"   : "financial_qa_dataset.csv",
                "page_number" : int(idx) + 1,
            }
        )
        documents.append(doc)

        # Progress log every 10,000 rows
        if (idx + 1) % 10000 == 0:
            logger.info(f"  Processed {idx + 1:,} rows...")

    logger.info(f"✅ Created {len(documents):,} documents")
    return documents


# ============================================================
# STEP 3 — STORE IN CHROMADB IN BATCHES
# ============================================================

def store_in_chromadb(
    documents: list,
    reset: bool = False
) -> None:
    """
    Store documents in ChromaDB using batched processing.
    Uses BAAI/bge-base-en-v1.5 embeddings.

    Args:
        documents: List of Document objects
        reset: If True, wipe existing collection first
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    import chromadb

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    logger.info("⏳ This may take a moment on first run...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logger.info("✅ Embedding model loaded")

    # Handle reset
    if reset:
        logger.warning("⚠️  Resetting ChromaDB collection...")
        try:
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            client.delete_collection(CHROMA_COLLECTION)
            logger.info(f"✅ Collection '{CHROMA_COLLECTION}' deleted")
        except Exception as e:
            logger.info(f"No existing collection to delete: {e}")

    # Store in batches to avoid memory issues with 68K rows
    total      = len(documents)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info(
        f"Storing {total:,} documents in {num_batches} batches "
        f"of {BATCH_SIZE} each..."
    )

    start_time = time.time()

    for batch_num in range(num_batches):
        batch_start = batch_num * BATCH_SIZE
        batch_end   = min(batch_start + BATCH_SIZE, total)
        batch       = documents[batch_start:batch_end]

        logger.info(
            f"  Batch {batch_num + 1}/{num_batches} "
            f"({batch_start:,} → {batch_end:,})"
        )

        try:
            if batch_num == 0 and reset:
                # First batch — create new collection
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=CHROMA_COLLECTION,
                    persist_directory=CHROMA_DB_PATH,
                )
            elif batch_num == 0:
                # First batch without reset
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    collection_name=CHROMA_COLLECTION,
                    persist_directory=CHROMA_DB_PATH,
                )
            else:
                # Subsequent batches — add to existing
                vectorstore.add_documents(batch)

            # Progress timing
            elapsed = time.time() - start_time
            docs_done = batch_end
            rate = docs_done / elapsed
            remaining = (total - docs_done) / rate if rate > 0 else 0
            logger.info(
                f"  ✅ Batch done | "
                f"Rate: {rate:.0f} docs/s | "
                f"ETA: {remaining/60:.1f} min"
            )

        except Exception as e:
            logger.error(f"  ❌ Batch {batch_num + 1} failed: {str(e)}")
            continue

    # Final count
    final_count = vectorstore._collection.count()
    total_time  = time.time() - start_time

    logger.info(f"\n✅ Total chunks in ChromaDB: {final_count:,}")
    logger.info(f"✅ Total time: {total_time/60:.1f} minutes")


# ============================================================
# MAIN
# ============================================================

def main(csv_path: str, reset: bool = False) -> None:
    """
    Full pipeline: Load CSV → Convert → Store in ChromaDB

    Args:
        csv_path: Path to financial CSV dataset
        reset: Wipe ChromaDB before storing
    """
    print("\n" + "="*60)
    print("   FINBOT — CSV DATASET INGESTION")
    print("="*60 + "\n")

    start = time.time()

    # Step 1: Load CSV
    df = load_csv(csv_path)

    # Step 2: Convert to documents
    documents = rows_to_documents(df)

    # Step 3: Store in ChromaDB
    store_in_chromadb(documents, reset=reset)

    # Summary
    elapsed = time.time() - start
    print("\n" + "="*60)
    print("   CSV INGESTION COMPLETE ✅")
    print("="*60)
    print(f"  Rows processed  : {len(df):,}")
    print(f"  Docs created    : {len(documents):,}")
    print(f"  Collection      : {CHROMA_COLLECTION}")
    print(f"  Time taken      : {elapsed/60:.1f} minutes")
    print("="*60)

    # Save summary
    summary = {
        "timestamp"      : datetime.now().isoformat(),
        "csv_path"       : csv_path,
        "rows_processed" : len(df),
        "docs_created"   : len(documents),
        "collection"     : CHROMA_COLLECTION,
        "elapsed_minutes": round(elapsed/60, 2),
    }
    with open("csv_ingestion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Summary saved to csv_ingestion_summary.json")
    print("👉 Now restart FastAPI — FinBot will use the new data!\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest financial CSV dataset into ChromaDB"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/financial_dataset.csv",
        help="Path to CSV file"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe existing ChromaDB collection before ingesting"
    )
    args = parser.parse_args()
    main(csv_path=args.csv, reset=args.reset)