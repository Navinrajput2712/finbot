"""
FinBot — rag/pipeline.py
========================
Full RAG chain connecting ChromaDB retriever with NVIDIA NIM LLM.
Handles: context formatting, prompt building, conversation memory,
query routing, and confidence scoring.

Usage:
    from rag.pipeline import ask_finbot, load_vectorstore
"""

import os
import time
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
NVIDIA_API_KEY   = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL  = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL     = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
MAX_HISTORY      = 10   # last 5 turns = 10 messages
MAX_CONTEXT_DOCS = 4


# ============================================================
# STEP 1 — NVIDIA NIM CLIENT
# ============================================================

def get_nim_client() -> OpenAI:
    """
    Initialize and return NVIDIA NIM OpenAI-compatible client.

    Returns:
        OpenAI client pointed at NVIDIA NIM base URL

    Raises:
        ValueError: If NVIDIA_API_KEY is missing
    """
    if not NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY not found in .env file!\n"
            "Get your free key at: https://build.nvidia.com"
        )

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY
    )
    return client


# ============================================================
# STEP 2 — SYSTEM PROMPT
# ============================================================

def build_system_prompt() -> str:
    """
    Build the FinBot system prompt that defines its identity,
    scope, and rules for grounded financial responses.

    Returns:
        System prompt string
    """
    return """You are FinBot, an AI-powered financial advisor specializing in Indian personal finance.

SCOPE — You help users with:
1. Personal Budgeting — expense tracking, savings plans, emergency fund (6 months)
2. Investing — mutual funds, SIPs, ELSS, Nifty/Sensex, portfolio allocation
3. Taxation — new vs old tax regime, 80C/80D deductions, ITR filing
4. Insurance — term life, health insurance, ULIP vs term comparison
5. Loans — home loan EMI, CIBIL score, RBI repo rate impact

RULES — Follow these strictly:
- ONLY answer using the provided CONTEXT below
- If the context does not contain enough information, say:
  "I don't have sufficient information in my knowledge base to answer this accurately. Please consult a SEBI-registered financial advisor."
- NEVER make up numbers, rates, or facts not present in context
- Always be specific with numbers when available (e.g., "Section 80C limit is Rs 1.5 lakh")
- Keep answers clear, structured, and easy to understand
- Use Indian Rupee (Rs) for all monetary values

DISCLAIMER — Always end every response with:
"⚠️ Disclaimer: This is AI-generated information for educational purposes only. Please consult a SEBI-registered investment advisor or certified financial planner before making any financial decisions."
"""


# ============================================================
# STEP 3 — FORMAT CONTEXT FROM RETRIEVED DOCS
# ============================================================

def format_context(documents: list) -> str:
    """
    Format retrieved documents into a clean context string
    to inject into the LLM prompt.

    Args:
        documents: List of Document objects from retriever

    Returns:
        Formatted context string with source citations
    """
    if not documents:
        return "No relevant context found in knowledge base."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        file_name = doc.metadata.get("file_name", "unknown")
        page_num  = doc.metadata.get("page_number", "?")
        score     = doc.metadata.get("rerank_score", 0)

        context_parts.append(
            f"[Source {i}: {file_name} | Page {page_num} | Score: {score:.3f}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(context_parts)


# ============================================================
# STEP 4 — BUILD FULL PROMPT WITH HISTORY
# ============================================================

def build_messages(
    query: str,
    context: str,
    chat_history: List[Dict]
) -> List[Dict]:
    """
    Build the complete messages list for NVIDIA NIM API call.
    Includes: system prompt + chat history (last 5 turns) + current query.

    Args:
        query: Current user question
        context: Formatted retrieved document context
        chat_history: List of previous {"role": ..., "content": ...} messages

    Returns:
        List of message dicts ready for NIM API
    """
    messages = []

    # 1. System prompt
    messages.append({
        "role": "system",
        "content": build_system_prompt()
    })

    # 2. Last 5 turns of chat history (max 10 messages)
    recent_history = chat_history[-MAX_HISTORY:] if chat_history else []
    messages.extend(recent_history)

    # 3. Current user message with context injected
    user_message = f"""CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION:
{query}

Please answer the question using only the context provided above."""

    messages.append({
        "role": "user",
        "content": user_message
    })

    return messages


# ============================================================
# STEP 5 — QUERY ROUTING
# ============================================================

def is_market_query(query: str) -> bool:
    """
    Detect if user is asking about live market data
    (stock prices, indices) vs static knowledge.

    Args:
        query: User's question string

    Returns:
        True if query needs live market data
    """
    market_keywords = [
        "price", "stock price", "share price", "current price",
        "nifty", "sensex", "nse", "bse", "market",
        "trading at", "how much is", "what is the price",
        "today's price", "live price",
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in market_keywords)


def calculate_confidence(documents: list) -> float:
    """
    Calculate confidence score from reranked document scores.
    Normalized to 0.0 - 1.0 range using sigmoid-like scaling.

    Args:
        documents: List of Document objects with rerank_score in metadata

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not documents:
        return 0.0

    scores = [
        doc.metadata.get("rerank_score", 0.0)
        for doc in documents
    ]
    avg_score = sum(scores) / len(scores)

    # Normalize: cross-encoder scores typically range -10 to +10
    # Map to 0-1 using sigmoid
    import math
    normalized = 1 / (1 + math.exp(-avg_score * 0.5))
    return round(normalized, 3)


# ============================================================
# STEP 6 — MAIN ASK FUNCTION
# ============================================================

def ask_finbot(
    query: str,
    chat_history: List[Dict],
    vectorstore,
    market_context: Optional[str] = None
) -> Dict:
    """
    Main FinBot RAG pipeline function.

    Steps:
    1. Retrieve top-6 docs from ChromaDB
    2. Rerank to top-4 using cross-encoder
    3. Format context from retrieved docs
    4. Build prompt with history
    5. Call NVIDIA NIM llama-3.1-8b-instruct
    6. Return answer + sources + confidence + latency

    Args:
        query: User's financial question
        chat_history: Previous conversation turns
        vectorstore: Loaded ChromaDB vectorstore
        market_context: Optional live market data string to prepend

    Returns:
        Dict with keys: answer, sources, confidence, latency_ms, model
    """
    from rag.retriever import retrieve_and_rerank

    start_time = time.time()

    try:
        # Step 1 & 2: Retrieve + Rerank
        logger.info(f"Processing query: '{query[:80]}'")
        documents = retrieve_and_rerank(query, vectorstore, k=6, top_n=4)

        # Step 3: Format context
        context = format_context(documents)

        # Add live market data if available
        if market_context:
            context = f"LIVE MARKET DATA:\n{market_context}\n\n{context}"

        # Step 4: Build messages
        messages = build_messages(query, context, chat_history)

        # Step 5: Call NVIDIA NIM API
        client = get_nim_client()
        logger.info(f"Calling NVIDIA NIM: {NVIDIA_MODEL}")

        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
            top_p=0.9,
        )

        answer = response.choices[0].message.content.strip()

        # Step 6: Build sources list
        sources = []
        for doc in documents:
            sources.append({
                "file_name": doc.metadata.get("file_name", "unknown"),
                "page_number": doc.metadata.get("page_number", 0),
                "relevance_score": round(
                    doc.metadata.get("rerank_score", 0.0), 3
                ),
            })

        # Calculate confidence
        confidence = calculate_confidence(documents)

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"✅ Response generated | "
            f"confidence={confidence:.2f} | "
            f"latency={latency_ms}ms"
        )

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model": NVIDIA_MODEL,
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.error(f"❌ Pipeline error: {str(e)}")
        return {
            "answer": (
                "I encountered an error processing your request. "
                "Please try again or rephrase your question.\n\n"
                "⚠️ Disclaimer: This is AI-generated information for "
                "educational purposes only."
            ),
            "sources": [],
            "confidence": 0.0,
            "latency_ms": latency_ms,
            "model": NVIDIA_MODEL,
            "error": str(e),
        }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    from rag.retriever import load_vectorstore

    print("\n" + "="*60)
    print("   FINBOT — RAG PIPELINE QUICK TEST")
    print("="*60 + "\n")

    # Load vectorstore
    print("Loading ChromaDB vectorstore...")
    vs = load_vectorstore()

    # Single test
    test_query = "What is the Section 80C deduction limit?"
    print(f"Query: {test_query}\n")

    result = ask_finbot(
        query=test_query,
        chat_history=[],
        vectorstore=vs
    )

    print(f"Answer:\n{result['answer']}\n")
    print(f"Confidence : {result['confidence']}")
    print(f"Latency    : {result['latency_ms']}ms")
    print(f"Sources    : {len(result['sources'])} documents used")
    for src in result["sources"]:
        print(
            f"  - {src['file_name']} "
            f"p.{src['page_number']} "
            f"(score: {src['relevance_score']})"
        )
