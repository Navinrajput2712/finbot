"""
FinBot — rag/test_pipeline.py
==============================
Tests the full RAG pipeline across all 5 financial domains.
Saves results to evaluation/test_results.json.

Usage:
    python rag/test_pipeline.py
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# TEST QUERIES — 5 domains x 2 queries each
# ============================================================
TEST_QUERIES = [
    # Budgeting
    {
        "domain": "Budgeting",
        "query": "What is the 50/30/20 rule for budgeting?"
    },
    {
        "domain": "Budgeting",
        "query": "How much emergency fund should I maintain?"
    },
    # Investing
    {
        "domain": "Investing",
        "query": "How much SIP should I invest to build Rs 1 crore corpus?"
    },
    {
        "domain": "Investing",
        "query": "What is the difference between large cap and small cap funds?"
    },
    # Taxation
    {
        "domain": "Taxation",
        "query": "What is the Section 80C deduction limit for FY 2024-25?"
    },
    {
        "domain": "Taxation",
        "query": "What is the difference between new and old tax regime?"
    },
    # Insurance
    {
        "domain": "Insurance",
        "query": "How much term life insurance cover do I need?"
    },
    {
        "domain": "Insurance",
        "query": "What is the difference between term insurance and ULIP?"
    },
    # Loans
    {
        "domain": "Loans",
        "query": "How is home loan EMI calculated?"
    },
    {
        "domain": "Loans",
        "query": "What CIBIL score is needed for home loan approval?"
    },
]


# ============================================================
# RUN TESTS
# ============================================================

def run_tests():
    """
    Run all test queries through the FinBot RAG pipeline.
    Prints results and saves to evaluation/test_results.json.
    """
    from rag.retriever import load_vectorstore
    from rag.pipeline import ask_finbot

    print("\n" + "="*60)
    print("   FINBOT — RAG PIPELINE TEST (10 queries)")
    print("="*60 + "\n")

    # Load vectorstore once
    logger.info("Loading ChromaDB vectorstore...")
    vs = load_vectorstore()

    results = []
    domain_stats = {}

    for i, test in enumerate(TEST_QUERIES, 1):
        domain = test["domain"]
        query  = test["query"]

        print(f"\n[{i}/10] Domain: {domain}")
        print(f"Query : {query}")

        # Run pipeline
        result = ask_finbot(
            query=query,
            chat_history=[],
            vectorstore=vs
        )

        # Print summary
        print(f"Answer preview : {result['answer'][:200]}...")
        print(f"Confidence     : {result['confidence']}")
        print(f"Latency        : {result['latency_ms']}ms")
        print(f"Sources used   : {len(result['sources'])}")

        # Store result
        test_result = {
            "query_number": i,
            "domain": domain,
            "query": query,
            "answer_preview": result["answer"][:300],
            "full_answer": result["answer"],
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"],
            "sources": result["sources"],
            "has_disclaimer": "Disclaimer" in result["answer"],
        }
        results.append(test_result)

        # Track domain stats
        if domain not in domain_stats:
            domain_stats[domain] = {
                "queries": 0,
                "total_confidence": 0.0,
                "total_latency_ms": 0,
            }
        domain_stats[domain]["queries"] += 1
        domain_stats[domain]["total_confidence"] += result["confidence"]
        domain_stats[domain]["total_latency_ms"] += result["latency_ms"]

        # Small delay to avoid rate limiting
        time.sleep(1)

    # ── Summary Table ────────────────────────────────────────
    print("\n" + "="*60)
    print("   TEST RESULTS SUMMARY")
    print("="*60)
    print(f"{'Domain':<15} | {'Avg Confidence':<15} | {'Avg Latency':<12} | Queries")
    print("-"*60)

    total_confidence = 0.0
    total_latency    = 0
    total_queries    = 0

    for domain, stats in domain_stats.items():
        n     = stats["queries"]
        avg_c = stats["total_confidence"] / n
        avg_l = stats["total_latency_ms"] // n
        print(f"{domain:<15} | {avg_c:<15.3f} | {avg_l:<10}ms | {n}")
        total_confidence += stats["total_confidence"]
        total_latency    += stats["total_latency_ms"]
        total_queries    += n

    print("-"*60)
    overall_conf    = total_confidence / total_queries
    overall_latency = total_latency // total_queries
    print(f"{'OVERALL':<15} | {overall_conf:<15.3f} | {overall_latency:<10}ms | {total_queries}")
    print("="*60)

    # ── Save results ─────────────────────────────────────────
    Path("evaluation").mkdir(exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": total_queries,
        "overall_avg_confidence": round(overall_conf, 3),
        "overall_avg_latency_ms": overall_latency,
        "domain_stats": {
            d: {
                "avg_confidence": round(
                    s["total_confidence"] / s["queries"], 3
                ),
                "avg_latency_ms": s["total_latency_ms"] // s["queries"],
                "queries": s["queries"],
            }
            for d, s in domain_stats.items()
        },
        "results": results,
    }

    output_path = Path("evaluation/test_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")
    print("\n👉 Next: Start Hour 4 (FastAPI Backend)\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_tests()
