"""
FinBot — evaluation/evaluate.py
=================================
Runs 20 test queries across 5 financial domains and
prints a full evaluation summary table.

Usage:
    python evaluation/evaluate.py
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime

BACKEND_URL = "http://localhost:8000"

# ============================================================
# 20 TEST QUERIES — 4 per domain
# ============================================================
TEST_QUERIES = [
    # Budgeting (4)
    {"domain": "Budgeting", "query": "What is the 50/30/20 budgeting rule?"},
    {"domain": "Budgeting", "query": "How much emergency fund should I maintain?"},
    {"domain": "Budgeting", "query": "How can I track my monthly expenses?"},
    {"domain": "Budgeting", "query": "What percentage of salary should I save each month?"},
    # Investing (4)
    {"domain": "Investing", "query": "How much SIP should I invest to build Rs 1 crore corpus?"},
    {"domain": "Investing", "query": "What is the difference between large cap and small cap funds?"},
    {"domain": "Investing", "query": "What is ELSS and how does it save tax?"},
    {"domain": "Investing", "query": "How does power of compounding work in SIP investments?"},
    # Taxation (4)
    {"domain": "Taxation", "query": "What is the Section 80C deduction limit for FY 2024-25?"},
    {"domain": "Taxation", "query": "What is the difference between new and old tax regime?"},
    {"domain": "Taxation", "query": "What deductions are available under Section 80D?"},
    {"domain": "Taxation", "query": "What is the ITR filing deadline for individuals?"},
    # Insurance (4)
    {"domain": "Insurance", "query": "How much term life insurance cover do I need?"},
    {"domain": "Insurance", "query": "What is the difference between term insurance and ULIP?"},
    {"domain": "Insurance", "query": "What is a family floater health insurance plan?"},
    {"domain": "Insurance", "query": "What is a critical illness rider in insurance?"},
    # Loans (4)
    {"domain": "Loans", "query": "How is home loan EMI calculated?"},
    {"domain": "Loans", "query": "What CIBIL score is needed for home loan approval?"},
    {"domain": "Loans", "query": "What is the benefit of home loan prepayment?"},
    {"domain": "Loans", "query": "What deduction is available on home loan interest?"},
]


def run_evaluation():
    """Run all 20 test queries and print evaluation summary."""

    print("\n" + "="*65)
    print("   FINBOT — EVALUATION REPORT (20 queries / 5 domains)")
    print("="*65 + "\n")

    # Check API is running
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code != 200:
            print("❌ Backend not running! Start with:")
            print("   uvicorn backend.main:app --reload --port 8000")
            return
        print("✅ Backend connected\n")
    except Exception:
        print("❌ Cannot connect to backend at", BACKEND_URL)
        return

    results      = []
    domain_stats = {}

    for i, test in enumerate(TEST_QUERIES, 1):
        domain = test["domain"]
        query  = test["query"]

        print(f"[{i:02d}/20] {domain}: {query[:60]}")

        try:
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "message": query,
                    "session_id": f"eval_{i}",
                    "include_sources": True,
                },
                timeout=60,
            )
            data = response.json()

            confidence = data.get("confidence", 0.0)
            latency_ms = data.get("latency_ms", 0)
            sources    = data.get("sources", [])
            answer     = data.get("answer", "")
            has_disclaimer = "Disclaimer" in answer or "disclaimer" in answer

            print(
                f"        → confidence={confidence:.2f} | "
                f"latency={latency_ms}ms | "
                f"sources={len(sources)}"
            )

            result = {
                "query_number"  : i,
                "domain"        : domain,
                "query"         : query,
                "confidence"    : confidence,
                "latency_ms"    : latency_ms,
                "sources_count" : len(sources),
                "has_disclaimer": has_disclaimer,
                "answer_preview": answer[:200],
            }
            results.append(result)

            # Domain stats
            if domain not in domain_stats:
                domain_stats[domain] = {
                    "count": 0,
                    "total_confidence": 0.0,
                    "total_latency": 0,
                    "answered": 0,
                }
            domain_stats[domain]["count"]            += 1
            domain_stats[domain]["total_confidence"] += confidence
            domain_stats[domain]["total_latency"]    += latency_ms
            if confidence > 0.5:
                domain_stats[domain]["answered"] += 1

        except Exception as e:
            print(f"        → ❌ Error: {str(e)}")
            results.append({
                "query_number": i,
                "domain": domain,
                "query": query,
                "confidence": 0.0,
                "latency_ms": 0,
                "sources_count": 0,
                "has_disclaimer": False,
                "error": str(e),
            })

        time.sleep(0.5)

    # ── Summary Table ────────────────────────────────────────
    print("\n" + "="*65)
    print("   EVALUATION SUMMARY")
    print("="*65)
    print(
        f"{'Domain':<12} | "
        f"{'Avg Conf':<10} | "
        f"{'Avg Latency':<13} | "
        f"{'Answered':<10}"
    )
    print("-"*65)

    total_conf    = 0.0
    total_latency = 0
    total_answered= 0
    total_count   = 0

    for domain, stats in domain_stats.items():
        n          = stats["count"]
        avg_conf   = stats["total_confidence"] / n
        avg_lat    = stats["total_latency"] // n
        answered   = stats["answered"]

        print(
            f"{domain:<12} | "
            f"{avg_conf:<10.3f} | "
            f"{avg_lat:<10}ms   | "
            f"{answered}/{n}"
        )

        total_conf     += stats["total_confidence"]
        total_latency  += stats["total_latency"]
        total_answered += answered
        total_count    += n

    print("-"*65)
    overall_conf = total_conf / total_count
    overall_lat  = total_latency // total_count
    print(
        f"{'OVERALL':<12} | "
        f"{overall_conf:<10.3f} | "
        f"{overall_lat:<10}ms   | "
        f"{total_answered}/{total_count}"
    )
    print("="*65)

    # ── Save Results ──────────────────────────────────────────
    Path("evaluation").mkdir(exist_ok=True)
    output = {
        "timestamp"              : datetime.now().isoformat(),
        "total_queries"          : total_count,
        "overall_avg_confidence" : round(overall_conf, 3),
        "overall_avg_latency_ms" : overall_lat,
        "total_answered"         : total_answered,
        "domain_stats"           : {
            d: {
                "avg_confidence": round(s["total_confidence"]/s["count"], 3),
                "avg_latency_ms": s["total_latency"] // s["count"],
                "answered"      : f"{s['answered']}/{s['count']}",
            }
            for d, s in domain_stats.items()
        },
        "results": results,
    }

    out_path = Path("evaluation/evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to {out_path}")
    print("\n👉 Next: Push to GitHub + Deploy on Hugging Face Spaces!\n")


if __name__ == "__main__":
    run_evaluation()
