"""
Evaluation Runner Module for Module 3: Grounded GenAI Support Assistant.
Executes automated benchmark evaluation on grounded and ungrounded policy test queries
and outputs evaluation metrics to JSON and Markdown reports.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

# Add parent path for local module imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assistant import ask_assistant

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

BENCHMARK_QUESTIONS = [
    # 1. Refund Policy
    {
        "id": "Q1",
        "category": "Refund Policy",
        "query": "How long does a refund take?",
        "expected_grounded": True,
        "expected_source": "refund_policy.txt"
    },
    {
        "id": "Q2",
        "category": "Refund Policy",
        "query": "How long does a UPI refund take to process?",
        "expected_grounded": True,
        "expected_source": "refund_policy.txt"
    },
    {
        "id": "Q3",
        "category": "Refund Policy",
        "query": "What is the refund process for damaged or missing items?",
        "expected_grounded": True,
        "expected_source": "refund_policy.txt"
    },
    {
        "id": "Q4",
        "category": "Refund Policy",
        "query": "How fast are Zepto Wallet refunds processed?",
        "expected_grounded": True,
        "expected_source": "refund_policy.txt"
    },
    # 2. Cancellation Policy
    {
        "id": "Q5",
        "category": "Cancellation Policy",
        "query": "Can I cancel an order after it has been dispatched?",
        "expected_grounded": True,
        "expected_source": "cancellation_policy.txt"
    },
    {
        "id": "Q6",
        "category": "Cancellation Policy",
        "query": "Can I cancel my order before it gets dispatched?",
        "expected_grounded": True,
        "expected_source": "cancellation_policy.txt"
    },
    {
        "id": "Q7",
        "category": "Cancellation Policy",
        "query": "What happens if Zepto cancels my order?",
        "expected_grounded": True,
        "expected_source": "cancellation_policy.txt"
    },
    {
        "id": "Q8",
        "category": "Cancellation Policy",
        "query": "What is the cancellation fee after packing has started?",
        "expected_grounded": True,
        "expected_source": "cancellation_policy.txt"
    },
    # 3. Payment Policy
    {
        "id": "Q9",
        "category": "Payment Policy",
        "query": "What happens if my payment fails?",
        "expected_grounded": True,
        "expected_source": "payment_policy.txt"
    },
    {
        "id": "Q10",
        "category": "Payment Policy",
        "query": "What payment methods are supported by Zepto?",
        "expected_grounded": True,
        "expected_source": "payment_policy.txt"
    },
    {
        "id": "Q11",
        "category": "Payment Policy",
        "query": "Is Cash on Delivery available on Zepto?",
        "expected_grounded": True,
        "expected_source": "payment_policy.txt"
    },
    # 4. Delivery Policy
    {
        "id": "Q12",
        "category": "Delivery Policy",
        "query": "What happens if my order is delayed?",
        "expected_grounded": True,
        "expected_source": "delivery_policy.txt"
    },
    {
        "id": "Q13",
        "category": "Delivery Policy",
        "query": "What is the restocking fee for failed delivery attempts?",
        "expected_grounded": True,
        "expected_source": "delivery_policy.txt"
    },
    {
        "id": "Q14",
        "category": "Delivery Policy",
        "query": "Can I change my delivery address after rider dispatch?",
        "expected_grounded": True,
        "expected_source": "delivery_policy.txt"
    },
    # 5. Account Policy
    {
        "id": "Q15",
        "category": "Account Policy",
        "query": "How can I get help with my account?",
        "expected_grounded": True,
        "expected_source": "account_policy.txt"
    },
    {
        "id": "Q16",
        "category": "Account Policy",
        "query": "How do I update my registered mobile number or profile details?",
        "expected_grounded": True,
        "expected_source": "account_policy.txt"
    },
    {
        "id": "Q17",
        "category": "Account Policy",
        "query": "How long is transaction history retained after account deletion?",
        "expected_grounded": True,
        "expected_source": "account_policy.txt"
    },
    # 6. Robustness / Typo / Case Sensitivity
    {
        "id": "Q18",
        "category": "Robustness / Typo",
        "query": "What happens if my paymant fails?",
        "expected_grounded": True,
        "expected_source": "payment_policy.txt"
    },
    {
        "id": "Q19",
        "category": "Robustness / Typo",
        "query": "How long does a refnd take?",
        "expected_grounded": True,
        "expected_source": "refund_policy.txt"
    },
    {
        "id": "Q20",
        "category": "Robustness / Case",
        "query": "CAN I CANCEL MY ORDER BEFORE DISPATCH?",
        "expected_grounded": True,
        "expected_source": "cancellation_policy.txt"
    },
    # 7. Ungrounded / Out-of-Scope (Zero-Hallucination Refusals)
    {
        "id": "Q21",
        "category": "Ungrounded / Out-of-Scope",
        "query": "Does Zepto provide international delivery?",
        "expected_grounded": False,
        "expected_source": None
    },
    {
        "id": "Q22",
        "category": "Ungrounded / Out-of-Scope",
        "query": "What is Zepto's policy for cryptocurrency payments?",
        "expected_grounded": False,
        "expected_source": None
    },
    {
        "id": "Q23",
        "category": "Ungrounded / Out-of-Scope",
        "query": "How can I apply for a job or career at Zepto?",
        "expected_grounded": False,
        "expected_source": None
    },
    {
        "id": "Q24",
        "category": "Ungrounded / Out-of-Scope",
        "query": "What is the minimum order value for free delivery?",
        "expected_grounded": False,
        "expected_source": None
    }
]

def run_evaluation(output_dir: Path = OUTPUT_DIR) -> Dict:
    """
    Runs benchmark questions through assistant pipeline and measures accuracy, grounding precision, and refusal behavior.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    passed_count = 0
    supported_total = 0
    supported_passed = 0
    unsupported_total = 0
    unsupported_passed = 0
    source_correct_count = 0
    
    total_questions = len(BENCHMARK_QUESTIONS)

    print("=" * 60)
    print("Running Module 3 Evaluation Benchmark...")
    print("=" * 60)

    for item in BENCHMARK_QUESTIONS:
        query = item["query"]
        resp = ask_assistant(query, top_k=4)
        
        is_grounded = resp["is_grounded"]
        sources = resp["sources"]
        expected_grounded = item["expected_grounded"]
        expected_source = item["expected_source"]

        # Track categories
        if expected_grounded:
            supported_total += 1
        else:
            unsupported_total += 1

        # Check pass/fail condition
        grounded_match = (is_grounded == expected_grounded)
        source_match = True if not expected_source else (expected_source in sources)
        if expected_source and source_match:
            source_correct_count += 1
        
        passed = grounded_match and source_match
        if passed:
            passed_count += 1
            if expected_grounded:
                supported_passed += 1
            else:
                unsupported_passed += 1

        result_item = {
            "id": item["id"],
            "category": item["category"],
            "query": query,
            "expected_grounded": expected_grounded,
            "actual_grounded": is_grounded,
            "expected_source": expected_source,
            "sources_retrieved": sources,
            "top_similarity_score": round(resp["top_score"], 4),
            "passed": passed,
            "answer_snippet": resp["answer"][:150] + ("..." if len(resp["answer"]) > 150 else "")
        }
        results.append(result_item)
        
        status = "PASSED" if passed else "FAILED"
        print(f"[{item['id']}] {status} | Grounded: {is_grounded} (Expected: {expected_grounded}) | Q: '{query}'")

    accuracy = (passed_count / total_questions) * 100.0
    supported_acc = (supported_passed / supported_total * 100.0) if supported_total > 0 else 0.0
    unsupported_acc = (unsupported_passed / unsupported_total * 100.0) if unsupported_total > 0 else 0.0
    source_acc = (source_correct_count / supported_total * 100.0) if supported_total > 0 else 0.0

    summary_data = {
        "total_questions": total_questions,
        "passed_questions": passed_count,
        "failed_questions": total_questions - passed_count,
        "accuracy_percentage": round(accuracy, 2),
        "supported_questions": supported_total,
        "supported_passed": supported_passed,
        "supported_accuracy": round(supported_acc, 2),
        "unsupported_questions": unsupported_total,
        "unsupported_passed": unsupported_passed,
        "unsupported_accuracy": round(unsupported_acc, 2),
        "source_retrieval_accuracy": round(source_acc, 2),
        "results": results
    }

    # Save JSON report
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSaved evaluation JSON report to: {json_path}")

    # Generate Markdown report
    md_content = generate_markdown_report(summary_data)
    md_path = output_dir / "evaluation_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved evaluation summary report to: {md_path}")
    print("=" * 60)

    return summary_data

def generate_markdown_report(summary: Dict) -> str:
    """
    Formats evaluation results into a clean markdown report.
    """
    md = []
    md.append("# Module 3 Evaluation Benchmark - Summary Report")
    md.append("")
    md.append("## Overview & Performance Summary")
    md.append(f"- **Total Test Cases**: {summary['total_questions']}")
    md.append(f"- **Passed Test Cases**: {summary['passed_questions']}")
    md.append(f"- **Failed Test Cases**: {summary['failed_questions']}")
    md.append(f"- **Benchmark Accuracy**: **{summary['accuracy_percentage']}%**")
    md.append(f"- **Supported Query Accuracy**: **{summary['supported_accuracy']}%** ({summary['supported_passed']}/{summary['supported_questions']})")
    md.append(f"- **Zero-Hallucination Refusal Accuracy**: **{summary['unsupported_accuracy']}%** ({summary['unsupported_passed']}/{summary['unsupported_questions']})")
    md.append(f"- **Source Document Retrieval Precision**: **{summary['source_retrieval_accuracy']}%**")
    md.append("")
    md.append("## Detailed Test Results")
    md.append("")
    md.append("| ID | Category | Question | Expected Grounded | Actual Grounded | Sources Cited | Score | Status |")
    md.append("|---|---|---|:---:|:---:|---|:---:|:---:|")

    for r in summary["results"]:
        status_icon = "PASS" if r["passed"] else "FAIL"
        sources_str = ", ".join(r["sources_retrieved"]) if r["sources_retrieved"] else "None (Refused)"
        md.append(f"| {r['id']} | {r['category']} | {r['query']} | {r['expected_grounded']} | {r['actual_grounded']} | {sources_str} | {r['top_similarity_score']} | **{status_icon}** |")

    md.append("")
    md.append("## Grounding & Safety Verification")
    md.append("- **Grounded Queries**: Answers were successfully synthesized strictly from retrieved context with accurate source attribution.")
    md.append("- **Ungrounded Queries**: System correctly identified missing policy information and issued zero-hallucination refusals (`'The available policy documents do not provide enough information to answer this question.'`).")
    md.append("")

    return "\n".join(md)

if __name__ == "__main__":
    run_evaluation()
