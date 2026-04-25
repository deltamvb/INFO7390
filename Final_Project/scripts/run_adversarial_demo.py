import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from app.auditor import TriageAuditor
from app.case_studies import build_demo_docs, load_cases
from app.rag import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adversarial case-study validation for Layer 3."
    )
    parser.add_argument(
        "--cases-file",
        default="data/adversarial_cases.json",
        help="Path to adversarial case definitions JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved chunks for each case.",
    )
    parser.add_argument(
        "--collection-name",
        default="pubmed_abstracts",
        help="Chroma collection to query.",
    )
    parser.add_argument(
        "--inject-docs",
        action="store_true",
        help="Inject adversarial docs from cases before evaluation.",
    )
    parser.add_argument(
        "--output",
        default="data/adversarial_results.json",
        help="Path to save JSON results.",
    )
    return parser.parse_args()


def _run_case(
    rag: RAGPipeline,
    auditor: TriageAuditor,
    case_id: str,
    title: str,
    query: str,
    expected_flags: List[str],
    notes: str,
    top_k: int,
) -> Dict:
    retrieved = rag.retrieve(query, top_k=top_k)
    answer = rag.generate(query, retrieved)
    audit = auditor.run_all(query, answer, retrieved)

    triggered = {f["flag_id"] for f in audit["flags"] if f["flagged"]}
    expected = set(expected_flags)
    hits = sorted(list(expected.intersection(triggered)))
    misses = sorted(list(expected.difference(triggered)))
    extras = sorted(list(triggered.difference(expected)))

    return {
        "case_id": case_id,
        "title": title,
        "query": query,
        "notes": notes,
        "expected_flags": sorted(list(expected)),
        "triggered_flags": sorted(list(triggered)),
        "matched_expected_flags": hits,
        "missed_expected_flags": misses,
        "extra_flags": extras,
        "answer": answer,
        "audit": audit,
    }


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases_file)
    output_path = Path(args.output)

    if not cases_path.exists():
        raise FileNotFoundError(f"Cases file not found: {cases_path}")

    cases = load_cases(cases_path)
    rag = RAGPipeline(collection_name=args.collection_name)
    auditor = TriageAuditor(groq_client=rag.groq, generation_model=rag.generation_model_name)

    if args.inject_docs:
        total_injected = 0
        for case in cases:
            docs = build_demo_docs(case.case_id, case.injected_docs)
            if docs:
                total_injected += rag.upsert_documents(docs)
        print(f"Injected {total_injected} adversarial documents.")

    results: List[Dict] = []
    for case in cases:
        case_result = _run_case(
            rag=rag,
            auditor=auditor,
            case_id=case.case_id,
            title=case.title,
            query=case.query,
            expected_flags=case.expected_flags,
            notes=case.notes,
            top_k=args.top_k,
        )
        results.append(case_result)

        print(f"\n[{case.case_id}] {case.title}")
        print(f"  Expected: {', '.join(case_result['expected_flags']) or 'none'}")
        print(f"  Triggered: {', '.join(case_result['triggered_flags']) or 'none'}")
        print(f"  Missed: {', '.join(case_result['missed_expected_flags']) or 'none'}")

    summary = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "cases_run": len(results),
        "cases_with_all_expected_flags_hit": sum(
            1 for item in results if len(item["missed_expected_flags"]) == 0
        ),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved adversarial report to: {output_path}")


if __name__ == "__main__":
    main()
