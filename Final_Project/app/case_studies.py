import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class AdversarialCase:
    case_id: str
    title: str
    query: str
    expected_flags: List[str]
    injected_docs: List[Dict]
    notes: str


def load_cases(path: Path) -> List[AdversarialCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: List[AdversarialCase] = []
    for item in payload.get("cases", []):
        cases.append(
            AdversarialCase(
                case_id=item["case_id"],
                title=item["title"],
                query=item["query"],
                expected_flags=item.get("expected_flags", []),
                injected_docs=item.get("injected_docs", []),
                notes=item.get("notes", ""),
            )
        )
    return cases


def build_demo_docs(case_id: str, docs: List[Dict]) -> List[Dict]:
    transformed: List[Dict] = []
    for i, doc in enumerate(docs, start=1):
        transformed.append(
            {
                "id": f"adversarial-{case_id}-{i}",
                "pmid": doc.get("pmid", f"adv-{case_id}-{i}"),
                "title": doc.get("title", f"Adversarial document {i}"),
                "text": doc["text"],
                "source": doc.get("source", f"adversarial://{case_id}/{i}"),
            }
        )
    return transformed
