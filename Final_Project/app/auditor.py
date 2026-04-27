import json
import re
from typing import Dict, List


CONFIDENT_WORDS = [
    "definitely",
    "certainly",
    "clearly",
    "proves",
    "always",
    "never",
    "conclusive",
    "without doubt",
]


def _safe_parse_json(text: str) -> Dict:
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return {}


def _answer_confidence_score(answer: str) -> float:
    lower = answer.lower()
    hits = sum(1 for term in CONFIDENT_WORDS if term in lower)
    # Light heuristic confidence score in [0, 1]
    return min(1.0, 0.2 + hits * 0.12)


def _normalized_retrieval_confidence(distances: List[float]) -> float:
    if not distances:
        return 0.0
    best = min(distances)
    # Chroma distance is lower-is-better; map approximately to confidence.
    # Works as a practical heuristic for triage logic.
    score = 1.0 - best
    return max(0.0, min(1.0, score))


class TriageAuditor:
    def __init__(self, groq_client, generation_model: str) -> None:
        self.groq = groq_client
        self.model = generation_model

    def _llm_json(self, system_prompt: str, user_prompt: str) -> Dict:
        response = self.groq.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        return _safe_parse_json(text)

    def _build_sources(self, retrieval_result: Dict) -> str:
        docs = retrieval_result.get("documents", [[]])[0]
        metas = retrieval_result.get("metadatas", [[]])[0]
        lines = []
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            lines.append(
                f"[Source {i}] {meta.get('title', 'Untitled')} ({meta.get('source', '')})\n{doc}"
            )
        return "\n\n".join(lines)

    def flag_claims_exceeding_data(self, answer: str, retrieval_result: Dict) -> Dict:
        payload = self._llm_json(
            system_prompt=(
                "You are a scientific claims auditor. Return strict JSON only with keys: "
                "flagged(boolean), findings(array of short strings), reason(string)."
            ),
            user_prompt=(
                "Task: flag answer sentences that overstate evidence strength.\n\n"
                f"Answer:\n{answer}\n\n"
                f"Sources:\n{self._build_sources(retrieval_result)}"
            ),
        )
        return {
            "flag_id": "flag_2_claims_exceeding_data",
            "flagged": bool(payload.get("flagged", False)),
            "findings": payload.get("findings", []),
            "reason": payload.get("reason", ""),
            "stakes": "high",
            "detectability": "medium",
        }

    def flag_load_bearing_citations(self, answer: str, retrieval_result: Dict) -> Dict:
        payload = self._llm_json(
            system_prompt=(
                "You are a citation entailment auditor. Return strict JSON only with keys: "
                "flagged(boolean), unsupported_claims(array of strings), reason(string)."
            ),
            user_prompt=(
                "Task: check whether cited claims are actually supported by their linked source chunks.\n\n"
                f"Answer:\n{answer}\n\n"
                f"Sources:\n{self._build_sources(retrieval_result)}"
            ),
        )
        return {
            "flag_id": "flag_3_load_bearing_citations",
            "flagged": bool(payload.get("flagged", False)),
            "findings": payload.get("unsupported_claims", []),
            "reason": payload.get("reason", ""),
            "stakes": "high",
            "detectability": "medium",
        }

    def flag_handoff_confidence_laundering(self, answer: str, retrieval_result: Dict) -> Dict:
        distances = retrieval_result.get("distances", [[]])[0]
        retrieval_conf = _normalized_retrieval_confidence(distances)
        generation_conf = _answer_confidence_score(answer)
        # Slightly lower thresholds improve adversarial-demo sensitivity.
        flagged = retrieval_conf < 0.55 and generation_conf > 0.45
        return {
            "flag_id": "flag_5_handoff_confidence_laundering",
            "flagged": flagged,
            "findings": [
                (
                    f"retrieval_confidence={retrieval_conf:.3f}, "
                    f"generation_confidence={generation_conf:.3f}"
                )
            ],
            "reason": (
                "Low retrieval confidence but confident answer tone detected."
                if flagged
                else "Confidence handoff appears proportionate."
            ),
            "stakes": "high",
            "detectability": "high",
            "metrics": {
                "retrieval_confidence": round(retrieval_conf, 3),
                "generation_confidence": round(generation_conf, 3),
            },
        }

    def flag_out_of_distribution_confidence(self, answer: str, retrieval_result: Dict) -> Dict:
        distances = retrieval_result.get("distances", [[]])[0]
        best_distance = min(distances) if distances else 1.0
        generation_conf = _answer_confidence_score(answer)
        flagged = best_distance > 0.6 and generation_conf > 0.45
        return {
            "flag_id": "flag_6_ood_confidence",
            "flagged": flagged,
            "findings": [f"best_distance={best_distance:.3f}, answer_confidence={generation_conf:.3f}"],
            "reason": (
                "Weak retrieval match but confident generation suggests out-of-domain response."
                if flagged
                else "Best retrieval match is acceptable for the response confidence."
            ),
            "stakes": "high",
            "detectability": "high",
        }

    def flag_suspiciously_clean_convergence(self, retrieval_result: Dict) -> Dict:
        distances = retrieval_result.get("distances", [[]])[0]
        if len(distances) < 3:
            return {
                "flag_id": "flag_7_suspiciously_clean_convergence",
                "flagged": False,
                "findings": [],
                "reason": "Insufficient retrieved sources to evaluate convergence.",
                "stakes": "medium",
                "detectability": "low",
            }
        spread = max(distances) - min(distances)
        flagged = spread < 0.05
        return {
            "flag_id": "flag_7_suspiciously_clean_convergence",
            "flagged": flagged,
            "findings": [f"distance_spread={spread:.4f}"],
            "reason": (
                "Retrieved evidence appears too homogeneous; corroboration may be shallow."
                if flagged
                else "Retrieved evidence shows a healthy spread."
            ),
            "stakes": "medium",
            "detectability": "low",
        }

    def run_all(self, question: str, answer: str, retrieval_result: Dict) -> Dict:
        flags = [
            self.flag_claims_exceeding_data(answer, retrieval_result),
            self.flag_load_bearing_citations(answer, retrieval_result),
            self.flag_handoff_confidence_laundering(answer, retrieval_result),
            self.flag_out_of_distribution_confidence(answer, retrieval_result),
            self.flag_suspiciously_clean_convergence(retrieval_result),
        ]
        flagged_count = sum(1 for item in flags if item["flagged"])
        return {
            "question": question,
            "flagged_count": flagged_count,
            "total_flags_checked": len(flags),
            "audit_score": round(1.0 - (flagged_count / len(flags)), 3),
            "flags": flags,
        }
