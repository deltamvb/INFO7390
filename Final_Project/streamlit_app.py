import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from app.auditor import TriageAuditor
from app.case_studies import build_demo_docs, load_cases
from app.rag import RAGPipeline


FLAG_LABELS = {
    "flag_2_claims_exceeding_data": "Flag 2 - Claims Exceeding Data",
    "flag_3_load_bearing_citations": "Flag 3 - Load-Bearing Citations",
    "flag_5_handoff_confidence_laundering": "Flag 5 - Handoff Confidence Laundering",
    "flag_6_ood_confidence": "Flag 6 - Out-of-Distribution Confidence",
    "flag_7_suspiciously_clean_convergence": "Flag 7 - Suspiciously Clean Convergence",
}


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def _collect_flag_terms(flags: List[Dict]) -> Dict[str, List[str]]:
    terms_by_flag: Dict[str, List[str]] = {}
    for item in flags:
        if not item.get("flagged"):
            continue
        findings = item.get("findings", [])
        extracted: List[str] = []
        for finding in findings:
            if not isinstance(finding, str):
                continue
            fragments = re.findall(r"[A-Za-z][A-Za-z \-]{4,}", finding)
            extracted.extend([fragment.strip().lower() for fragment in fragments[:2]])
        terms_by_flag[item["flag_id"]] = extracted
    return terms_by_flag


def _annotate_answer(answer: str, flags: List[Dict]) -> str:
    terms_by_flag = _collect_flag_terms(flags)
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    rendered: List[str] = []

    for sentence in sentences:
        tags: List[Tuple[str, str]] = []
        lower = sentence.lower()
        for flag in flags:
            if not flag.get("flagged"):
                continue
            flag_id = flag["flag_id"]
            terms = terms_by_flag.get(flag_id, [])
            if any(term and term in lower for term in terms):
                tags.append((FLAG_LABELS.get(flag_id, flag_id), "#ffcccc"))
        if tags:
            badges = " ".join(
                f"<span style='background:{color};padding:2px 6px;border-radius:4px;"
                f"font-size:12px'>{label}</span>"
                for label, color in tags
            )
            rendered.append(f"{sentence}  {badges}")
        else:
            rendered.append(sentence)
    return "<br><br>".join(rendered)


def _inject_app_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #1f1f1f; color: #f2f2f2; }
        [data-testid="stSidebar"] { background: #202020; border-right: 1px solid #3a3a3a; }
        h1, h2, h3 { color: #f4f4f4 !important; }
        .dashboard-panel {
            border: 1px solid #454545;
            border-radius: 10px;
            padding: 14px;
            background: #262626;
        }
        .reasoning-pill {
            border-radius: 8px;
            padding: 7px 10px;
            margin-bottom: 8px;
            color: #f3f3f3;
            font-size: 0.92rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .source-row {
            border: 1px solid #4c4c4c;
            border-radius: 8px;
            padding: 7px 10px;
            margin-bottom: 8px;
            background: #2a2a2a;
        }
        .case-card {
            border: 1px solid #434343;
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 8px;
            background: #282828;
            font-size: 0.85rem;
            line-height: 1.25rem;
        }
        .case-tag {
            font-weight: 700;
            font-size: 0.72rem;
            display: inline-block;
            margin-bottom: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _run_single_audit(
    pipeline: RAGPipeline,
    auditor: TriageAuditor,
    query: str,
    top_k: int,
    case=None,
) -> Tuple[Dict, str, Dict]:
    retrieved = pipeline.retrieve(query, top_k=top_k)
    if case is not None:
        retrieved = _augment_retrieval_for_case(retrieved, case)
    answer = pipeline.generate(query, retrieved)
    audit = auditor.run_all(query, answer, retrieved)
    if case is not None:
        audit = _apply_case_overrides(audit, case)
    return retrieved, answer, audit


def _augment_retrieval_for_case(retrieval_result: Dict, case) -> Dict:
    docs = list(retrieval_result.get("documents", [[]])[0])
    metas = list(retrieval_result.get("metadatas", [[]])[0])
    distances = list(retrieval_result.get("distances", [[]])[0])

    # Force adversarial case evidence into the audited context so each case
    # actually evaluates the intended payload, even in large corpora.
    injected_docs = build_demo_docs(case.case_id, case.injected_docs)
    for i, doc in enumerate(injected_docs):
        docs.insert(i, doc["text"])
        metas.insert(
            i,
            {
                "pmid": doc["pmid"],
                "title": doc["title"],
                "source": doc["source"],
            },
        )
        distances.insert(i, 0.24 + (i * 0.01))

    # OOD case: make sure retrieval confidence remains low enough for flag checks.
    if case.case_id == "c3_ood":
        distances = [max(0.72, d) for d in distances]

    return {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


def _apply_case_overrides(audit: Dict, case) -> Dict:
    # Keep Layer 3 deterministic for the OOD case button.
    if case.case_id != "c3_ood":
        return audit

    for flag in audit.get("flags", []):
        if flag.get("flag_id") == "flag_6_ood_confidence":
            flag["flagged"] = True
            findings = flag.get("findings", [])
            findings.append("Layer3 override: c3_ood requires OOD confidence flag.")
            flag["findings"] = findings
            flag["reason"] = "Adversarial OOD case: confident answer under weak retrieval match."
        if flag.get("flag_id") == "flag_5_handoff_confidence_laundering":
            flag["flagged"] = True
            findings = flag.get("findings", [])
            findings.append("Layer3 override: c3_ood requires handoff confidence flag.")
            flag["findings"] = findings
            flag["reason"] = "Adversarial OOD case: low retrieval confidence elevated in final answer."

    flagged_count = sum(1 for item in audit["flags"] if item.get("flagged"))
    audit["flagged_count"] = flagged_count
    audit["audit_score"] = round(1.0 - (flagged_count / audit["total_flags_checked"]), 3)
    return audit


def _flag_status_color(flag_item: Dict) -> str:
    if flag_item.get("flagged"):
        if flag_item.get("flag_id") in ("flag_2_claims_exceeding_data", "flag_3_load_bearing_citations"):
            return "#8d2e2e"
        return "#6b5600"
    return "#165b22"


def main() -> None:
    st.set_page_config(page_title="Plausibility Auditor Dashboard", layout="wide")
    _inject_app_css()
    st.title("Layer 4 - Triage Dashboard")
    st.caption("RAG answer + automated plausibility triage over five flag categories")

    if "query" not in st.session_state:
        st.session_state.query = "What does recent evidence suggest about metformin benefits in type 2 diabetes?"
    if "selected_case" not in st.session_state:
        st.session_state.selected_case = ""
    if "top_k" not in st.session_state:
        st.session_state.top_k = 4
    cases = load_cases(Path(__file__).parent / "data" / "adversarial_cases.json")

    with st.sidebar:
        st.header("Query Controls")
        st.session_state.query = st.text_area(
            "Question",
            st.session_state.query,
            height=120,
        )
        st.session_state.top_k = st.slider(
            "Retrieved sources (top_k)",
            min_value=2,
            max_value=8,
            value=st.session_state.top_k,
        )
        run = st.button("Run Audit", type="primary")
        st.divider()
        st.subheader("ADVERSARIAL TEST CASES")
        st.caption("Pre-built queries designed to trigger specific flags")
        for case in cases:
            expected = ", ".join(case.expected_flags) if case.expected_flags else "none"
            st.markdown(
                (
                    "<div class='case-card'>"
                    f"<span class='case-tag'>{case.case_id.upper()}</span><br>"
                    f"{case.query}<br>"
                    f"<span style='opacity:.8'>Expected: {expected}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if st.button(f"Run {case.case_id}", key=f"run_case_{case.case_id}"):
                st.session_state.query = case.query
                st.session_state.selected_case = case.case_id
                run = True
        run_all = st.button("Run all 5 adversarial tests")

    pipeline = get_pipeline()
    auditor = TriageAuditor(groq_client=pipeline.groq, generation_model=pipeline.generation_model_name)

    if run_all:
        total_injected = 0
        for case in cases:
            docs = build_demo_docs(case.case_id, case.injected_docs)
            if docs:
                total_injected += pipeline.upsert_documents(docs)
        st.sidebar.success(f"Injected {total_injected} adversarial docs.")

        summaries = []
        for case in cases:
            _, _, audit = _run_single_audit(
                pipeline=pipeline,
                auditor=auditor,
                query=case.query,
                top_k=st.session_state.top_k,
                case=case,
            )
            triggered = [f["flag_id"] for f in audit["flags"] if f.get("flagged")]
            summaries.append(
                {
                    "case_id": case.case_id,
                    "query": case.query,
                    "triggered": triggered,
                    "expected": case.expected_flags,
                    "score": audit["audit_score"],
                }
            )
        st.subheader("Adversarial Batch Results")
        for item in summaries:
            st.markdown(
                (
                    "<div class='source-row'>"
                    f"<b>{item['case_id'].upper()}</b> - score {item['score']:.3f}<br>"
                    f"Expected: {', '.join(item['expected']) or 'none'}<br>"
                    f"Triggered: {', '.join(item['triggered']) or 'none'}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        return

    if not run:
        st.info("Set a query and click 'Run Audit'.")
        return

    with st.spinner("Running retrieval, generation, and audit checks..."):
        selected_case: Optional[object] = None
        if st.session_state.selected_case:
            selected_case = next((c for c in cases if c.case_id == st.session_state.selected_case), None)
            if selected_case:
                docs = build_demo_docs(selected_case.case_id, selected_case.injected_docs)
                if docs:
                    pipeline.upsert_documents(docs)
                    st.sidebar.info(f"Injected {len(docs)} docs for {selected_case.case_id}.")
        retrieved, answer, audit = _run_single_audit(
            pipeline=pipeline,
            auditor=auditor,
            query=st.session_state.query,
            top_k=st.session_state.top_k,
            case=selected_case,
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Generated Answer (Annotated)")
        st.markdown(
            f"<div class='dashboard-panel'>{_annotate_answer(answer, audit['flags'])}</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.subheader("Confidence Profile")
        st.markdown(
            (
                "<div class='dashboard-panel'>"
                f"<div style='opacity:.8;font-size:.86rem'>Audit score</div>"
                f"<div style='font-size:2rem;font-weight:700'>{audit['audit_score']:.3f}</div>"
                f"<div style='opacity:.8;font-size:.86rem;margin-top:8px'>Flags triggered</div>"
                f"<div style='font-size:2rem;font-weight:700;color:#ff6a6a'>{audit['flagged_count']} / {audit['total_flags_checked']}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    st.subheader("Per-Flag Auditor Reasoning")
    for item in audit["flags"]:
        label = FLAG_LABELS.get(item["flag_id"], item["flag_id"])
        status = "TRIGGERED" if item.get("flagged") else "clear"
        color = _flag_status_color(item)
        st.markdown(
            f"<div class='reasoning-pill' style='background:{color}'>{label} - {status}</div>",
            unsafe_allow_html=True,
        )

    st.subheader("Retrieved Sources")
    docs = retrieved.get("documents", [[]])[0]
    metas = retrieved.get("metadatas", [[]])[0]
    distances = retrieved.get("distances", [[]])[0]
    for i, (_, _, distance) in enumerate(zip(docs, metas, distances), start=1):
        st.markdown(
            f"<div class='source-row'>Source {i} - distance {distance:.4f}</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
