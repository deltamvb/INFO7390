import re
from typing import Dict, List, Tuple

import streamlit as st

from app.auditor import TriageAuditor
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


def _render_matrix(flags: List[Dict]) -> None:
    buckets = {
        ("high", "high"): [],
        ("high", "medium"): [],
        ("medium", "high"): [],
        ("medium", "low"): [],
    }
    for item in flags:
        key = (item.get("stakes", "medium"), item.get("detectability", "low"))
        if key in buckets:
            buckets[key].append(item)

    st.sidebar.subheader("Stakes x Detectability")
    st.sidebar.caption("Counts and triggered flags by quadrant")
    st.sidebar.markdown(
        (
            f"**High Stakes / High Detectability:** {len(buckets[('high', 'high')])}  \n"
            f"**High Stakes / Medium Detectability:** {len(buckets[('high', 'medium')])}  \n"
            f"**Medium Stakes / High Detectability:** {len(buckets[('medium', 'high')])}  \n"
            f"**Medium Stakes / Low Detectability:** {len(buckets[('medium', 'low')])}"
        )
    )

    for quadrant, items in buckets.items():
        names = [FLAG_LABELS.get(i["flag_id"], i["flag_id"]) for i in items if i.get("flagged")]
        label = f"{quadrant[0].title()} / {quadrant[1].title()}"
        st.sidebar.write(f"{label}: {', '.join(names) if names else 'No triggered flags'}")


def main() -> None:
    st.set_page_config(page_title="Plausibility Auditor Dashboard", layout="wide")
    st.title("Layer 4 - Triage Dashboard")
    st.caption("RAG answer + automated plausibility triage over five flag categories")

    with st.sidebar:
        st.header("Query Controls")
        query = st.text_area(
            "Question",
            "What does recent evidence suggest about metformin benefits in type 2 diabetes?",
            height=120,
        )
        top_k = st.slider("Retrieved sources (top_k)", min_value=2, max_value=8, value=4)
        run = st.button("Run Audit", type="primary")

    if not run:
        st.info("Set a query and click 'Run Audit'.")
        return

    pipeline = get_pipeline()
    auditor = TriageAuditor(groq_client=pipeline.groq, generation_model=pipeline.generation_model_name)

    with st.spinner("Running retrieval, generation, and audit checks..."):
        retrieved = pipeline.retrieve(query, top_k=top_k)
        answer = pipeline.generate(query, retrieved)
        audit = auditor.run_all(query, answer, retrieved)

    _render_matrix(audit["flags"])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Generated Answer (Annotated)")
        st.markdown(_annotate_answer(answer, audit["flags"]), unsafe_allow_html=True)
    with col2:
        st.subheader("Confidence Profile")
        st.metric("Audit Score", f"{audit['audit_score']:.3f}")
        st.metric("Flags Triggered", f"{audit['flagged_count']} / {audit['total_flags_checked']}")

    st.subheader("Per-Flag Auditor Reasoning")
    for item in audit["flags"]:
        label = FLAG_LABELS.get(item["flag_id"], item["flag_id"])
        status = "TRIGGERED" if item.get("flagged") else "clear"
        with st.expander(f"{label} - {status}"):
            st.write(f"**Stakes:** {item.get('stakes', 'n/a')}")
            st.write(f"**Detectability:** {item.get('detectability', 'n/a')}")
            st.write(f"**Reason:** {item.get('reason', '')}")
            findings = item.get("findings", [])
            if findings:
                st.write("**Findings:**")
                for finding in findings:
                    st.write(f"- {finding}")
            if "metrics" in item:
                st.write("**Metrics:**")
                st.json(item["metrics"])

    st.subheader("Retrieved Sources")
    docs = retrieved.get("documents", [[]])[0]
    metas = retrieved.get("metadatas", [[]])[0]
    distances = retrieved.get("distances", [[]])[0]
    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        with st.expander(f"Source {i} - distance {distance:.4f}"):
            st.write(f"**Title:** {meta.get('title', 'Untitled')}")
            st.write(f"**URL:** {meta.get('source', '')}")
            st.write(doc)


if __name__ == "__main__":
    main()
