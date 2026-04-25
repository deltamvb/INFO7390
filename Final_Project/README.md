# Layer 1 RAG Foundation (PubMed + ChromaDB + Groq)

This is the first layer of the expanded project scope:

- ingest PubMed abstracts
- embed with a local sentence-transformer model
- store vectors locally in ChromaDB
- retrieve relevant evidence from ChromaDB
- generate cited answers with Groq

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

Then set `GROQ_API_KEY` in `.env`.

If `python` points to a global install on Windows, always use the venv interpreter explicitly:

```bash
.\.venv\Scripts\python -m scripts.ingest_pubmed --query "metformin type 2 diabetes randomized trial" --max-results 40
.\.venv\Scripts\python -m scripts.query_rag --question "What does recent evidence suggest about metformin benefits in type 2 diabetes?" --top-k 4
```

## 2) Index PubMed abstracts

```bash
.\.venv\Scripts\python -m scripts.ingest_pubmed --query "metformin type 2 diabetes randomized trial" --max-results 40
```

## 3) Ask questions

```bash
.\.venv\Scripts\python -m scripts.query_rag --question "What does recent evidence suggest about metformin benefits in type 2 diabetes?" --top-k 4
```

The query script now also runs Layer 2 auditing and prints a JSON report with:

- Flag 2: claims exceeding data
- Flag 3: load-bearing citations
- Flag 5: handoff confidence laundering
- Flag 6: out-of-distribution confidence
- Flag 7: suspiciously clean convergence

## Project Structure

- `app/pubmed.py`: fetch PubMed IDs + abstracts via NCBI E-utilities
- `app/rag.py`: local embeddings, Chroma retrieval, and Groq answer generation
- `scripts/ingest_pubmed.py`: data ingestion/indexing entrypoint
- `scripts/query_rag.py`: query entrypoint

## Notes for Layer 2

Layer 2 (auditor agent) can plug directly into `scripts/query_rag.py` after retrieval and generation by adding:

- claim-evidence checks
- citation support checks
- confidence boundary checks
- out-of-distribution checks
- convergence checks

## Layer 3: Adversarial Demonstration

Run case-study validation that stress-tests the 5 implemented flags:

```bash
.\.venv\Scripts\python -m scripts.run_adversarial_demo --inject-docs --top-k 4
```

Outputs:

- Terminal summary of expected vs triggered flags per case
- JSON report at `data/adversarial_results.json`

Case definitions live in `data/adversarial_cases.json`. You can add your own cases by editing:

- `query`
- `expected_flags`
- `injected_docs`

## Layer 4: Streamlit Triage Dashboard

Launch the dashboard:

```bash
.\.venv\Scripts\python -m streamlit run streamlit_app.py
```

Dashboard includes:

- generated answer with inline flag annotations
- stakes x detectability matrix in the sidebar
- per-flag expandable reasoning panels
- overall audit score and trigger count
