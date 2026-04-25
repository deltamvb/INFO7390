import argparse
import json

from app.auditor import TriageAuditor
from app.rag import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query local RAG pipeline")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--top-k", type=int, default=4, help="Number of retrieved chunks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag = RAGPipeline()
    retrieved = rag.retrieve(args.question, top_k=args.top_k)
    answer = rag.generate(args.question, retrieved)
    auditor = TriageAuditor(groq_client=rag.groq, generation_model=rag.generation_model_name)
    audit = auditor.run_all(args.question, answer, retrieved)

    print("\n=== Answer ===")
    print(answer)

    print("\n=== Retrieved Sources ===")
    docs = retrieved.get("documents", [[]])[0]
    metas = retrieved.get("metadatas", [[]])[0]
    distances = retrieved.get("distances", [[]])[0]
    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
        preview = doc.replace("\n", " ")[:220]
        print(
            f"[Source {i}] distance={distance:.4f} | {meta.get('title', 'Untitled')} | {meta.get('source', '')}"
        )
        print(f"  {preview}...")

    print("\n=== Auditor Report (Layer 2) ===")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
