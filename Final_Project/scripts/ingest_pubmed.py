import argparse

from tqdm import tqdm

from app.pubmed import fetch_pubmed_abstracts, search_pubmed_ids
from app.rag import RAGPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts and index in ChromaDB.")
    parser.add_argument("--query", required=True, help="PubMed search query")
    parser.add_argument("--max-results", type=int, default=40, help="Number of PubMed records")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pmids = search_pubmed_ids(args.query, max_results=args.max_results)
    docs = fetch_pubmed_abstracts(pmids)

    rag = RAGPipeline()
    indexed = 0
    for doc in tqdm(docs, desc="Indexing documents"):
        indexed += rag.upsert_documents([doc])

    print(f"Indexed {indexed} abstracts into ChromaDB.")


if __name__ == "__main__":
    main()
