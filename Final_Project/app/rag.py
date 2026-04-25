from typing import Dict, List

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from app.config import CHROMA_DIR, EMBEDDING_MODEL, GENERATION_MODEL, GROQ_API_KEY


class RAGPipeline:
    def __init__(self, collection_name: str = "pubmed_abstracts") -> None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.groq = Groq(api_key=GROQ_API_KEY)
        self.generation_model_name = GENERATION_MODEL
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def upsert_documents(self, docs: List[Dict]) -> int:
        if not docs:
            return 0

        ids = [doc["id"] for doc in docs]
        texts = [doc["text"] for doc in docs]
        embeddings = self._embed(texts)
        metadatas = [
            {
                "pmid": doc["pmid"],
                "title": doc["title"][:512],
                "source": doc["source"],
            }
            for doc in docs
        ]

        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(docs)

    def retrieve(self, query: str, top_k: int = 4) -> Dict:
        query_embedding = self._embed([query])[0]
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def generate(self, query: str, retrieval_result: Dict) -> str:
        docs = retrieval_result.get("documents", [[]])[0]
        metas = retrieval_result.get("metadatas", [[]])[0]

        context_lines = []
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            context_lines.append(
                f"[Source {i}] {meta.get('title', 'Untitled')} ({meta.get('source', '')})\n{doc}"
            )
        context = "\n\n".join(context_lines)

        system_prompt = (
            "You are a biomedical research assistant. "
            "Answer only using the provided context. "
            "If context is insufficient, say what is missing."
        )
        user_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Write a concise answer with citations like [Source 1], [Source 2]."
        )

        full_prompt = (
            f"{system_prompt}\n\n"
            f"{user_prompt}"
        )
        response = self.groq.chat.completions.create(
            model=self.generation_model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
