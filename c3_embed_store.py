from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class EmbedStore:
    def __init__(
        self,
        persist_dir="chroma_db",
        collection_name="rag",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        embeddings=None,
    ):
        print("  [loading embedding model] ...")

        self.embeddings = embeddings or HuggingFaceEmbeddings(model_name=model_name)

        self.db = Chroma(
            persist_directory=persist_dir,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},  # relevance score = cosine similarity
        )

    def add_documents(self, chunks):
        self.db.add_documents(chunks)  # langchain_chroma persists automatically
        print(f"  [stored] {len(chunks)} chunks → total in db: {self.db._collection.count()}")

    def search(self, query: str, k: int = 3):
        """Return (Document, relevance) pairs, best first. Relevance is in [0, 1]; higher is better."""
        return self.db.similarity_search_with_relevance_scores(query, k=k)
