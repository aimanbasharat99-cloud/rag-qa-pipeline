from c3_embed_store import EmbedStore


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:

    def __init__(self, store: EmbedStore, top_k: int = 3):
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        # LangChain's Chroma embeds the query and searches in one call
        results = self.store.search(query, k=self.top_k)
        # returns list of (Document, relevance) pairs, sorted best-first

        hits = []
        for doc, relevance in results:
            hits.append({
                "content":  doc.page_content,
                "metadata": doc.metadata,
                "score":    round(relevance, 4),      # cosine similarity: 1 = identical
            })

        return hits

    def inspect(self, query: str, hits: list[dict]) -> None:
        print(f"\n{'─'*55}")
        print(f"Query : {query!r}")
        print(f"Top-{self.top_k} results\n")
        for i, hit in enumerate(hits):
            print(f"  [{i}] score    : {hit['score']}")
            print(f"       source   : {hit['metadata'].get('source')}")
            print(f"       chunk_id : {hit['metadata'].get('chunk_id')}")
            print(f"       preview  : {hit['content'][:100].replace(chr(10), ' ')!r}")
            print()
        print(f"{'─'*55}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    store     = EmbedStore(persist_dir="chroma_db", collection_name="rag")
    retriever = Retriever(store=store, top_k=3)

    query = "What is RAG?"
    hits  = retriever.retrieve(query)
    retriever.inspect(query, hits)
