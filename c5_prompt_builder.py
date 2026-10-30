from langchain_core.prompts import PromptTemplate

from c4_retrieve import Retriever
from c3_embed_store import EmbedStore


# ── Prompt builder ────────────────────────────────────────────────────────────

class PromptBuilder:

    def __init__(self, max_context_chars: int = 2000):
        self.max_context_chars = max_context_chars
        self.template = PromptTemplate.from_template(   # {context} and {question} are the placeholders
            "Use ONLY the context below to answer the question.\n"
            "If the answer is not in the context, say 'I don't know'.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )

    def build(self, query: str, hits: list[dict]) -> str:
        context = self._build_context(hits)
        return self.template.format(context=context, question=query)  # fills placeholders → returns plain string

    def inspect(self, prompt: str) -> None:
        print(f"\n{'─'*55}")
        print(f"Prompt ({len(prompt)} chars)\n")
        print(prompt)
        print(f"{'─'*55}\n")

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_context(self, hits: list[dict]) -> str:
        # Concatenate retrieved chunks into a single context string.
        # Chunks are added in order of relevance (best score first).
        # We stop early if adding the next chunk would exceed max_context_chars,
        # so the prompt never gets too long for the LLM.
        # Each chunk is labeled with its rank and source file for traceability.
        chunks = []
        total  = 0

        for i, hit in enumerate(hits):
            content = hit["content"].strip()
            source  = hit["metadata"].get("source", "?")

            if total + len(content) > self.max_context_chars:   # budget check
                break

            chunks.append(f"[{i+1}] (source: {source})\n{content}")
            total += len(content)

        return "\n\n".join(chunks)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    store     = EmbedStore(persist_dir="chroma_db", collection_name="rag")
    retriever = Retriever(store=store, top_k=3)
    builder   = PromptBuilder(max_context_chars=2000)

    query  = "What is RAG?"
    hits   = retriever.retrieve(query)
    prompt = builder.build(query, hits)
    builder.inspect(prompt)
