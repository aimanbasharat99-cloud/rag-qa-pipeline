from c3_embed_store import EmbedStore
from c6_llm import LLMClient


QUESTION = "What did John McCarthy do in 1956?"


# ─────────────────────────────────────────────
# Format retrieved documents
# ─────────────────────────────────────────────

def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] (source: {d.metadata.get('source', 'unknown')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )


# ─────────────────────────────────────────────
# Build prompt
# ─────────────────────────────────────────────

def build_prompt(question, docs):
    context = format_docs(docs)

    return f"""
You are a helpful AI assistant.

Use ONLY the context below to answer the question.
If the answer is not present, say "I don't know".

IMPORTANT:
- Do NOT repeat the context
- Do NOT repeat the question
- Give ONLY the final answer in 1-2 sentences

Context:
{context}

Question: {question}

Answer:
""".strip()


# ─────────────────────────────────────────────
# QUERY PIPELINE
# ─────────────────────────────────────────────

def query(question: str):
    print("\n── QUERY START ──")

    # Load components
    store = EmbedStore()
    llm = LLMClient()

    # Retriever (modern API)
    retriever = store.db.as_retriever(search_kwargs={"k": 3})

    # Step 1: Retrieve documents
    docs = retriever.invoke(question)

    # Step 2: Build prompt
    prompt = build_prompt(question, docs)

    # Step 3: Generate answer
    answer = llm.generate(prompt)

    print("\n────────────────────────────")
    print("ANSWER:\n")
    print(answer)

    print("\nSOURCES:\n")
    unique_sources = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src not in unique_sources:
            print("-", src)
            unique_sources.add(src)

    print("────────────────────────────\n")

    return answer


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    query(QUESTION)