import os

import gradio as gr

from c3_embed_store import EmbedStore
from c6_llm import LLMClient


# ── CONFIG ─────────────────────────────────────────────
TOP_K_DEFAULT = 3
MIN_SCORE_DEFAULT = 0.5


# ── LOAD ONCE ──────────────────────────────────────────
print("Loading RAG pipeline...")

store = EmbedStore()
llm = LLMClient()

print("Pipeline ready.")


# ── HELPERS ────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] (source: {d.metadata.get('source')})\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def build_prompt(question, docs):
    context = format_docs(docs)

    return f"""
You are a helpful AI assistant.

Use ONLY the context below to answer the question.
If the answer is not present, say "I don't know".

IMPORTANT:
- Do NOT repeat the context
- Give ONLY the final answer

Context:
{context}

Question: {question}

Answer:
""".strip()


# ── RAG FUNCTION ───────────────────────────────────────
def rag_pipeline(question, top_k, min_score):

    if not question.strip():
        return "⚠️ Enter a question", "", "No input"

    try:
        # (Document, relevance) pairs; relevance = cosine similarity in [0, 1]
        results = store.search(question, k=int(top_k))

        if not results:
            return "⚠️ No documents retrieved", "", ""

        filtered = [
            {
                "doc": doc,
                "score": float(score)
            }
            for doc, score in results
            if score >= float(min_score)
        ]

        if len(filtered) == 0:
            return (
                "⚠️ No answer found",
                "",
                "No sources passed the similarity threshold → lower MIN_SCORE"
            )

        docs_filtered = [f["doc"] for f in filtered]

        prompt = build_prompt(question, docs_filtered)

        answer = llm.generate(prompt)

        # 🔥 show full source info (REQUIRED)
        sources = "\n\n".join([
            f"{f['doc'].metadata.get('source')} | "
            f"chunk {f['doc'].metadata.get('chunk_id', '?')} | "
            f"score {round(f['score'], 4)}"
            for f in filtered
        ])

        return answer, sources, ""

    except Exception as e:
        return "❌ Error", "", str(e)


# ── GRADIO UI ──────────────────────────────────────────
with gr.Blocks() as demo:

    gr.Markdown("# 📚 RAG Pipeline (Assignment UI)")

    question = gr.Textbox(label="Enter your question")

    with gr.Row():
        top_k = gr.Slider(1, 10, value=TOP_K_DEFAULT, step=1, label="TOP_K")
        min_score = gr.Slider(0.0, 1.0, value=MIN_SCORE_DEFAULT, step=0.05, label="MIN_SCORE")

    btn = gr.Button("Run RAG")

    answer_box = gr.Textbox(label="Answer", lines=6)
    sources_box = gr.Textbox(label="Sources (file | chunk | score)", lines=8)
    warning_box = gr.Textbox(label="Warning")

    btn.click(
        fn=rag_pipeline,
        inputs=[question, top_k, min_score],
        outputs=[answer_box, sources_box, warning_box]
    )


# ── LAUNCH ─────────────────────────────────────────────
# set GRADIO_SHARE=1 to get a temporary public link
demo.launch(share=os.getenv("GRADIO_SHARE") == "1")