"""Build the Chroma vector index from the .txt files in data/.

Run once before main.py or app.py:
    python ingest.py
"""
from c1_load_documents import DocumentLoader
from c2_chunking import FixedChunker
from c3_embed_store import EmbedStore

DATA_DIR = "data"
PERSIST_DIR = "chroma_db"


def build_index(data_dir: str = DATA_DIR, persist_dir: str = PERSIST_DIR, embeddings=None) -> EmbedStore:
    docs = DocumentLoader().load_folder(data_dir)
    chunks = FixedChunker(chunk_size=500, overlap=50).chunk_many(docs)

    store = EmbedStore(persist_dir=persist_dir, embeddings=embeddings)
    store.db.reset_collection()  # start clean so re-running doesn't duplicate chunks
    store.add_documents(chunks)
    return store


if __name__ == "__main__":
    build_index()
    print("Index built. Now run: python main.py  or  python app.py")
