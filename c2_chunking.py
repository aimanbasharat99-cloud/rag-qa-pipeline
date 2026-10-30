from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


# ── Chunker ───────────────────────────────────────────────────────────────────

class FixedChunker:

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.splitter   = CharacterTextSplitter(
            chunk_size    = chunk_size,   # max characters per chunk
            chunk_overlap = overlap,      # how many chars the next chunk re-uses from the previous one
            separator     = "",           # pure character-level, no semantic splitting
        )

    def chunk_many(self, docs: list[Document]) -> list[Document]:
        all_chunks = []
        for doc in docs:
            chunks = self.splitter.split_documents([doc])   # splits one Document into many
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i              # tag each chunk with its position index
            all_chunks.extend(chunks)                       # collect across all documents
            print(f"  [chunked] {doc.metadata.get('source')} → {len(chunks)} chunks")
        return all_chunks

    def inspect(self, chunks: list[Document]) -> None:
        print(f"\n{'─'*55}")
        print(f"Total chunks : {len(chunks)}")
        print(f"Chunk size   : {self.chunk_size} chars")
        print(f"Overlap      : {self.overlap} chars\n")
        for i, chunk in enumerate(chunks[:10]):
            print(f"  [{i}] source   : {chunk.metadata.get('source')}")
            print(f"       chunk_id : {chunk.metadata.get('chunk_id')}")
            print(f"       chars    : {len(chunk.page_content)}")
            print(f"       preview  : {chunk.page_content[:80].replace(chr(10), ' ')!r}")
            print()
        if len(chunks) > 10:
            print(f"  ... and {len(chunks) - 5} more chunks")
        print(f"{'─'*55}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    from c1_load_documents import DocumentLoader

    loader  = DocumentLoader()
    chunker = FixedChunker(chunk_size=500, overlap=50)

    docs   = loader.load_folder("data/")
    chunks = chunker.chunk_many(docs)
    chunker.inspect(chunks)



