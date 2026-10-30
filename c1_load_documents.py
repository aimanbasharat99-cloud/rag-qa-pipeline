from pathlib import Path

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document


# ── Loader ────────────────────────────────────────────────────────────────────

class DocumentLoader:

    def load_file(self, path: str | Path) -> Document:
        loader = TextLoader(str(path), encoding="utf-8")   # reads a single .txt file
        return loader.load()[0]                             # .load() returns a list; grab the one document

    def load_folder(self, folder: str | Path, recursive: bool = False) -> list[Document]:
        loader = DirectoryLoader(
            str(folder),                                    # root folder to scan
            glob          = "**/*.txt" if recursive else "*.txt",  # file pattern: deep scan or top-level only
            loader_cls    = TextLoader,                     # which loader to use for each matched file
            loader_kwargs = {"encoding": "utf-8"},          # extra args forwarded to TextLoader
            show_progress = True,                           # print a progress bar while loading
        )
        docs = loader.load()                                # loads all matched files into Document objects
        for doc in docs:
            print(f"  [OK]   {Path(doc.metadata['source']).name} — {len(doc.page_content)} chars")
        return docs

    def inspect(self, docs: list[Document]) -> None:
        print(f"\n{'─'*55}")
        print(f"Loaded {len(docs)} document(s)\n")
        for i, doc in enumerate(docs):
            print(f"  [{i}] source : {doc.metadata.get('source')}")
            print(f"       chars  : {len(doc.page_content)}")
            print(f"       preview: {doc.page_content[:80].replace(chr(10), ' ')!r}")
            print()
        print(f"{'─'*55}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    loader = DocumentLoader()
    docs   = loader.load_folder("data/")
    loader.inspect(docs)
