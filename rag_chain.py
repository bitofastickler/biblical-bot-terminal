# rag_chain.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from pathlib import Path
import json

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class BibleRAG:
    def __init__(self, bible_data_path: Path):
        self.bible_data_path = Path(bible_data_path)
        self.embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        self.vectorstore = None

    def load_bible_documents(self):
        documents = []
        for book_file in self.bible_data_path.glob("*.json"):
            book = book_file.stem
            with open(book_file, encoding="utf-8") as f:
                verses = json.load(f)
            for v in verses:
                if v.get("type") != "paragraph text":
                    continue
                chapter = v["chapterNumber"]
                verse = v["verseNumber"]
                text = v["value"]
                metadata = {"book": book, "chapter": chapter, "verse": verse}
                documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def load_or_build_vectorstore(self, persist_directory=".chromadb"):

        try:
            self.vectorstore = Chroma(
                embedding_function=self.embedding,
                persist_directory=persist_directory
            )
            # Check for actual data
            results = self.vectorstore.similarity_search("Jesus", k=1)
            if not results:
                raise RuntimeError("Vectorstore appears empty.")
            print("✅ Vector store loaded from disk.")
        except Exception as e:
            print(f"⚠️ Vectorstore error ({e}). Building a new one...")
            documents = self.load_bible_documents()
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
            chunks = splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding,
                persist_directory=persist_directory
            )
            self.vectorstore.persist()
            print("✅ Vector store built and saved.")


        # add this method
    def query_docs(self, question: str, k: int = 5):
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not initialized")
        return self.vectorstore.similarity_search(question, k=k)

    # keep existing query if you like, or refactor it to call query_docs
    def query(self, question: str) -> str:
        docs = self.query_docs(question, k=5)
        return "\n".join([d.page_content for d in docs])
