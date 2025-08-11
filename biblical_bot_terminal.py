# main.py
from chat_agent import BibleChatAgent

if __name__ == "__main__":
    print("\n📖 Welcome to the Offline Biblical Bot CLI! Type 'exit' to quit.\n")
    agent = BibleChatAgent("./bible/bible_books")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye. May your study be blessed.")
            break

        response = agent.ask(user_input)
        print(f"Bot: {response}\n")


# chat_agent.py
import re
from llama_cpp import Llama
from rag_chain import BibleRAG
from bible_loader import Bible
from thefuzz import process

class BibleChatAgent:
    def __init__(self, bible_data_path):
        self.chat_history = []
        self.rag = BibleRAG(bible_data_path)
        self.bible = Bible(bible_data_path)

        self.rag.load_or_build_vectorstore()

        print("⏳ Loading local LLM (Phi-2)...")
        self.llm = Llama(model_path="./models/phi-2.gguf", n_ctx=2048, n_threads=4)

        self.casual_responses = {
            "hello": "Hello there! How can I serve your study today?",
            "hi": "Hi! Feel free to ask anything about the Bible.",
            "thank you": "You're so welcome. It’s a joy to walk this journey with you.",
            "thanks": "Of course! I’m glad to be of help.",
            "good morning": "Good morning! Ready to dive into the Word?",
            "good evening": "Good evening. I’m here if you’d like to explore scripture."
        }

    def ask(self, question: str) -> str:
        memory_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.chat_history[-3:]]) if self.chat_history else ""
        question = question.strip()

        category = self.classify_question(question)

        if category == "verse_lookup":
            answer = self.handle_verse_reference(question, memory_context)
        elif category == "bible_question":
            context = self.rag.query(question)
            prompt = f"You are a helpful Bible study assistant. Use scripture to answer clearly.\n\nContext:\n{context}\n\nQuestion:\n{question}\n"
            answer = self.run_llm(prompt)
        else:
            match, score = process.extractOne(question.lower(), self.casual_responses.keys())
            if score > 70:
                answer = self.casual_responses[match]
            else:
                answer = "Hello! I’m here to help with Bible study questions. Ask me anything."

        self.chat_history.append((question, answer))
        return answer

    def classify_question(self, question: str) -> str:
        prompt = f"Classify the user input as one of the following: verse_lookup, bible_question, casual_chat.\nInput: {question}\nCategory:"
        output = self.run_llm(prompt)
        return output.strip().lower()

    def handle_verse_reference(self, question, memory_context):
        ref = re.search(r'([1-3]?\s?[A-Za-z ]+?)\s+(\d+):(\d+)(?:[-–](\d+))?', question)
        if ref:
            book = ref.group(1).strip()
            chapter = int(ref.group(2))
            start_verse = int(ref.group(3))
            end_verse = int(ref.group(4)) if ref.group(4) else start_verse
            verses = []
            for v in range(start_verse, end_verse + 1):
                text = self.bible.get_verse(book, chapter, v)
                if "not found" not in text.lower():
                    verses.append(f"{chapter}:{v} — {text}")
            combined = "\n".join(verses)
            prompt = f"Context:\n{memory_context}\n{combined}\n\nQuestion: {question}\nAnswer:"
            return self.run_llm(prompt)
        return "I'm sorry, I couldn't understand the verse reference."

    def run_llm(self, prompt):
        output = self.llm(prompt, max_tokens=256, stop=["\n", "Q:"])
        return output["choices"][0]["text"].strip()


# rag_chain.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
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
            # Try a dummy call to ensure it's not empty
            _ = self.vectorstore.similarity_search("Jesus", k=1)
            print("✅ Vector store loaded from disk.")
        except Exception:
            print("⚠️ No existing vectorstore found. Building a new one...")
            documents = self.load_bible_documents()
            splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
            chunks = splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding,
                persist_directory=persist_directory
            )
            self.vectorstore.persist()

    def query(self, question: str) -> str:
        if not self.vectorstore:
            raise RuntimeError("Vectorstore not initialized")
        results = self.vectorstore.similarity_search(question, k=5)
        return "\n".join([doc.page_content for doc in results])


# bible_loader.py
import json
from pathlib import Path
from thefuzz import process

class Bible:
    def __init__(self, data_path):
        self.verse_map = {}
        self.book_aliases = {}
        self.load_bible(data_path)

    def load_bible(self, data_path):
        path = Path(data_path)
        for file in path.glob("*.json"):
            book_name = file.stem.lower()
            with open(file, encoding="utf-8") as f:
                verses = json.load(f)
            for v in verses:
                if v.get("type") != "paragraph text":
                    continue
                key = (book_name, str(v["chapterNumber"]), str(v["verseNumber"]))
                self.verse_map[key] = v["value"]
            self.book_aliases[book_name.replace("_", " ")] = book_name

    def get_verse(self, book, chapter, verse):
        normalized = book.lower().replace("_", " ")
        match, score = process.extractOne(normalized, self.book_aliases.keys())
        if score < 70:
            return "Hmmm, not seeing that book. Please check your spelling."
        key = (self.book_aliases[match], str(chapter), str(verse))
        return self.verse_map.get(key, "Verse not found.")
