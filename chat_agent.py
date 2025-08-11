# chat_agent.py
import re
import os
os.environ["LLAMA_LOG_LEVEL"] = "40"  # Suppress llama.cpp logs
from llama_cpp import Llama
from rag_chain import BibleRAG
from bible_loader import Bible
from thefuzz import process
from bootstrap_model import bootstrap_model


class BibleChatAgent:
    def __init__(self, bible_data_path):
        self.chat_history = []
        self.rag = BibleRAG(bible_data_path)
        self.bible = Bible(bible_data_path)

        self.rag.load_or_build_vectorstore()

        # Pick model file (small/large) and ensure it exists locally
        model_path = bootstrap_model()

        # Basic, portable params (no psutil required)
        cores = os.cpu_count() or 4
        n_ctx = int(os.getenv("N_CTX", "4096"))
        n_batch = int(os.getenv("N_BATCH", "1024"))

        print(f"⏳ Loading local LLM from: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=cores,
            n_batch=n_batch,
            verbose=False,
            chat_format=None
        )
        print("✅ LLM loaded.")
        # Expanded casual responses
        self.casual_responses = {
            "hello": "Hey there! Ready to dive into the Word?",
            "hi": "Hi! What would you like to explore today?",
            "hey": "Hey! How can I help your study?",
            "hey there": "Hi there! Feel free to ask anything.",
            "yo": "Yo! Bible questions welcome.",
            "peace be with you": "And also with you. Let’s explore the scriptures together.",
            "blessings": "Blessings to you too. How can I serve your study?",
            "good morning": "Good morning! Let’s begin today’s study.",
            "good afternoon": "Good afternoon! What would you like to look at?",
            "good evening": "Good evening. I’m here to support your Bible journey.",
            "shalom": "Shalom! How can I assist with your Bible questions?",
            "greetings": "Greetings! I’m happy to help with your study."
        }

        # Keys used for greeting fuzzy-match
        self.greeting_keys = list(self.casual_responses.keys())
    def _build_answer_prompt(self, context: str, question: str, allowed_refs: list[str]) -> str:
        allowed = "; ".join(allowed_refs)
        return (
            "INSTRUCTIONS:\n"
            "- You are a Bible study assistant.\n"
            "- Use ONLY the Context passages below; do NOT quote or cite anything not present in Context.\n"
            "- If Context is insufficient, say so briefly.\n"
            "- Begin with a concise summary (2–4 sentences) that directly answers the question.\n"
            "- You may weave in 1–3 direct Bible verse quotations naturally within the summary **only if they add clarity**.\n"
            "- Avoid bullet points or separate verse lists unless absolutely necessary.\n"
            "- Do NOT fabricate content or references.\n"
            "- Do NOT include a line that starts with 'Supporting verses:'.\n\n"
            f"Allowed references (must ONLY cite from this set): {allowed}\n\n"
            f"Context passages:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:\n"
        )

    def _extract_book_chapter(self, question: str):
        """Return (book_lower, chapter_str_or_None) inferred from the question.
        Special-case Nicodemus → John 3."""
        q = question.lower()

        # direct pattern like "john 3", "1 john 4", "romans 8"
        m = re.search(r"\b([1-3]?\s?[a-z]+)\s+(\d{1,3})\b", q)
        book = chapter = None
        if m:
            book = m.group(1).strip()          # e.g., "john" or "1 john"
            chapter = m.group(2).strip()       # e.g., "3"

        # phrases like "gospel of john", "book of john"
        if not book and ("gospel of john" in q or "book of john" in q or "john's gospel" in q):
            book = "john"

        # Nicodemus almost always refers to John 3
        if "nicodemus" in q and (book is None or book == "john"):
            book = "john"
            chapter = chapter or "3"

        # normalize spacing for comparison (metadata likely lowercased file stems)
        if book:
            book = book.replace("  ", " ").strip().lower()

        return book, chapter

    def ask(self, question: str) -> str:
        memory_context = "\n".join(
            [f"Q: {q}\nA: {a}" for q, a in self.chat_history[-3:]]
        ) if self.chat_history else ""
        question = question.strip()

        category = self.classify_question(question)

        # 1. Verse lookup
        if category == "verse_lookup":
            answer = self.handle_verse_reference(question, memory_context)

        # 2. Bible question
        elif category == "bible_question":
            try:
                docs = self.rag.query_docs(question, k=10)
            except Exception:
                docs = []

            # Try to infer a target book/chapter from the question
            book_hint, chap_hint = self._extract_book_chapter(question)

            # Narrow by book / chapter if we found a hint
            if docs and (book_hint or chap_hint):
                filtered = []
                for d in docs:
                    meta = d.metadata or {}
                    mb = str(meta.get("book", "")).replace("_", " ").strip().lower()
                    mc = str(meta.get("chapter", "")).strip().lower()
                    keep = True
                    if book_hint and mb != book_hint:
                        keep = False
                    if chap_hint and mc != chap_hint:
                        keep = False
                    if keep:
                        filtered.append(d)
                docs = filtered[:5] if filtered else docs[:5]

            # If we have passages, format them with refs; else provide fallback
            if docs:
                # Build the formatted context and the list of allowed references
                ctx_lines = []
                allowed_refs = []
                for d in docs:
                    m = d.metadata or {}
                    book = (m.get('book','') or '').strip()
                    chap = str(m.get('chapter','?'))
                    verse = str(m.get('verse','?'))
                    ref = f"{book} {chap}:{verse}".strip()
                    allowed_refs.append(ref)
                    ctx_lines.append(f"{ref} — {d.page_content}")

                context = "\n".join(ctx_lines)
                prompt = self._build_answer_prompt(context, question, allowed_refs)
                answer = self.run_llm(prompt)

            else:
                context = "No directly relevant passages were found in the index for this query."
                prompt = (
                    "INSTRUCTIONS:\n"
                    "- You are a Bible study assistant.\n"
                    "- No specific passages were retrieved; answer from general biblical teaching in 2–3 sentences.\n"
                    "- If appropriate, you may reference well-known verses by book name (e.g., John 3:16) without quoting.\n"
                    "- Do NOT invent tasks, math problems, hypotheticals, or numbered lists.\n\n"
                    f"Context:\n{context}\n\n"
                    f"Question:\n{question}\n\n"
                    "Answer:"
                )
                answer = self.run_llm(prompt)
                answer += "\n\nNote: No specific passages were retrieved for this question; response is based on general biblical teaching."

        # 3. Casual chat (fallback)
        else:
            matches = process.extract(question.lower(), self.casual_responses.keys(), limit=3)
            if matches and matches[0][1] > 75:
                answer = self.casual_responses[matches[0][0]]
            else:
                answer = (
                    "Hi there! I'm here to help with your Bible study. "
                    "You can ask about a verse, a topic, or just say hello."
                )

        self.chat_history.append((question, answer))
        return answer

    def classify_question(self, question: str) -> str:
        q = question.strip().lower()
        wc = len(q.split())

        # 1) Verse reference pattern: e.g., "john 3:16" or "1 john 4:8-10"
        if re.search(r"\b[1-3]?\s?[a-z]+(?:\s[a-z]+)?\s+\d+:\d+(?:[-–]\d+)?\b", q):
            return "verse_lookup"

        # 2) Greetings via fuzzy match (ONLY for short inputs)
        if wc <= 4 or len(q) <= 25:
            match = process.extractOne(q, self.greeting_keys)
            if match and match[1] >= 90:  # stricter threshold
                return "casual_chat"

        # 3) Heuristics for bible-themed questions
        bible_terms = [
            "bible","scripture","verse","passage","god","jesus","holy spirit","paul","john",
            "gospel","commandment","sin","grace","salvation","love","faith","hope","spirit",
            "pray","prayer","wisdom","proverb","psalm","law","covenant","testament","nicodemus"
        ]
        questiony_phrases = [
            "what does","what do","what is","teach","say about","meaning of","where does","how does",
            "tell me about","explain","describe"
        ]
        if any(p in q for p in questiony_phrases) or any(t in q for t in bible_terms):
            return "bible_question"

        # 4) Safe default: treat as bible_question rather than casual
        return "bible_question"

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

    def _tidy_answer(self, text: str) -> str:
        """
        If the last line appears to have an unclosed double-quote, drop it.
        Safer than the previous regex which could remove valid lines.
        """
        lines = text.rstrip().splitlines()
        if not lines:
            return text

        last = lines[-1].strip()

        # Count ASCII double-quotes and check curly pair balance
        ascii_unbalanced = (last.count('"') % 2) == 1
        curly_unbalanced = last.count("“") != last.count("”")

        if ascii_unbalanced or curly_unbalanced:
            # Only drop if it actually looks like a verse line (has Chap:Verse)
            if re.search(r"\b\d+:\d+\b", last):
                lines = lines[:-1]

        return "\n".join(lines).rstrip()


    def run_llm(self, prompt):
        out = self.llm(
            prompt,
            max_tokens=int(os.getenv("MAX_TOKENS", "500")),
            temperature=0.33,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.2,
            stop=[
                "\nUser:", "\nQ:", "\nQuestion:", "Answer format:"
            ]
        )
        text = out["choices"][0].get("text", "").strip()
        text = self._tidy_answer(text)
        return text

