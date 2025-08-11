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