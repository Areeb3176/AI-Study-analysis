# backend/extractor.py
import pdfplumber
import re
from typing import List

def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def clean_text(txt: str) -> str:
    txt = txt.replace("\r", " ")
    txt = re.sub(r'\n{2,}', '\n', txt)
    return txt.strip()

def chunk_text(txt: str, words_per_chunk: int = 1000) -> List[dict]:
    words = txt.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i:i+words_per_chunk]
        chunk_text = " ".join(chunk_words)
        chunk_id = (i // words_per_chunk) + 1
        chunks.append({"id": chunk_id, "text": chunk_text})
    return chunks
