import re
from document_loader import load_pdf


def create_chunks(text):
    raw_chunks = re.split(r"(?=Ques\.\s*\d+:)", text)

    chunks = [
        chunk.strip()
        for chunk in raw_chunks
        if chunk.strip().startswith("Ques.")
    ]

    return chunks