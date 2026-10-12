"""Split page text into overlapping chunks of words.

Why chunk? An LLM can only read a limited amount of text per request, and
smaller pieces make search more precise. Overlap keeps a sentence that falls
on a boundary from being cut off from its context.
"""

from dataclasses import dataclass

from loaders import Page


@dataclass
class Chunk:
    chunk_id: str  # e.g. "paper.pdf#p3-c2"
    doc_name: str
    page_number: int
    text: str


def split_words(words: list[str], chunk_size: int, overlap: int) -> list[list[str]]:
    """Slide a window of `chunk_size` words, stepping forward by chunk_size - overlap."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(words[start : start + chunk_size])
        if start + chunk_size >= len(words):
            break
    return pieces


def chunk_pages(pages: list[Page], chunk_size: int = 200, overlap: int = 40) -> list[Chunk]:
    chunks = []
    for page in pages:
        words = page.text.split()
        for i, piece in enumerate(split_words(words, chunk_size, overlap), start=1):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.doc_name}#p{page.page_number}-c{i}",
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    text=" ".join(piece),
                )
            )
    return chunks
