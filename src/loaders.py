"""Turn uploaded files into plain text.

Each document becomes a list of pages so we can cite page numbers later.
A .txt file is treated as a single page.
"""

from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass
class Page:
    doc_name: str
    page_number: int  # 1-based, like a human would count
    text: str


def load_txt(name: str, data: bytes) -> list[Page]:
    # errors="replace" keeps going if the file has a few bad bytes
    text = data.decode("utf-8", errors="replace")
    return [Page(doc_name=name, page_number=1, text=text)]


def load_pdf(name: str, data: bytes) -> list[Page]:
    reader = PdfReader(BytesIO(data))
    pages = []
    for i, pdf_page in enumerate(reader.pages, start=1):
        text = pdf_page.extract_text() or ""  # scanned/image-only pages return nothing
        pages.append(Page(doc_name=name, page_number=i, text=text))
    return pages


def load_document(name: str, data: bytes) -> list[Page]:
    """Pick the right loader based on the file extension."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        return load_pdf(name, data)
    if lower.endswith(".txt"):
        return load_txt(name, data)
    raise ValueError(f"Unsupported file type: {name}")
