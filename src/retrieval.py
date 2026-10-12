"""Keyword search over chunks using BM25.

BM25 is the classic search-engine scoring formula. For each query word it asks:
  - How often does the word appear in this chunk? (more = better, with diminishing returns)
  - How rare is the word across all chunks? (rare words like "transformer" matter more than "data")
  - How long is the chunk? (long chunks shouldn't win just by containing more words)

A chunk that shares no words with the question scores 0, which is how we
detect "the documents don't cover this".
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from chunking import Chunk

# Very common words that carry little meaning for search.
STOPWORDS = set(
    "a an and are as at be but by can did do does for from has have how i if in "
    "into is it its me my of on or our so than that the their them then there "
    "these they this to was we were what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1  # how quickly repeated words stop adding score
        self.b = b  # how strongly to penalize long chunks
        self.chunk_tokens = [Counter(tokenize(c.text)) for c in chunks]
        self.chunk_lengths = [sum(t.values()) for t in self.chunk_tokens]
        self.avg_length = sum(self.chunk_lengths) / len(chunks) if chunks else 0

        # Document frequency: in how many chunks does each word appear?
        doc_freq = Counter()
        for tokens in self.chunk_tokens:
            doc_freq.update(tokens.keys())
        n = len(chunks)
        self.idf = {
            word: math.log(1 + (n - df + 0.5) / (df + 0.5)) for word, df in doc_freq.items()
        }

    def score(self, query_words: list[str], i: int) -> float:
        tokens = self.chunk_tokens[i]
        length_norm = 1 - self.b + self.b * self.chunk_lengths[i] / (self.avg_length or 1)
        total = 0.0
        for word in query_words:
            tf = tokens.get(word, 0)
            if tf == 0:
                continue
            total += self.idf[word] * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return total

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list[SearchResult]:
        query_words = list(dict.fromkeys(tokenize(query)))  # unique, keep order
        results = [
            SearchResult(chunk=c, score=self.score(query_words, i))
            for i, c in enumerate(self.chunks)
        ]
        results = [r for r in results if r.score > min_score]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
