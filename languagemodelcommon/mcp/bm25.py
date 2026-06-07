"""BM25 Okapi search index for ranked keyword retrieval.

Shared by tool_catalog and resource_catalog for consistent search behavior.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


@dataclass
class BM25Index:
    """In-memory BM25 Okapi index over tokenized documents."""

    k1: float = 1.5
    b: float = 0.75
    corpus: list[list[str]] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    n_docs: int = 0
    inverted_index: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)

    def build(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus
        self.n_docs = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.n_docs if self.n_docs > 0 else 0.0

        self.inverted_index = {}
        for doc_idx, doc in enumerate(corpus):
            term_freqs: dict[str, int] = {}
            for term in doc:
                term_freqs[term] = term_freqs.get(term, 0) + 1
            for term, freq in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, freq))

        self.idf = {}
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(
        self, query_tokens: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by descending score."""
        scores: dict[int, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf = self.idf[token]
            for doc_idx, tf in self.inverted_index[token]:
                dl = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * numerator / denominator
                scores[doc_idx] = scores.get(doc_idx, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


_TOKENIZE_RE = re.compile(r"[_\-\s/.,;:]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace, underscores, hyphens, and punctuation."""
    return [t for t in _TOKENIZE_RE.split(text.lower()) if t]
