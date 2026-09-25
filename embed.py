"""Two tiny, offline text embedders (no models, no network).

* ``HashedEmbedder``: feature hashing of word unigrams, word bigrams and character
  3-grams into a fixed-size vector, then L2 normalisation. No fitting needed.
* ``TfidfEmbedder``: classic TF-IDF over word unigrams + bigrams, fitted on a corpus.

Cosine similarity between two unit vectors is just their dot product.
"""
from __future__ import annotations

import re
import zlib
from typing import Dict, Iterable, List

import numpy as np

STOP = {"a", "an", "the", "i", "my", "me", "do", "to", "is", "please", "can", "how", "what", "on", "of", "for", "your",
        "it", "you", "are", "does", "in", "and"}


def tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def features(text: str) -> List[str]:
    w = tokens(text)
    feats = [f"w:{t}" for t in w] + [f"b:{a}_{b}" for a, b in zip(w, w[1:])]
    for t in w:
        s = f"#{t}#"
        feats += [f"c:{s[i:i + 3]}" for i in range(len(s) - 2)]
    return feats


def _h(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


class HashedEmbedder:
    name = "hashed"

    def __init__(self, dim: int = 1024, char_weight: float = 0.35):
        self.dim, self.char_weight = dim, char_weight

    def fit(self, corpus: Iterable[str]) -> "HashedEmbedder":
        return self

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for f in features(text):
            h = _h(f)
            sign = 1.0 if (h >> 31) & 1 else -1.0  # signed hashing reduces collision bias
            v[h % self.dim] += sign * (self.char_weight if f.startswith("c:") else 1.0)
        n = np.linalg.norm(v)
        return v / n if n else v


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf = np.zeros(0)

    @staticmethod
    def _terms(text: str) -> List[str]:
        w = tokens(text)
        return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]

    def fit(self, corpus: Iterable[str]) -> "TfidfEmbedder":
        docs = [set(self._terms(d)) for d in corpus]
        for d in docs:
            for t in sorted(d):
                self.vocab.setdefault(t, len(self.vocab))
        df = np.zeros(len(self.vocab))
        for d in docs:
            for t in d:
                df[self.vocab[t]] += 1
        self.idf = np.log((1 + len(docs)) / (1 + df)) + 1.0
        return self

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in self._terms(text):
            if t in self.vocab:
                v[self.vocab[t]] += 1.0
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n else v
