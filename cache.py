"""A semantic cache: look up answers by *embedding similarity* instead of exact string match.

get(query) embeds the query, finds the most similar cached key (cosine = dot product of
unit vectors) and returns its answer if the similarity is >= ``threshold``. Entries
expire after ``ttl`` time units and the least-recently-used entry is evicted when the
cache is over ``capacity``.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class Entry:
    key: str
    vec: np.ndarray
    value: Any
    created: float
    meta: Dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    def __init__(self, embedder, threshold: float = 0.8, capacity: Optional[int] = None, ttl: Optional[float] = None):
        self.embedder, self.threshold, self.capacity, self.ttl = embedder, threshold, capacity, ttl
        self.store: "OrderedDict[int, Entry]" = OrderedDict()  # insertion/recency order = LRU order
        self._next = 0
        self.stats = {"lookups": 0, "hits": 0, "misses": 0, "expired": 0, "evicted": 0}

    def _expire(self, now: float) -> None:
        if self.ttl is None:
            return
        dead = [i for i, e in self.store.items() if now - e.created > self.ttl]
        for i in dead:
            del self.store[i]
        self.stats["expired"] += len(dead)

    def lookup(self, query: str, now: float = 0.0) -> Tuple[Optional[Entry], float]:
        """Return (best entry if above threshold else None, best similarity)."""
        self.stats["lookups"] += 1
        self._expire(now)
        if not self.store:
            self.stats["misses"] += 1
            return None, 0.0
        q = self.embedder.embed(query)
        ids = list(self.store)
        M = np.stack([self.store[i].vec for i in ids])
        sims = M @ q
        j = int(np.argmax(sims))
        best = float(sims[j])
        if best >= self.threshold:
            self.store.move_to_end(ids[j])  # mark as recently used
            self.stats["hits"] += 1
            return self.store[ids[j]], best
        self.stats["misses"] += 1
        return None, best

    def put(self, query: str, value: Any, now: float = 0.0, **meta) -> None:
        self.store[self._next] = Entry(query, self.embedder.embed(query), value, now, meta)
        self._next += 1
        if self.capacity is not None:
            while len(self.store) > self.capacity:
                self.store.popitem(last=False)  # evict least recently used
                self.stats["evicted"] += 1

    def __len__(self) -> int:
        return len(self.store)
