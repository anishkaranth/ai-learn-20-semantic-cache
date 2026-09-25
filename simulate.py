"""Replay a query stream through a SemanticCache and account for correctness, latency and cost.

The "LLM" is an oracle stub: on a cache miss it returns the true answer for the query's
intent (current version) after a *simulated* latency/cost. On a hit we return whatever
was cached. A hit is a *false hit* if the cached entry belongs to a different intent,
and *stale* if it's the right intent but an outdated answer version.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from cache import SemanticCache
from data import INTENTS

LLM_LATENCY_MS = 900.0  # simulated per-call generation latency
LLM_COST_USD = 0.002  # simulated per-call cost


def replay(stream: List[Tuple[str, str]], embedder, threshold: float, capacity: Optional[int] = None,
           ttl: Optional[float] = None, updates: Optional[Dict[int, List[str]]] = None) -> Dict[str, float]:
    """``updates`` maps time step -> intents whose answer changes (version += 1) at that step."""
    cache = SemanticCache(embedder, threshold, capacity, ttl)
    version = {k: 0 for k in INTENTS}
    hits = false_hits = stale = llm_calls = 0
    lookup_ms = []
    for t, (intent, q) in enumerate(stream):
        for k in (updates or {}).get(t, []):
            version[k] += 1
        t0 = time.perf_counter()
        entry, _ = cache.lookup(q, now=float(t))
        lookup_ms.append((time.perf_counter() - t0) * 1000)
        if entry is not None:
            hits += 1
            if entry.meta["intent"] != intent:
                false_hits += 1
            elif entry.meta["version"] != version[intent]:
                stale += 1
        else:
            llm_calls += 1
            cache.put(q, INTENTS[intent]["answer"], now=float(t), intent=intent, version=version[intent])
    n = len(stream)
    lk = float(np.mean(lookup_ms))
    base_latency = LLM_LATENCY_MS
    cached_latency = lk + (llm_calls / n) * LLM_LATENCY_MS
    return {
        "threshold": threshold, "n": n, "hit_rate": hits / n, "false_hit_rate": false_hits / n,
        "hit_precision": (hits - false_hits) / hits if hits else 1.0, "stale_rate": stale / n,
        "wrong_answer_rate": (false_hits + stale) / n, "llm_calls": llm_calls, "final_cache_size": len(cache),
        "evicted": cache.stats["evicted"], "expired": cache.stats["expired"],
        "mean_lookup_ms_measured": lk, "mean_latency_ms_no_cache": base_latency,
        "mean_latency_ms_with_cache": cached_latency, "latency_saving_pct": 100 * (1 - cached_latency / base_latency),
        "cost_usd_no_cache": n * LLM_COST_USD, "cost_usd_with_cache": llm_calls * LLM_COST_USD,
        "cost_saving_pct": 100 * (1 - llm_calls / n),
    }


def probe_eval(entries, probes, embedder, threshold: float) -> Dict[str, float]:
    """Static cache of canonical questions; count correct hits, false hits and misses."""
    cache = SemanticCache(embedder, threshold)
    for k, q in entries:
        cache.put(q, INTENTS[k]["answer"], intent=k, version=0)
    tp = fh_pos = fh_neg = miss_pos = 0
    n_pos = sum(1 for p in probes if p[2])
    n_neg = len(probes) - n_pos
    for intent, q, should_hit in probes:
        e, _ = cache.lookup(q)
        if e is None:
            miss_pos += should_hit
        elif e.meta["intent"] == intent:
            tp += 1
        elif should_hit:
            fh_pos += 1
        else:
            fh_neg += 1
    return {"threshold": threshold, "n_should_hit": n_pos, "n_should_miss": n_neg,
            "hit_rate": tp / n_pos,  # paraphrases of cached intents answered correctly
            "false_hit_rate": (fh_neg + fh_pos) / len(probes),  # any wrong answer served from cache
            "false_hit_rate_uncached": fh_neg / n_neg,  # hard negatives wrongly served
            "wrong_answer_rate_of_hits": (fh_neg + fh_pos) / max(1, tp + fh_neg + fh_pos)}


def exact_replay(stream) -> float:
    """Baseline: an exact-string cache (dict). Returns hit rate."""
    seen, hits = set(), 0
    for _, q in stream:
        hits += q in seen
        seen.add(q)
    return hits / len(stream)


def pair_sims(pairs, embedder) -> Tuple[np.ndarray, np.ndarray]:
    s = np.array([float(embedder.embed(a) @ embedder.embed(b)) for a, b, _ in pairs])
    y = np.array([lab for _, _, lab in pairs])
    return s, y


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Probability a random positive outranks a random negative (ties count half)."""
    pos, neg = scores[y == 1], scores[y == 0]
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)
