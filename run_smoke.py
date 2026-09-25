#!/usr/bin/env python3
"""Semantic cache smoke: pair similarity -> threshold sweep -> LRU/TTL eviction -> savings -> results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from data import CACHED_INTENTS, INTENTS, all_queries, labeled_pairs, probe_set, query_stream
from embed import HashedEmbedder, TfidfEmbedder
from simulate import LLM_COST_USD, LLM_LATENCY_MS, auc, exact_replay, pair_sims, probe_eval, replay
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {
    "n_intents": len(INTENTS), "stream_len": 2000, "zipf_a": 1.1, "hashed_dim": 1024, "char_weight": 0.35,
    "thresholds": [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)], "max_false_hit_rate": 0.05, "cached_intents": CACHED_INTENTS,
    "capacities": [2, 4, 8, 16, 32, None], "ttls": [25, 50, 100, 200, 400, None],
    "update_step": 1000, "updated_intents": ["billing_refund", "shipping_cost", "store_hours", "order_return"],
    "llm_latency_ms_simulated": LLM_LATENCY_MS, "llm_cost_usd_simulated": LLM_COST_USD,
}


def _compact(js: str) -> str:
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", js)


def _rows(js: str) -> str:
    """Put flat dicts (one sweep row each) on one line."""
    return re.sub(r"\{\n([^{}\[\]]*?)\n\s*\}", lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "}", js)


def _r(d):
    if isinstance(d, dict):
        return {k: _r(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_r(v) for v in d]
    return round(d, 4) if isinstance(d, float) else d


def main() -> None:
    t0 = time.perf_counter()
    np.random.seed(SEED)
    corpus = [q for _, q in all_queries()]
    embedders = {"hashed": HashedEmbedder(CFG["hashed_dim"], CFG["char_weight"]).fit(corpus),
                 "tfidf": TfidfEmbedder().fit(corpus)}
    pairs = labeled_pairs(SEED)
    stream = query_stream(CFG["stream_len"], SEED, CFG["zipf_a"])

    m = {"project": "ai-learn-20-semantic-cache", "seed": SEED, "config": CFG,
         "n_pairs": {"paraphrase": sum(p[2] for p in pairs), "non_paraphrase": sum(1 - p[2] for p in pairs)},
         "pair_similarity": {}, "probe_sweep": {}, "sweep": {}, "chosen": {}, "stream_at_chosen": {},
         "stream_exact_match_hit_rate": exact_replay(stream)}
    entries, probes = probe_set()
    sims = {}
    for name, emb in embedders.items():
        s, y = pair_sims(pairs, emb)
        sims[name] = (s, y)
        m["pair_similarity"][name] = {
            "auc": auc(s, y), "mean_sim_paraphrase": float(s[y == 1].mean()), "mean_sim_non_paraphrase": float(s[y == 0].mean()),
            "pair_recall_by_threshold": [float((s[y == 1] >= t).mean()) for t in CFG["thresholds"]],
            "pair_false_accept_by_threshold": [float((s[y == 0] >= t).mean()) for t in CFG["thresholds"]],
        }
        prow = [probe_eval(entries, probes, emb, t) for t in CFG["thresholds"]]
        m["probe_sweep"][name] = prow
        ok = [r for r in prow if r["false_hit_rate"] <= CFG["max_false_hit_rate"]]
        best = max(ok, key=lambda r: (r["hit_rate"], -r["threshold"])) if ok else min(prow, key=lambda r: r["false_hit_rate"])
        m["chosen"][name] = best
        rows = [replay(stream, emb, t) for t in CFG["thresholds"]]
        m["sweep"][name] = rows
        m["stream_at_chosen"][name] = rows[CFG["thresholds"].index(best["threshold"])]

    # Eviction study on the better embedder at its chosen threshold
    name = max(m["chosen"], key=lambda k: (m["chosen"][k]["hit_rate"], -m["chosen"][k]["false_hit_rate"]))
    emb, thr = embedders[name], m["chosen"][name]["threshold"]
    upd = {CFG["update_step"]: CFG["updated_intents"]}
    m["eviction"] = {"embedder": name, "threshold": thr,
                     "lru_capacity": [replay(stream, emb, thr, capacity=c) for c in CFG["capacities"]],
                     "ttl_with_answer_update": [replay(stream, emb, thr, ttl=t, updates=upd) for t in CFG["ttls"]]}
    st = m["stream_at_chosen"][name]
    m["headline"] = {
        "embedder": name, "threshold": thr, "probe_hit_rate": m["chosen"][name]["hit_rate"],
        "probe_false_hit_rate": m["chosen"][name]["false_hit_rate"], "stream_hit_rate": st["hit_rate"],
        "stream_exact_match_hit_rate": m["stream_exact_match_hit_rate"], "stream_false_hit_rate": st["false_hit_rate"],
        "cost_saving_pct": st["cost_saving_pct"], "latency_saving_pct": st["latency_saving_pct"],
        "pair_auc_hashed": m["pair_similarity"]["hashed"]["auc"], "pair_auc_tfidf": m["pair_similarity"]["tfidf"]["auc"],
    }
    def slim(rows, keep):
        return [{k: r[k] for k in keep} for r in rows]

    m["probe_sweep"] = {k: slim(v, ("threshold", "hit_rate", "false_hit_rate")) for k, v in m["probe_sweep"].items()}
    m["sweep"] = {k: slim(v, ("threshold", "hit_rate", "false_hit_rate", "llm_calls", "cost_saving_pct")) for k, v in m["sweep"].items()}
    ev_keep = ("hit_rate", "false_hit_rate", "stale_rate", "wrong_answer_rate", "llm_calls", "evicted", "expired")
    m["eviction"]["lru_capacity"] = slim(m["eviction"]["lru_capacity"], ev_keep)
    m["eviction"]["ttl_with_answer_update"] = slim(m["eviction"]["ttl_with_answer_update"], ev_keep)
    m["wall_time_s"] = time.perf_counter() - t0
    m = _r(m)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(_rows(_compact(json.dumps(m, indent=1))) + "\n")
    shot = {"project": m["project"], "seed": SEED, "config": CFG, "headline": m["headline"],
            "chosen": {k: {kk: v[kk] for kk in ("threshold", "hit_rate", "false_hit_rate", "false_hit_rate_uncached")}
                       for k, v in m["chosen"].items()}}
    (RESULTS / "JSON.shot").write_text(_compact(json.dumps(shot, indent=2)) + "\n")
    plots = make_plots(RESULTS, m, sims)
    write_results_md(RESULTS / "RESULTS.md", m, plots)
    json.loads((RESULTS / "JSON.shot").read_text())
    print(json.dumps(m["headline"], indent=2))
    print(f"wrote {len(plots)} plots; wall {m['wall_time_s']:.2f}s")


if __name__ == "__main__":
    main()
