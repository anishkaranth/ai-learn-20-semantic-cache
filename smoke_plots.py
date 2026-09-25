"""Matplotlib SVG plots + RESULTS.md writer for the semantic cache smoke run."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "ai-learn-20", "svg.fonttype": "none", "font.family": "sans-serif",
                     "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False})
COL = {"hashed": "#126782", "tfidf": "#e76f51"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any], sims) -> List[str]:
    names = []
    th = m["config"]["thresholds"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, rows in m["probe_sweep"].items():
        ax.plot(th, [r["hit_rate"] for r in rows], "-o", ms=3, color=COL[k], label=f"{k}: hit rate (cached paraphrases)")
        ax.plot(th, [r["false_hit_rate"] for r in rows], "--x", ms=4, color=COL[k], label=f"{k}: false-hit rate (wrong answer)")
        c = m["chosen"][k]["threshold"]
        ax.axvline(c, color=COL[k], lw=0.8, alpha=0.5)
    ax.set_xlabel("similarity threshold")
    ax.set_ylabel("fraction of queries")
    ax.set_title("Probe set: hit rate vs false hits (vertical = chosen)")
    ax.legend(fontsize=7)
    names.append(_save(fig, out / "threshold_sweep.svg"))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, (k, (s, y)) in zip(axes, sims.items()):
        bins = np.linspace(0, 1, 26)
        ax.hist(s[y == 1], bins=bins, alpha=0.6, color="#2a9d8f", label="paraphrase pairs")
        ax.hist(s[y == 0], bins=bins, alpha=0.6, color="#e76f51", label="non-paraphrase pairs")
        ax.set_title(f"{k} (AUC {m['pair_similarity'][k]['auc']:.3f})")
        ax.set_xlabel("cosine similarity")
    axes[0].set_ylabel("pairs")
    axes[0].legend(fontsize=7)
    names.append(_save(fig, out / "similarity_hist.svg"))

    ev = m["eviction"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    caps = [str(c) if c is not None else "inf" for c in m["config"]["capacities"]]
    axes[0].plot(caps, [r["hit_rate"] for r in ev["lru_capacity"]], "-o", color="#126782", label="hit rate")
    axes[0].plot(caps, [r["false_hit_rate"] for r in ev["lru_capacity"]], "--x", color="#e76f51", label="false-hit rate")
    axes[0].set_xlabel("LRU capacity (entries)")
    axes[0].set_title("LRU eviction")
    axes[0].legend(fontsize=7)
    ttls = [str(t) if t is not None else "inf" for t in m["config"]["ttls"]]
    axes[1].plot(ttls, [r["hit_rate"] for r in ev["ttl_with_answer_update"]], "-o", color="#126782", label="hit rate")
    axes[1].plot(ttls, [r["stale_rate"] for r in ev["ttl_with_answer_update"]], "--s", color="#6a4c93", label="stale-answer rate")
    axes[1].set_xlabel("TTL (requests)")
    axes[1].set_title("TTL with an answer update at t=1000")
    axes[1].legend(fontsize=7)
    names.append(_save(fig, out / "eviction.svg"))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for k, rows in m["sweep"].items():
        ax.plot(th, [r["cost_saving_pct"] for r in rows], "-o", ms=3, color=COL[k], label=f"{k}: cost saving %")
        ax.plot(th, [100 * r["false_hit_rate"] for r in rows], ":", color=COL[k], label=f"{k}: false-hit %")
    ax.axhline(100 * m["stream_exact_match_hit_rate"], color="grey", lw=1, ls="--", label="exact-match cache: cost saving %")
    ax.set_xlabel("similarity threshold")
    ax.set_ylabel("% of 2000 stream queries")
    ax.set_title("Stream replay: simulated LLM cost saving vs false hits")
    ax.legend(fontsize=7)
    names.append(_save(fig, out / "savings.svg"))
    return names


def _f(x):
    return "inf" if x is None else str(x)


def write_results_md(path: Path, m: Dict[str, Any], plots: List[str]) -> None:
    c, h = m["config"], m["headline"]
    L = [f"# Results -- {m['project']}", "",
         f"**Seed:** `{m['seed']}` | {c['n_intents']} intents | stream = {c['stream_len']} Zipf(a={c['zipf_a']}) queries | "
         f"pairs: {m['n_pairs']['paraphrase']} paraphrase / {m['n_pairs']['non_paraphrase']} non-paraphrase | "
         f"simulated LLM call = {c['llm_latency_ms_simulated']:.0f} ms, ${c['llm_cost_usd_simulated']}", "",
         "## Pair-level separability", "", "| Embedder | AUC | mean sim (paraphrase) | mean sim (non-paraphrase) |", "|---|---:|---:|---:|"]
    for k, p in m["pair_similarity"].items():
        L.append(f"| {k} | {p['auc']:.4f} | {p['mean_sim_paraphrase']:.4f} | {p['mean_sim_non_paraphrase']:.4f} |")
    L += ["", f"## Probe set: hit rate vs wrong answers", "",
          f"Cache holds the canonical question of {len(c['cached_intents'])} intents; probes are the 64 paraphrases of all 16 intents "
          "(32 should hit, 32 should miss; every uncached intent has a lexically similar cached sibling).", "",
          f"Chosen threshold = max hit rate with false-hit rate <= {c['max_false_hit_rate']}:", "",
          "| Embedder | threshold | hit rate | false-hit rate (all probes) | false-hit rate (uncached probes) | wrong answers among hits |",
          "|---|---:|---:|---:|---:|---:|"]
    for k, r in m["chosen"].items():
        L.append(f"| {k} | {r['threshold']:.2f} | {r['hit_rate']:.4f} | {r['false_hit_rate']:.4f} | {r['false_hit_rate_uncached']:.4f} | {r['wrong_answer_rate_of_hits']:.4f} |")
    L += ["", "| threshold | " + " | ".join(f"{k} hit / false-hit" for k in m["probe_sweep"]) + " |", "|---:|" + "---:|" * len(m["probe_sweep"])]
    for i, t in enumerate(c["thresholds"]):
        L.append(f"| {t:.2f} | " + " | ".join(f"{m['probe_sweep'][k][i]['hit_rate']:.3f} / {m['probe_sweep'][k][i]['false_hit_rate']:.3f}" for k in m["probe_sweep"]) + " |")
    L += ["", f"## Stream replay at the chosen threshold ({c['stream_len']} Zipf queries, cache filled on miss)", "",
          f"Exact-string cache baseline hit rate: **{m['stream_exact_match_hit_rate']:.4f}**.", "",
          "| Embedder | threshold | hit rate | false-hit rate | LLM calls | cost saving | mean latency (ms) | lookup (ms, measured) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in m["stream_at_chosen"].items():
        L.append(f"| {k} | {r['threshold']:.2f} | {r['hit_rate']:.4f} | {r['false_hit_rate']:.4f} | {r['llm_calls']} | {r['cost_saving_pct']:.1f}% | "
                 f"{r['mean_latency_ms_with_cache']:.1f} vs {r['mean_latency_ms_no_cache']:.0f} | {r['mean_lookup_ms_measured']:.3f} |")
    ev = m["eviction"]
    L += ["", f"## Eviction ({ev['embedder']} @ threshold {ev['threshold']:.2f})", "",
          "| LRU capacity | hit rate | false-hit rate | evicted |", "|---:|---:|---:|---:|"]
    for cap, r in zip(c["capacities"], ev["lru_capacity"]):
        L.append(f"| {_f(cap)} | {r['hit_rate']:.4f} | {r['false_hit_rate']:.4f} | {r['evicted']} |")
    L += ["", f"Answers for {', '.join(c['updated_intents'])} change at t={c['update_step']}; a hit that returns the old version is *stale*.", "",
          "| TTL | hit rate | stale rate | wrong-answer rate (false + stale) | expired |", "|---:|---:|---:|---:|---:|"]
    for t, r in zip(c["ttls"], ev["ttl_with_answer_update"]):
        L.append(f"| {_f(t)} | {r['hit_rate']:.4f} | {r['stale_rate']:.4f} | {r['wrong_answer_rate']:.4f} | {r['expired']} |")
    L += ["", "## Takeaways", "",
          f"- Best setting: **{h['embedder']} @ {h['threshold']:.2f}** answers {h['probe_hit_rate']:.1%} of cached-intent paraphrases with a "
          f"probe false-hit rate of {h['probe_false_hit_rate']:.1%}. On the stream it hits {h['stream_hit_rate']:.1%} of queries "
          f"(exact match: {h['stream_exact_match_hit_rate']:.1%}), cutting simulated LLM cost by {h['cost_saving_pct']:.1f}%.",
          "- Lower thresholds raise the hit rate but start returning answers for the *wrong* intent (cancel vs track vs return an order). "
          "The false-hit rate is the metric to cap, not the hit rate to maximise.",
          "- The stream only has 80 distinct strings, so an exact-match cache already hits most repeats; the semantic cache's extra value "
          "is the paraphrase hits it adds on top, paid for with a small false-hit rate.",
          "- Latency and cost numbers are simulated (fixed per-call LLM latency/cost); cache lookup time is measured.", "",
          "## Plots", ""] + [f"![{p}]({p})" for p in plots] + ["", f"Wall time: {m['wall_time_s']:.2f}s on CPU."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
