"""Synthetic help-desk queries grouped by intent.

Each intent has one canonical question (what gets cached first), several paraphrases,
and a canned answer. Intents are deliberately grouped into "families" that share
vocabulary (cancel/track/return *my order*), so near-miss queries with a different
intent look lexically similar. Those are the queries that cause false cache hits.
"""
from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple

INTENTS: Dict[str, Dict[str, List[str] | str]] = {
    # --- order family ---
    "order_cancel": {"canonical": "how do i cancel my order",
                     "paraphrases": ["can i cancel an order i just placed", "i want to cancel my order please",
                                     "cancel my recent order", "how can i cancel the order i made"],
                     "answer": "Open Orders, pick the order and press Cancel within 2 hours of purchase."},
    "order_track": {"canonical": "how do i track my order",
                    "paraphrases": ["where is my order right now", "can i track the order i placed",
                                    "track my recent order", "how can i see where my order is"],
                    "answer": "Use the tracking link in your confirmation email or the Orders page."},
    "order_return": {"canonical": "how do i return my order",
                     "paraphrases": ["can i return an order i received", "i want to return my order please",
                                     "return my recent order", "how can i send the order back"],
                     "answer": "Start a return from Orders within 30 days; a prepaid label is emailed to you."},
    # --- password / account family ---
    "password_reset": {"canonical": "how do i reset my password",
                       "paraphrases": ["i forgot my password how do i reset it", "reset my password please",
                                       "how can i reset a forgotten password", "password reset steps"],
                       "answer": "Click 'Forgot password' on the sign-in page; the link is valid for 30 minutes."},
    "password_change": {"canonical": "how do i change my password",
                        "paraphrases": ["i want to change my password", "change my password please",
                                        "how can i update my current password", "steps to change password"],
                        "answer": "Go to Settings > Security > Change password while signed in."},
    "account_delete": {"canonical": "how do i delete my account",
                       "paraphrases": ["i want to delete my account permanently", "delete my account please",
                                       "how can i remove my account", "steps to delete account"],
                       "answer": "Settings > Account > Delete account. Data is erased after 14 days."},
    "account_email": {"canonical": "how do i change the email on my account",
                      "paraphrases": ["i want to update my account email", "change my account email address",
                                      "how can i use a different email for my account", "update email on account"],
                      "answer": "Settings > Profile > Email, then confirm the new address from your inbox."},
    # --- billing family ---
    "billing_refund": {"canonical": "how do i get a refund",
                       "paraphrases": ["can i get my money back", "i want a refund please",
                                       "how can i request a refund", "refund request steps"],
                       "answer": "Refunds are available within 14 days from Billing > Request refund."},
    "billing_invoice": {"canonical": "how do i download my invoice",
                        "paraphrases": ["where can i get my invoice", "i need a copy of my invoice",
                                        "how can i download an invoice", "download invoice pdf"],
                        "answer": "Billing > Invoices lists every invoice as a downloadable PDF."},
    "billing_card": {"canonical": "how do i update my credit card",
                     "paraphrases": ["i want to change my payment card", "update my card details",
                                     "how can i change the credit card on file", "change payment method card"],
                     "answer": "Billing > Payment method > Replace card."},
    "plan_upgrade": {"canonical": "how do i upgrade my plan",
                     "paraphrases": ["i want to upgrade to a bigger plan", "upgrade my subscription plan",
                                     "how can i move to a higher plan", "steps to upgrade plan"],
                     "answer": "Billing > Plan > Upgrade; the price difference is prorated."},
    "plan_cancel": {"canonical": "how do i cancel my subscription",
                    "paraphrases": ["i want to cancel my subscription", "cancel my plan please",
                                    "how can i stop my subscription", "end my subscription"],
                    "answer": "Billing > Plan > Cancel subscription; access continues to the period end."},
    # --- shipping / misc family ---
    "shipping_time": {"canonical": "how long does shipping take",
                      "paraphrases": ["when will my package arrive", "what is the delivery time",
                                      "how many days for shipping", "how long does delivery take"],
                      "answer": "Standard shipping takes 3-5 business days; express takes 1-2."},
    "shipping_cost": {"canonical": "how much does shipping cost",
                      "paraphrases": ["what is the price of shipping", "is shipping free",
                                      "how much do you charge for delivery", "shipping fee amount"],
                      "answer": "Shipping is free over $50; otherwise it costs $4.99."},
    "store_hours": {"canonical": "what are your store opening hours",
                    "paraphrases": ["when is the store open", "what time does the shop open",
                                    "store hours today", "when do you open and close"],
                    "answer": "Stores open 9am-8pm Monday to Saturday and 10am-6pm on Sunday."},
    "contact_support": {"canonical": "how do i contact customer support",
                        "paraphrases": ["how can i talk to a human agent", "i need to reach support",
                                        "customer service phone number", "how do i get help from support"],
                        "answer": "Chat is available 24/7 from Help; phone support is 9am-5pm."},
}


def cache_prompts() -> List[Tuple[str, str]]:
    """(intent, canonical question) -- the first query of each intent fills the cache."""
    return [(k, v["canonical"]) for k, v in INTENTS.items()]


def all_queries() -> List[Tuple[str, str]]:
    return [(k, q) for k, v in INTENTS.items() for q in [v["canonical"], *v["paraphrases"]]]


def labeled_pairs(seed: int = 42) -> List[Tuple[str, str, int]]:
    """All paraphrase pairs (label 1) plus an equal number of non-paraphrase pairs (label 0).

    Half of the negatives are *hard* (same family, shared vocabulary), half random.
    """
    rng = random.Random(seed)
    pos, hard, easy = [], [], []
    by_intent = {k: [v["canonical"], *v["paraphrases"]] for k, v in INTENTS.items()}
    keys = list(by_intent)
    fam = {k: k.split("_")[0] for k in keys}
    for k, qs in by_intent.items():
        pos += [(a, b, 1) for a, b in itertools.combinations(qs, 2)]
    for a, b in itertools.combinations(keys, 2):
        pairs = [(x, y, 0) for x in by_intent[a] for y in by_intent[b]]
        (hard if fam[a] == fam[b] or a.endswith(b.split("_")[1]) else easy).extend(pairs)
    n_neg = len(pos)
    neg = rng.sample(hard, min(n_neg // 2, len(hard)))
    neg += rng.sample(easy, n_neg - len(neg))
    return pos + neg


def query_stream(n: int, seed: int = 42, zipf_a: float = 1.1) -> List[Tuple[str, str]]:
    """A realistic traffic stream: Zipf-distributed intents, random wording per request."""
    rng = random.Random(seed)
    keys = list(INTENTS)
    rng.shuffle(keys)
    w = [1.0 / (i + 1) ** zipf_a for i in range(len(keys))]
    out = []
    for _ in range(n):
        k = rng.choices(keys, weights=w)[0]
        v = INTENTS[k]
        out.append((k, rng.choice([v["canonical"], *v["paraphrases"]])))
    return out


# Probe set: cache the canonical question of ONE intent per sibling pair; every uncached
# intent has a lexically similar cached sibling, so its queries are hard "should miss" cases.
CACHED_INTENTS = ["order_cancel", "password_reset", "account_delete", "billing_refund", "plan_upgrade",
                  "shipping_time", "store_hours", "billing_card"]


def probe_set() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, bool]]]:
    """Returns (cache entries, probe queries). Probes are the paraphrases of all intents;
    ``should_hit`` is True iff the probe's intent is cached."""
    entries = [(k, INTENTS[k]["canonical"]) for k in CACHED_INTENTS]
    probes = [(k, q, k in CACHED_INTENTS) for k, v in INTENTS.items() for q in v["paraphrases"]]
    return entries, probes
