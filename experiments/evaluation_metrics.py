"""Evaluation metrics (Sec. 7.6, Table 4), computed on the query keys of a split.

For a set of conclusions {key -> Conclusion}:
  Correctness          |R_valid| / |R_total|   R = answered keys; a key is valid
                       when its accepted tails are jointly consistent with O
  Support ratio        Σ |S_i| / n             over the n accepted triples
  Conflict mitigation  1 - |C_prop| / |C_total|  C_total = queried keys with
                       competing assertions; C_prop = those whose answer still
                       contains more than one tail
  Reasoning accuracy   |R_correct| / |R_predicted|   precision of accepted triples
Additionally reported: recall (true triples recovered / true triples asserted),
F1, and conflict-resolution accuracy (conflict keys answered with exactly the
true tail).
"""
from __future__ import annotations

import numpy as np


def evaluate(conclusions: dict, truth: dict, asserted: dict, ontology) -> dict:
    """
    conclusions : key -> Conclusion
    truth       : key -> set of true tails
    asserted    : key -> {tail: count} candidates asserted for the queried keys
    """
    # A query asks which of its asserted candidate statements are entailed, so
    # conclusions are restricted to those candidates (the KB may hold other
    # tails of the same key, e.g. training facts of a non-functional relation).
    tails_of = {k: [t for t in c.tails if t in asserted.get(k, ())] for k, c in conclusions.items()}
    answered = [k for k, ts in tails_of.items() if ts]
    valid = sum(1 for k in answered if not (ontology.is_functional(k[1]) and len(tails_of[k]) > 1))
    predicted = [(k, t) for k in answered for t in tails_of[k]]
    correct = sum(1 for k, t in predicted if t in truth.get(k, ()))
    recoverable = sum(len(set(asserted[k]) & truth.get(k, set())) for k in asserted)
    supports = [conclusions[k].support.get(t, 0) for k, t in predicted]
    conflict_keys = [k for k, c in asserted.items() if ontology.is_functional(k[1]) and len(c) > 1]
    propagated = sum(1 for k in conflict_keys if len(tails_of.get(k, ())) > 1)
    resolved = sum(1 for k in conflict_keys
                   if len(tails_of.get(k, ())) == 1 and tails_of[k][0] in truth.get(k, ()))
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / recoverable if recoverable else 0.0
    return {
        "queries": len(asserted),
        "answered": len(answered),
        "correctness": valid / len(answered) if answered else 0.0,
        "accuracy": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "support_ratio": float(np.mean(supports)) if supports else 0.0,
        "conflicts": len(conflict_keys),
        "conflict_mitigation": 1 - propagated / len(conflict_keys) if conflict_keys else 1.0,
        "conflict_resolution_acc": resolved / len(conflict_keys) if conflict_keys else 1.0,
    }


def aggregate(runs: list[dict]) -> dict:
    """mean and std over seeds for every numeric metric."""
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
            for k in keys}
