"""LLM-based reasoning baselines (reviewer R1-A, R2-6).

Every test query (h, r, {candidate tails}) is scored by an LLM in two regimes:
  * zero-shot prompting : the query only
  * KG-RAG              : the query plus facts about the k nearest locations
                          retrieved from the training graph
Each regime is evaluated
  (i)  LLM-only            - accept a candidate iff p >= 0.5
  (ii) LLM + paraconsistent - the LLM probabilities replace the learned
                              reliability w'(t) inside Algorithm 2 (δ = 0)
so the contribution of the symbolic layer on top of an LLM can be measured.
Prompts: prompts/conflict_*.txt. Token usage and latency are logged (R1-B).
"""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree

from algorithms.algorithm2_learning import algorithm2_learning_reasoning
from algorithms.paraconsistent import Conclusion
from config import LLM_DIR as RESULTS_DIR
from kg.llm import LLMError, load_prompt, safe_name
from kg.ontology import is_location, location_coords

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "query_id": {"type": "integer"},
                    "candidate": {"type": "string"},
                    "probability": {"type": "number"},
                },
                "required": ["query_id", "candidate", "probability"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}

# USD per million tokens (input, output); used only for the cost column
# DeepSeek: official off-peak rate (peak is 2x); third-party gateways price differently
PRICES = {"deepseek-flash": (0.15, 0.60),  # gateway models: cost not estimated
           "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


class Retriever:
    """k nearest training locations and their asserted facts."""

    def __init__(self, train_df, k=10):
        self.k = k
        self.facts = train_df.groupby("head")[["relation", "tail"]].apply(lambda g: list(map(tuple, g.values)))
        locs = [h for h in self.facts.index if is_location(h)]
        self.locs = locs
        self.tree = cKDTree(np.array([location_coords(h) for h in locs], dtype=float)) if locs else None

    def context(self, head):
        if self.tree is None or not is_location(head):
            facts = self.facts.get(head, [])
            return "; ".join(f"{r}={t}" for r, t in facts) or "(no facts)"
        dist, idx = self.tree.query(np.array(location_coords(head), dtype=float), k=self.k)
        lines = []
        for d, i in zip(np.atleast_1d(dist), np.atleast_1d(idx)):
            h = self.locs[int(i)]
            facts = ", ".join(f"{r}={t}" for r, t in self.facts[h])
            lines.append(f"{h} (distance {d:.0f}): {facts}")
        return " | ".join(lines)


class LLMScorer:
    """Plugs LLM probabilities into Algorithm 2 as the reliability w'(t)."""

    def __init__(self, probs: dict):
        self.probs = probs

    def prob(self, triples):
        return np.array([self.probs.get(t, 0.5) for t in triples])

    def predict(self, keys, pool):
        return {}


def llm_probabilities(backend, queries: dict, mode: str, retriever: Retriever | None,
                      value_description: str, batch: int = 20) -> tuple[dict, int]:
    system = load_prompt("conflict_system.txt").format(value_description=value_description)
    user_tpl = load_prompt("conflict_zero_shot_user.txt" if mode == "zero-shot" else "conflict_rag_user.txt")
    keys = list(queries)
    probs, failures = {}, 0
    for s in range(0, len(keys), batch):
        chunk = keys[s:s + batch]
        lines = []
        for qi, (h, r) in enumerate(chunk):
            cands = ", ".join(sorted(queries[(h, r)]))
            line = f"Q{qi}: location={h}; relation={r}; candidates=[{cands}]"
            if mode == "kg-rag":
                line += f"\n    retrieved: {retriever.context(h)}"
            lines.append(line)
        try:
            out = backend.complete_json(system, user_tpl.format(queries="\n".join(lines)), ANSWER_SCHEMA)
        except LLMError as exc:
            print(f"  batch {s // batch}: {exc}")
            failures += 1
            continue
        for a in out.get("answers", []):
            qi = a["query_id"]
            if 0 <= qi < len(chunk):
                h, r = chunk[qi]
                probs[(h, r, str(a["candidate"]).strip())] = float(np.clip(a["probability"], 0, 1))
    return probs, failures


def run_llm_baselines(task, backend, value_description: str, max_queries: int | None = None, k: int = 10):
    queries = dict(list(task.queries["test"].items())[:max_queries]) if max_queries else task.queries["test"]
    retriever = Retriever(task.ds.split("train"), k=k)
    results = {"backend": backend.describe(), "queries": len(queries)}
    for mode in ("zero-shot", "kg-rag"):
        before = (backend.usage.calls, backend.usage.input_tokens, backend.usage.output_tokens, backend.usage.latency_s)
        probs, failures = llm_probabilities(backend, queries, mode, retriever, value_description)
        only = {}
        for key, cand in queries.items():
            acc = [t for t in cand if probs.get((key[0], key[1], t), 0.0) >= 0.5]
            if acc:
                only[key] = Conclusion(acc, {t: cand[t] for t in acc}, {})
        task.queries["_llm"] = queries
        symbolic = algorithm2_learning_reasoning(task.kb, LLMScorer(probs), list(queries), lam=task.cfg.reason.lam,
                                                 tau=0.0, delta=0.0, prior=task.prior)
        u = backend.usage
        usage = {"calls": u.calls - before[0], "input_tokens": u.input_tokens - before[1],
                 "output_tokens": u.output_tokens - before[2], "latency_s": u.latency_s - before[3],
                 "failed_batches": failures}
        price = PRICES.get(backend.model)
        if price:
            usage["cost_usd"] = (usage["input_tokens"] * price[0] + usage["output_tokens"] * price[1]) / 1e6
        results[mode] = {
            "LLM only": task.score(only, "_llm"),
            "LLM + paraconsistent (Alg. 2)": task.score(symbolic, "_llm"),
            "usage": usage,
        }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"llm_baselines_{backend.name}_{safe_name(backend.model)}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return results
