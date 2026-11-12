"""Algorithm 2 - Learning-Assisted Reasoning (Sec. 5, 6.2).

1. B' = U(B): the learning operator
     w'(t)   = σ(f_θ(t))                         learned reliability
     μ'(t)   = λ μ(t) + (1-λ) w'(t)              convex belief update
     B^(1)   = {t ∈ B : μ'(t) >= τ}              admission threshold
     T_pred  = {argmax_v f_θ(h, r, v) if >= γ}   graph-based prediction
   A prediction that matches a held candidate adds one supporting statement;
   it opens a new candidate only for keys left empty (completion). So U never
   creates a new conflict, which is the premise of Theorem 2.
2. Fragment-based reasoning on B': among the admissible fragments of a key,
   select the preferred one (highest μ'), and commit only if it beats the
   runner-up by δ; otherwise return every fragment (credulous answer).
"""
from __future__ import annotations

import numpy as np
import torch

from algorithms.paraconsistent import Conclusion, ParaconsistentKB, decide


class ModelScorer:
    """Wraps a trained KGModel: reliability w'(t) and tail prediction."""

    def __init__(self, model, ds):
        self.model, self.ds = model, ds
        self.emb = model.embed()
        self._cache = {}

    def prob(self, triples) -> np.ndarray:
        missing = [t for t in triples if t not in self._cache]
        if missing:
            ds = self.ds
            enc = torch.tensor([[ds.ent2id[h], ds.rel2id[r], ds.ent2id[t]] for h, r, t in missing])
            p = self.model.prob(enc, self.emb).numpy()
            self._cache.update(zip(missing, p.tolist()))
        return np.array([self._cache[t] for t in triples])

    def predict(self, keys, pool: dict) -> dict:
        """key -> (best tail among pool[r], probability). Memoised per key."""
        memo = self.__dict__.setdefault("_pred", {})
        todo = [k for k in keys if k not in memo]
        if todo:
            memo.update(self._predict(todo, pool))
        return {k: memo[k] for k in keys if k in memo}

    @torch.no_grad()
    def _predict(self, keys, pool: dict) -> dict:
        out = {}
        ds = self.ds
        for r in {k[1] for k in keys}:
            cands = pool.get(r, [])
            ks = [k for k in keys if k[1] == r]
            if not cands or not ks:
                continue
            emb = self.emb
            c = torch.tensor([ds.ent2id[t] for t in cands], device=emb.device)
            chunk = max(1, 2_000_000 // (len(c) * emb.shape[1]))
            for s in range(0, len(ks), chunk):
                part = ks[s:s + chunk]
                h = torch.tensor([ds.ent2id[k[0]] for k in part], device=emb.device)
                rr = torch.full_like(h, ds.rel2id[r])
                logits = self.model.decoder(emb[h].unsqueeze(1).expand(-1, len(c), -1), rr,
                                            emb[c].unsqueeze(0).expand(len(h), -1, -1))
                p, best = torch.sigmoid(logits).max(1)
                p, best = p.cpu(), best.cpu()
                for k, b, pb in zip(part, best.tolist(), p.tolist()):
                    out[k] = (cands[b], pb)
        return out


def learning_operator(kb: ParaconsistentKB, scorer, queries, lam: float, tau: float, gamma: float,
                      predictions: dict | None = None, prior=None):
    """Returns key -> {tail: (μ', support)} for the updated belief state B'."""
    prior = prior or {}
    triples = [(k[0], k[1], t) for k in queries for t in kb.candidates(k)]
    w = dict(zip(triples, scorer.prob(triples))) if triples else {}
    beliefs = {}
    for key in queries:
        cand = kb.candidates(key)
        b = {}
        for t, n in cand.items():
            mu = lam * prior.get((key[0], key[1], t), 1.0) + (1 - lam) * w[(key[0], key[1], t)]
            if mu >= tau:
                b[t] = [mu, n]
        if predictions and key in predictions:
            t_hat, p = predictions[key]
            if p >= gamma:
                if t_hat in b:
                    b[t_hat][1] += 1
                elif not b and kb.ontology.violation(key[0], key[1], t_hat) is None:
                    b[t_hat] = [(1 - lam) * p, 1]
        beliefs[key] = b
    return beliefs


def algorithm2_learning_reasoning(kb: ParaconsistentKB, scorer, queries, lam=0.5, tau=0.0, gamma=0.5,
                                  delta=0.0, pool=None, prior=None) -> dict:
    predictions = scorer.predict(list(queries), pool) if pool else None
    beliefs = learning_operator(kb, scorer, queries, lam, tau, gamma, predictions, prior)
    out = {}
    for key, b in beliefs.items():
        if not b:
            continue
        weights = {t: v[0] for t, v in b.items()}
        tails = decide(weights, delta, kb.ontology.is_functional(key[1]))
        out[key] = Conclusion(tails=tails, support={t: b[t][1] for t in tails},
                              weight={t: weights[t] for t in tails})
    return out
