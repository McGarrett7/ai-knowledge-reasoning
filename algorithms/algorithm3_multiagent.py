"""Algorithm 3 - Learning-Based Multi-Agent Reasoning (Sec. 5, 6.3).

Agents a_1..a_m observe partially overlapping subgraphs: locations are split
into m spatial regions (k-means on coordinates) and every triple is also
shared with each other agent with probability `overlap`. Each agent
  1. trains its own scorer on its local training triples (local learning
     operator U_i) and
  2. scores the candidates of the aggregated knowledge base K = ∪_i B_i
     (communication operator C_ij: beliefs about shared candidates are exchanged).
The aggregated belief is the reliability-weighted mean
     μ̄(t) = Σ_i π_i μ'_i(t) / Σ_i π_i ,   π_i = validation accuracy of agent i,
and |S(t)| counts the agents that hold t plus the agents whose prediction is t.
Reasoning over K uses the same fragment selection as Algorithm 2.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from algorithms.paraconsistent import Conclusion, ParaconsistentKB, decide
from kg.ontology import is_location, location_coords


@dataclass
class Agent:
    idx: int
    rows: pd.Index            # dataframe rows this agent observes
    scorer: object = None
    reliability: float = 1.0


def _kmeans(xy: np.ndarray, m: int, seed: int, iters: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = xy[rng.choice(len(xy), size=m, replace=False)]
    for _ in range(iters):
        lab = ((xy[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.array([xy[lab == j].mean(0) if (lab == j).any() else centers[j] for j in range(m)])
        if np.allclose(new, centers):
            break
        centers = new
    return lab


def partition_agents(df: pd.DataFrame, m: int, overlap: float, seed: int) -> list[pd.Index]:
    """Rows observed by each agent: a spatial home region plus random overlap."""
    heads = df["head"].unique()
    if all(is_location(h) for h in heads[:100]):
        xy = np.array([location_coords(h) for h in heads], dtype=float)
        lab = _kmeans(xy, m, seed)
    else:
        lab = np.random.default_rng(seed).integers(0, m, size=len(heads))
    home = dict(zip(heads, lab))
    rng = np.random.default_rng(seed + 1)
    primary = df["head"].map(home).values
    extra = rng.random((len(df), m)) < overlap
    return [df.index[(primary == i) | extra[:, i]] for i in range(m)]


def algorithm3_multiagent_reasoning(agents: list[Agent], df: pd.DataFrame, ontology, queries,
                                    lam=0.5, tau=0.0, gamma=0.5, delta=0.0, pool=None, prior=None) -> dict:
    prior = prior or {}
    local_kbs = [ParaconsistentKB(df.loc[a.rows, ["head", "relation", "tail"]].itertuples(index=False, name=None),
                                  ontology) for a in agents]
    # K = ∪ B_i : a candidate's evidence = number of agents asserting it
    holders = defaultdict(lambda: defaultdict(int))
    for kb in local_kbs:
        for key in queries:
            for t in kb.candidates(key):
                holders[key][t] += 1
    triples = [(k[0], k[1], t) for k in queries for t in holders.get(k, {})]
    pi = np.array([a.reliability for a in agents], dtype=float)
    pi = pi / pi.sum()
    agg = defaultdict(float)
    for a, w in zip(agents, pi):
        if not triples:
            break
        p = a.scorer.prob(triples)
        for tr, pt in zip(triples, p):
            agg[tr] += w * (lam * prior.get(tr, 1.0) + (1 - lam) * pt)
    votes = defaultdict(int)
    fallback = {}
    if pool:
        for a in agents:
            for key, (t_hat, p) in a.scorer.predict(list(queries), pool).items():
                if p >= gamma:
                    votes[(key, t_hat)] += 1
                    if key not in fallback or p > fallback[key][1]:
                        fallback[key] = (t_hat, p)
    out = {}
    for key in queries:
        weights, support = {}, {}
        for t, n in holders.get(key, {}).items():
            mu = agg[(key[0], key[1], t)]
            if mu >= tau:
                weights[t] = mu
                support[t] = n + votes.get((key, t), 0)
        if not weights and key in fallback and ontology.violation(key[0], key[1], fallback[key][0]) is None:
            t_hat, p = fallback[key]
            weights[t_hat], support[t_hat] = (1 - lam) * p, votes[(key, t_hat)]
        if not weights:
            continue
        tails = decide(weights, delta, ontology.is_functional(key[1]))
        out[key] = Conclusion(tails=tails, support={t: support[t] for t in tails},
                              weight={t: weights[t] for t in tails})
    return out
