"""Shared experiment pipeline: one call evaluates every method on one seed."""
from __future__ import annotations

import itertools
import time
from collections import Counter

import numpy as np
import torch
from scipy.spatial import cKDTree

from algorithms.algorithm1_paraconsistent import algorithm1_paraconsistent_reasoning
from algorithms.algorithm2_learning import ModelScorer, algorithm2_learning_reasoning
from algorithms.algorithm3_multiagent import Agent, algorithm3_multiagent_reasoning, partition_agents
from algorithms.paraconsistent import Conclusion, ParaconsistentKB
from config import ExperimentConfig
from experiments.evaluation_metrics import evaluate
from kg.data import KGDataset
from kg.ontology import is_location, is_number, location_coords
from models.kge import KGModel, train_model

# Table 10 baselines: (label, encoder, decoder)
NEURAL_BASELINES = [
    ("TransE", "embed", "transe"),
    ("DistMult", "embed", "distmult"),
    ("ComplEx", "embed", "complex"),
    ("RotatE", "embed", "rotate"),
    ("R-GCN", "rgcn", "distmult"),
    ("CompGCN", "compgcn", "distmult"),
]


class Task:
    """Everything the algorithms need for one dataset/seed."""

    def __init__(self, ds: KGDataset, ontology, cfg: ExperimentConfig, seed: int):
        self.ds, self.ontology, self.cfg, self.seed = ds, ontology, cfg, seed
        triples = ds.df[["head", "relation", "tail"]].itertuples(index=False, name=None)
        self.kb = ParaconsistentKB(triples, ontology)
        self.queries = {s: ds.candidates(s) for s in ("val", "test")}
        self.pool = {r: list(g["tail"].unique()) for r, g in ds.split("train").groupby("relation")}
        self.prior = None
        if "weight" in ds.df.columns:
            self.prior = {(h, r, t): float(w) for h, r, t, w in
                          ds.df[["head", "relation", "tail", "weight"]].itertuples(index=False)}

    def score(self, conclusions, split="test"):
        return evaluate(conclusions, self.ds.truth, self.queries[split], self.ontology)

    # ----------------------------------------------------------- training
    def train(self, encoder, decoder, train_df=None, graph_df=None, verbose=False):
        cfg = self.cfg.train
        train_df = self.ds.split("train") if train_df is None else train_df
        graph = None
        if encoder not in ("embed", "id"):
            graph = self.ds.build_graph(train_df if graph_df is None else graph_df)
        torch.manual_seed(self.seed)
        model = KGModel(encoder, decoder, self.ds, graph, cfg)
        info = train_model(model, self.ds, train_df, cfg, self.seed, verbose=verbose)
        return ModelScorer(model, self.ds), info

    # ----------------------------------------------------------- tuning
    def tuned(self, run, grid: dict) -> tuple[dict, dict]:
        """Pick hyper-parameters on validation F1 (ties -> correctness), report test."""
        best, best_params = None, None
        names = list(grid)
        for values in itertools.product(*(grid[n] for n in names)):
            params = dict(zip(names, values))
            m = self.score(run(split="val", **params), "val")
            key = (round(m["f1"], 6), round(m["correctness"], 6))
            if best is None or key > best:
                best, best_params = key, params
        return self.score(run(split="test", **best_params), "test"), best_params

    # ----------------------------------------------------------- methods
    def alg1(self, kb=None):
        kb = kb or self.kb
        return self.score(algorithm1_paraconsistent_reasoning(kb, list(self.queries["test"])))

    def alg2(self, scorer):
        rc = self.cfg.reason

        def run(split, tau, delta):
            return algorithm2_learning_reasoning(self.kb, scorer, list(self.queries[split]), lam=rc.lam,
                                                 tau=tau, gamma=rc.gamma, delta=delta, pool=self.pool,
                                                 prior=self.prior)
        return self.tuned(run, {"tau": rc.tau_grid, "delta": rc.delta_grid})

    def make_agents(self, m, encoder="rgcn", decoder="bilinear", overlap=None):
        rc = self.cfg.reason
        parts = partition_agents(self.ds.df, m, rc.overlap if overlap is None else overlap, self.seed)
        agents, infos = [], []
        for i, rows in enumerate(parts):
            local = self.ds.df.loc[rows]
            local_train = local[local["split"] == "train"]
            scorer, info = self.train(encoder, decoder, train_df=local_train)
            agents.append(Agent(i, rows, scorer, info["best_val_acc"]))
            infos.append(info)
        return agents, infos

    def alg3(self, agents, equal_weights=False):
        rc = self.cfg.reason
        if equal_weights:
            agents = [Agent(a.idx, a.rows, a.scorer, 1.0) for a in agents]

        def run(split, tau, delta):
            return algorithm3_multiagent_reasoning(agents, self.ds.df, self.ontology, list(self.queries[split]),
                                                   lam=rc.lam, tau=tau, gamma=rc.gamma, delta=delta,
                                                   pool=self.pool, prior=self.prior)
        return self.tuned(run, {"tau": rc.tau_grid, "delta": rc.delta_grid})

    def neural_only(self, scorer):
        """Pure neural decision: accept a candidate iff σ(f) >= θ_r (θ_r tuned on validation)."""
        def run(split, theta):
            out = {}
            for key, cand in self.queries[split].items():
                tails = list(cand)
                p = scorer.prob([(key[0], key[1], t) for t in tails])
                acc = [t for t, pt in zip(tails, p) if pt >= theta]
                if acc:
                    out[key] = Conclusion(acc, {t: cand[t] for t in acc}, {})
            return out
        return self.tuned(run, {"theta": self.cfg.reason.theta_grid})

    def majority(self):
        out = {}
        for key, cand in self.queries["test"].items():
            top = max(cand.values())
            tails = [t for t, n in cand.items() if n == top]
            out[key] = Conclusion(tails, {t: cand[t] for t in tails}, {})
        return self.score(out)

    def rule_based(self, k=8):
        """Symbolic rules without learning: numeric values follow the median of the
        k nearest training locations; other tails follow the relation's most
        frequent training tail."""
        train = self.ds.split("train")
        freq = {r: Counter(g["tail"]) for r, g in train.groupby("relation")}
        trees = {}
        for r, g in train.groupby("relation"):
            g = g[g["head"].map(is_location) & g["tail"].map(is_number)]
            if len(g):
                xy = np.array([location_coords(h) for h in g["head"]], dtype=float)
                trees[r] = (cKDTree(xy), g["tail"].astype(float).values)
        out = {}
        for key, cand in self.queries["test"].items():
            h, r = key
            tails = list(cand)
            nums = [t for t in tails if is_number(t)]
            if r in trees and is_location(h) and nums:
                tree, vals = trees[r]
                _, idx = tree.query(np.array(location_coords(h), dtype=float), k=min(k, len(vals)))
                med = float(np.median(vals[np.atleast_1d(idx)]))
                best = min(nums, key=lambda t: abs(float(t) - med))
            else:
                best = max(tails, key=lambda t: freq.get(r, Counter())[t])
            out[key] = Conclusion([best], {best: cand[best]}, {})
        return self.score(out)


def run_seed(task: Task, include_baselines=True, include_ablations=True, verbose=False) -> dict:
    """All methods on one seed. Returns {method: metrics} plus training logs."""
    cfg = task.cfg
    res, logs, timing = {}, {}, {}

    t0 = time.perf_counter()
    res["Algorithm 1 (paraconsistent)"] = task.alg1()
    timing["Algorithm 1"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gnn, info = task.train("rgcn", "bilinear", verbose=verbose)
    logs["proposed_scorer"] = info
    res["Algorithm 2 (learning-assisted)"], params2 = task.alg2(gnn)
    timing["Algorithm 2"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    agents, infos = task.make_agents(cfg.reason.num_agents)
    logs["agents"] = infos
    res["Algorithm 3 (multi-agent)"], params3 = task.alg3(agents)
    timing["Algorithm 3"] = time.perf_counter() - t0

    if include_ablations:
        from kg.ontology import Ontology
        no_range = Ontology(task.ontology.individuals, task.ontology.subclass, task.ontology.functional,
                            task.ontology.domains, {})
        kb_nr = ParaconsistentKB(task.ds.df[["head", "relation", "tail"]].itertuples(index=False, name=None),
                                 no_range)
        res["Ablation: Alg.1 without range axioms"] = task.alg1(kb_nr)
        res["Ablation: GNN scorer, no paraconsistent layer"], _ = task.neural_only(gnn)
        shallow, info_s = task.train("embed", "bilinear")
        logs["shallow_scorer"] = info_s
        res["Ablation: Alg.2 with shallow scorer (no GNN)"], _ = task.alg2(shallow)
        res["Ablation: Alg.3 with equal agent weights"], _ = task.alg3(agents, equal_weights=True)

    if include_baselines:
        res["Majority vote"] = task.majority()
        res["Rule-based (spatial kNN rules)"] = task.rule_based()
        for label, enc, dec in NEURAL_BASELINES:
            t0 = time.perf_counter()
            scorer, info = task.train(enc, dec)
            res[label], _ = task.neural_only(scorer)
            timing[label] = time.perf_counter() - t0
            logs[label] = {k: v for k, v in info.items() if k != "history"}

    return {"metrics": res, "params": {"alg2": params2, "alg3": params3}, "logs": logs, "timing": timing,
            "kb": {"triples": len(task.ds.df), "conflict_keys": len(task.kb.conflict_keys()),
                   "rejected_by_ontology": int(sum(task.kb.rejected.values())),
                   "log10_fragments": task.kb.log10_num_fragments()}}
