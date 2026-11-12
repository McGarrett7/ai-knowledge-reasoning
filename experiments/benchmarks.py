"""Standard KG benchmarks (reviewer R2-2, R2-7, R2-8).

Datasets: UMLS, Kinship, Nations (default; every model trains on CPU) and
optionally WN18RR / FB15k-237 (standard splits from the ConvE repository).

Protocol
  * Inject tail-corrupted triples into every split at the SolarKG-VN rate
    (5.8% of triples). A corrupted tail is drawn from the relation's tail pool
    and is never a true fact, so its label is known.
  * Ontology: Func(r) axioms are mined from the noisy training split and kept
    when their confidence >= tau_func (operator G, Sec. 4.3).
  * Task A - noisy-fact reasoning: the queries are the (h, r) keys of the test
    split with all asserted tails; metrics as on SolarKG-VN.
  * Task B - link prediction: filtered MRR and Hits@{1,3,10} on the clean test
    triples (head and tail prediction) for every scorer, trained on the noisy
    training split.
"""
from __future__ import annotations

import json
import tarfile
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from config import BENCHMARK_DIR, RESULTS_DIR, ExperimentConfig
from experiments.evaluation_metrics import aggregate
from experiments.pipeline import NEURAL_BASELINES, Task
from kg.data import KGDataset
from kg.ontology import Ontology, mine_axioms

URL = "https://github.com/TimDettmers/ConvE/raw/master/{name}.tar.gz"
DATASETS = ("umls", "kinship", "nations")


def download(name: str):
    target = BENCHMARK_DIR / name
    if (target / "train.txt").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    archive = BENCHMARK_DIR / f"{name}.tar.gz"
    print(f"downloading {name} ...")
    urllib.request.urlretrieve(URL.format(name=name), archive)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            base = member.name.split("/")[-1]
            if base in ("train.txt", "valid.txt", "test.txt") and member.isfile():
                member.name = base
                tar.extract(member, target, filter="data")
    archive.unlink()
    return target


def load_benchmark(name: str, noise_rate: float, seed: int):
    path = download(name)
    parts = []
    for split, fname in (("train", "train.txt"), ("val", "valid.txt"), ("test", "test.txt")):
        df = pd.read_csv(path / fname, sep="\t", header=None, names=["head", "relation", "tail"], dtype=str)
        df["split"] = split
        parts.append(df)
    clean = pd.concat(parts, ignore_index=True).drop_duplicates(["head", "relation", "tail"])
    truth = defaultdict(set)
    for h, r, t in clean[["head", "relation", "tail"]].itertuples(index=False):
        truth[(h, r)].add(t)
    clean["label"], clean["source"] = 1, "data"
    rng = np.random.default_rng(seed)
    pool = clean.groupby("relation")["tail"].unique().to_dict()
    noise = []
    for split, grp in clean.groupby("split"):
        n = int(round(noise_rate * len(grp)))
        for h, r, t in grp.sample(n=n, random_state=seed)[["head", "relation", "tail"]].itertuples(index=False):
            cands = [c for c in pool[r] if c not in truth[(h, r)]]
            if cands:
                noise.append((h, r, cands[int(rng.integers(len(cands)))], split))
    noise = pd.DataFrame(noise, columns=["head", "relation", "tail", "split"])
    noise["label"], noise["source"] = 0, "noise"
    return pd.concat([clean, noise], ignore_index=True), dict(truth)


@torch.no_grad()
def link_prediction(scorer, ds: KGDataset, truth: dict) -> dict:
    model, emb = scorer.model, scorer.emb
    test = ds.split("test")
    test = test[test["label"] == 1]
    x = ds.encode(test)
    heads_of = defaultdict(set)
    for (h, r), ts in truth.items():
        for t in ts:
            heads_of[(t, r)].add(h)
    ranks = []
    for side in ("tail", "head"):
        fixed = x[:, 0] if side == "tail" else x[:, 2]
        scores = model.tail_scores(fixed, x[:, 1], emb) if side == "tail" else model.head_scores(fixed, x[:, 1], emb)
        for i, (h, r, t) in enumerate(test[["head", "relation", "tail"]].itertuples(index=False)):
            target = ds.ent2id[t] if side == "tail" else ds.ent2id[h]
            known = truth.get((h, r), set()) if side == "tail" else heads_of[(t, r)]
            s = scores[i].clone()
            target_score = s[target].item()
            filt = [ds.ent2id[e] for e in known if e in ds.ent2id and ds.ent2id[e] != target]
            if filt:
                s[filt] = -float("inf")
            ranks.append(1 + int((s > target_score).sum()))
    ranks = np.array(ranks, dtype=float)
    return {"MRR": float(np.mean(1 / ranks)), "Hits@1": float(np.mean(ranks <= 1)),
            "Hits@3": float(np.mean(ranks <= 3)), "Hits@10": float(np.mean(ranks <= 10))}


def run_benchmark(name: str, cfg: ExperimentConfig, seeds, noise_rate=0.058, tau_func=0.9):
    per_seed = []
    for seed in seeds:
        df, truth = load_benchmark(name, noise_rate, seed)
        if len(df) > 20000 and not cfg.train.fanout:
            cfg.train.fanout = (10, 5)        # WN18RR / FB15k-237: sampled message passing
        ds = KGDataset(df, truth=truth)
        gamma_axioms = mine_axioms(ds.split("train")[["head", "relation", "tail"]].itertuples(index=False, name=None))
        onto = Ontology().extend(gamma_axioms, tau_func)           # O' = O ∪ Γ_tau with O = ∅
        task = Task(ds, onto, cfg, seed)
        res, lp = {}, {}
        res["Algorithm 1 (paraconsistent)"] = task.alg1()
        gnn, _ = task.train("rgcn", "bilinear")
        res["Algorithm 2 (learning-assisted)"], _ = task.alg2(gnn)
        lp["Proposed scorer (R-GCN + bilinear)"] = link_prediction(gnn, ds, truth)
        agents, _ = task.make_agents(cfg.reason.num_agents)
        res["Algorithm 3 (multi-agent)"], _ = task.alg3(agents)
        res["Majority vote"] = task.majority()
        res["Rule-based (frequency rules)"] = task.rule_based()
        for label, enc, dec in NEURAL_BASELINES:
            scorer, _ = task.train(enc, dec)
            res[label], _ = task.neural_only(scorer)
            lp[label] = link_prediction(scorer, ds, truth)
        per_seed.append({"reasoning": res, "link_prediction": lp,
                         "stats": {"entities": len(ds.entities), "relations": len(ds.relations),
                                   "triples": len(df), "noise": int((df["label"] == 0).sum()),
                                   "functional_relations": len(onto.functional),
                                   "test_conflict_keys": sum(1 for k, c in task.queries["test"].items()
                                                             if onto.is_functional(k[1]) and len(c) > 1)}})
        print(f"  {name} seed {seed} done")
    summary = {
        "stats": per_seed[0]["stats"],
        "reasoning": {m: aggregate([s["reasoning"][m] for s in per_seed]) for m in per_seed[0]["reasoning"]},
        "link_prediction": {m: aggregate([s["link_prediction"][m] for s in per_seed])
                            for m in per_seed[0]["link_prediction"]},
    }
    out = RESULTS_DIR / f"benchmark_{name}.json"
    out.write_text(json.dumps({"summary": summary, "per_seed": per_seed}, indent=2), encoding="utf-8")
    return summary
