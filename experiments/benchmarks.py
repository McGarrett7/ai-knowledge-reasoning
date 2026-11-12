"""Standard KG benchmarks (reviewer R2-2, R2-7, R2-8).

Datasets: UMLS, Kinship, Nations (ConvE splits), CoDEx-S (Wikidata,
Safavi & Koutra 2020) and optionally WN18RR / FB15k-237.

UMLS, Kinship and Nations have almost no functional relations (Kinship: none
at confidence 0.9), so they contain (almost) no logical conflicts under Func
axioms; on them the task reduces to noisy-fact filtering. CoDEx-S has 16
functional relations (sex, place of birth, ...) and is the benchmark for
conflict resolution.

Protocol
  * Ontology: Func(r) axioms are mined from the CLEAN training split and kept
    when their confidence >= tau_func (operator G, Sec. 4.3). They play the
    role of schema constraints (e.g. Wikidata single-value constraints).
  * Conflict noise (as on SolarKG-VN): every (h, r) key of a functional
    relation receives, with probability conflict_rate, a second contradicting
    tail t' drawn from the relation's tail pool.
  * Random noise: tail-corrupted triples of the remaining relations at the
    SolarKG-VN rate (5.8% of their triples); false, but not contradicting O.
  * A corrupted tail is never a true fact, so its label is known.
  * Task A - noisy-fact reasoning: the queries are the (h, r) keys of the test
    split with all asserted tails; metrics as on SolarKG-VN.
  * Task B - link prediction: filtered MRR and Hits@{1,3,10} on the clean test
    triples (head and tail prediction) for every scorer, trained on the noisy
    training split.
  * Training (identical for every model, BENCH_TRAIN): 1-N scoring with
    multi-label BCE over all entities for (h, r, ?) and (?, r, t), Adam 3e-3,
    up to 300 epochs, model selection on filtered validation MRR.
  * Each seed is written to results/benchmarks/shards/<name>_seed<s>.json, so the
    seeds can run in parallel; summarize_benchmark merges them.
"""
from __future__ import annotations

import copy
import json
import tarfile
import time
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
from config import BENCH_RESULTS_DIR as RESULTS_DIR
from config import BENCHMARK_DIR, ExperimentConfig
from experiments.evaluation_metrics import aggregate
from experiments.pipeline import NEURAL_BASELINES, Task
from kg.data import KGDataset
from kg.ontology import Ontology, mine_axioms
from models.kge import rank_metrics

URL = "https://github.com/TimDettmers/ConvE/raw/master/{name}.tar.gz"
CODEX_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/{name}/{split}.txt"
DATASETS = ("umls", "kinship", "nations", "codex-s")
SHARD_DIR = RESULTS_DIR / "shards"


def download(name: str):
    target = BENCHMARK_DIR / name
    if (target / "train.txt").exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"downloading {name} ...")
    if name.startswith("codex"):
        for split in ("train", "valid", "test"):
            urllib.request.urlretrieve(CODEX_URL.format(name=name, split=split), target / f"{split}.txt")
        return target
    archive = BENCHMARK_DIR / f"{name}.tar.gz"
    urllib.request.urlretrieve(URL.format(name=name), archive)
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            base = member.name.split("/")[-1]
            if base in ("train.txt", "valid.txt", "test.txt") and member.isfile():
                member.name = base
                tar.extract(member, target, filter="data")
    archive.unlink()
    return target


def load_benchmark(name: str, noise_rate: float, seed: int, tau_func: float = 0.9,
                   conflict_rate: float = 0.3):
    """Clean benchmark + injected noise. Returns (df, truth, functional relations)."""
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
    train = clean[clean["split"] == "train"]
    mined = mine_axioms(train[["head", "relation", "tail"]].itertuples(index=False, name=None))
    functional = Ontology().extend(mined, tau_func).functional        # schema: Func mined on clean train
    rng = np.random.default_rng(seed)
    pool = clean.groupby("relation")["tail"].unique().to_dict()

    def corrupt(h, r):
        cands = [c for c in pool[r] if c not in truth[(h, r)]]
        return cands[int(rng.integers(len(cands)))] if cands else None

    noise = []
    is_func = clean["relation"].isin(list(functional))
    for split, grp in clean[is_func].groupby("split"):         # conflict noise, at most one per key
        keys = grp.drop_duplicates(["head", "relation"])
        chosen = keys[rng.random(len(keys)) < conflict_rate]
        for h, r in chosen[["head", "relation"]].itertuples(index=False):
            t = corrupt(h, r)
            if t is not None:
                noise.append((h, r, t, split, "conflict"))
    for split, grp in clean[~is_func].groupby("split"):        # random, non-contradicting noise
        n = int(round(noise_rate * len(grp)))
        for h, r, _ in grp.sample(n=n, random_state=seed)[["head", "relation", "tail"]].itertuples(index=False):
            t = corrupt(h, r)
            if t is not None:
                noise.append((h, r, t, split, "random"))
    noise = pd.DataFrame(noise, columns=["head", "relation", "tail", "split", "source"])
    noise["label"] = 0
    return pd.concat([clean, noise], ignore_index=True), dict(truth), functional


def link_prediction(scorer, ds: KGDataset, truth: dict) -> dict:
    """Filtered MRR / Hits@k on the clean test triples (head and tail prediction)."""
    test = ds.split("test")
    return rank_metrics(scorer.model, ds, test[test["label"] == 1], truth, scorer.emb)


# Training protocol for the benchmarks (the same for every model): 1-N scoring
# with multi-label BCE, model selection on filtered validation MRR.
BENCH_TRAIN = {"objective": "kvsall", "select_metric": "mrr", "epochs": 300, "eval_every": 5,
               "patience": 50, "batch_size": 128, "lr": 3e-3, "label_smoothing": 0.0, "fanout": None}


def run_benchmark(name: str, cfg: ExperimentConfig, seeds, noise_rate=0.058, tau_func=0.9, conflict_rate=0.3,
                  train_overrides: dict | None = None):
    cfg = copy.deepcopy(cfg)
    for k, v in {**BENCH_TRAIN, **(train_overrides or {})}.items():
        setattr(cfg.train, k, v)
    per_seed = []
    for seed in seeds:
        t0 = time.perf_counter()
        df, truth, functional = load_benchmark(name, noise_rate, seed, tau_func, conflict_rate)
        ds = KGDataset(df, truth=truth)
        onto = Ontology(functional=dict(functional))                  # O' = O ∪ Γ_tau with O = ∅
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
                                   "conflict_noise": int((df["source"] == "conflict").sum()),
                                   "random_noise": int((df["source"] == "random").sum()),
                                   "functional_relations": len(onto.functional),
                                   "test_functional_keys": sum(1 for k in task.queries["test"]
                                                               if onto.is_functional(k[1])),
                                   "test_conflict_keys": sum(1 for k, c in task.queries["test"].items()
                                                             if onto.is_functional(k[1]) and len(c) > 1)}})
        per_seed[-1]["seconds"] = time.perf_counter() - t0
        per_seed[-1]["seed"] = seed
        print(f"  {name} seed {seed} done in {per_seed[-1]['seconds']:.0f}s: {per_seed[-1]['stats']}", flush=True)
        print("    MRR " + ", ".join(f"{m}={v['MRR']:.3f}" for m, v in lp.items()), flush=True)
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        (SHARD_DIR / f"{name}_seed{seed}.json").write_text(json.dumps({
            **per_seed[-1], "protocol": {"noise_rate": noise_rate, "tau_func": tau_func,
                                         "conflict_rate": conflict_rate, "functional_relations": sorted(functional),
                                         "train": vars(cfg.train)}}, indent=2), encoding="utf-8")
    return summarize_benchmark(name)


def summarize_benchmark(name: str) -> dict:
    """Merge every per-seed shard of a benchmark (several processes may share the seeds)."""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SHARD_DIR.glob(f"{name}_seed*.json"))]
    if not runs:
        raise SystemExit(f"no shards for {name} in {SHARD_DIR}")
    summary = {
        "seeds": [r["seed"] for r in runs],
        "stats": runs[0]["stats"],
        "reasoning": {m: aggregate([r["reasoning"][m] for r in runs]) for m in runs[0]["reasoning"]},
        "link_prediction": {m: aggregate([r["link_prediction"][m] for r in runs])
                            for m in runs[0]["link_prediction"]},
        "protocol": runs[0]["protocol"],
    }
    out = RESULTS_DIR / f"benchmark_{name}.json"
    out.write_text(json.dumps({"summary": summary, "per_seed": runs}, indent=2), encoding="utf-8")
    return summary
