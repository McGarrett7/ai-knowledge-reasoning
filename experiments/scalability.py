"""Runtime and scalability study (reviewer R1-B, R2-4).

1. Knowledge-base size: SolarKG-style graphs from 5k to ~400k triples
   (sizes above the 19.7k sampled locations are synthetic upscalings, used
   for timing only). Measured per size: fragment construction + Alg. 1,
   scorer training time per epoch, Alg. 2 / Alg. 3 inference, peak memory.
   A log-log fit gives the empirical growth exponent.
2. Number of agents m for Alg. 3 at fixed size.
3. Exhaustive fragment enumeration vs. the local decomposition on small K,
   which shows the exponential cost that Proposition 1 avoids.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
import psutil
import torch

from algorithms.algorithm1_paraconsistent import algorithm1_paraconsistent_reasoning
from algorithms.algorithm2_learning import ModelScorer, algorithm2_learning_reasoning
from algorithms.algorithm3_multiagent import Agent, algorithm3_multiagent_reasoning, partition_agents
from algorithms.paraconsistent import ParaconsistentKB, naive_maximal_consistent_subsets
from config import RESULTS_DIR, SOLAR_CSV, ExperimentConfig
from kg.construction import samples_to_triples, synthetic_upscale
from kg.data import KGDataset, assign_split, inject_noise
from kg.ontology import Ontology, solar_ontology
from models.kge import KGModel, train_model


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20


def build(n_locations: int, cfg: ExperimentConfig, seed: int = 0) -> pd.DataFrame:
    samples = pd.read_csv(SOLAR_CSV)
    if n_locations <= len(samples):
        samples = samples.sample(n=n_locations, random_state=seed)
    else:
        samples = synthetic_upscale(samples, n_locations, seed)
    kg = samples_to_triples(samples).drop_duplicates()
    kg = assign_split(kg, cfg.data.split, seed=seed)
    return inject_noise(kg, int(0.06 * len(kg)), cfg.data.noise_low, cfg.data.noise_high, seed)


def time_size(n_locations: int, cfg: ExperimentConfig, timing_epochs: int = 2, agents: int = 3,
              fanout=(10, 5), full_graph_limit: int = 45000) -> dict:
    df = build(n_locations, cfg)
    row = {"locations": n_locations, "triples": len(df)}
    mem0 = rss_mb()
    onto = solar_ontology(cfg.data.pv_range)

    t = time.perf_counter()
    kb = ParaconsistentKB(df[["head", "relation", "tail"]].itertuples(index=False, name=None), onto)
    row["fragments_s"] = time.perf_counter() - t
    row["log10_fragments"] = kb.log10_num_fragments()

    ds = KGDataset(df, freqs=cfg.train.fourier_freqs, knn=cfg.data.knn_neighbors)
    test = ds.candidates("test")
    keys = list(test)
    t = time.perf_counter()
    algorithm1_paraconsistent_reasoning(kb, keys)
    row["alg1_s"] = time.perf_counter() - t

    graph = ds.build_graph()
    if len(df) <= full_graph_limit:
        # exact full-graph message passing per batch (setting of the accuracy results)
        tf = ExperimentConfig().train
        tf.epochs = tf.patience = 1
        info_f = train_model(KGModel("rgcn", "bilinear", ds, graph, tf), ds, ds.split("train"), tf, seed=0)
        row["train_full_graph_s_per_epoch"] = info_f["train_seconds"]
    tc = ExperimentConfig().train
    tc.epochs, tc.patience, tc.fanout = timing_epochs, timing_epochs, fanout
    model = KGModel("rgcn", "bilinear", ds, graph, tc)
    info = train_model(model, ds, ds.split("train"), tc, seed=0)
    row["train_s_per_epoch"] = info["train_seconds"] / len(info["history"])
    scorer = ModelScorer(model, ds)
    pool = {r: list(g["tail"].unique()) for r, g in ds.split("train").groupby("relation")}
    t = time.perf_counter()
    algorithm2_learning_reasoning(kb, scorer, keys, pool=pool)
    row["alg2_inference_s"] = time.perf_counter() - t

    # Alg. 3: agents reuse the trained scorer for timing; local training cost
    # scales as m x (local share) x per-epoch cost and is reported separately.
    parts = partition_agents(ds.df, agents, cfg.reason.overlap, 0)
    ag = [Agent(i, rows, scorer, 1.0) for i, rows in enumerate(parts)]
    t = time.perf_counter()
    algorithm3_multiagent_reasoning(ag, ds.df, onto, keys, pool=pool)
    row["alg3_inference_s"] = time.perf_counter() - t
    row["alg3_local_train_share"] = float(np.mean([len(p) for p in parts]) / len(ds.df))
    row["peak_rss_mb"] = rss_mb()
    row["rss_increase_mb"] = row["peak_rss_mb"] - mem0
    return row


def time_agents(cfg: ExperimentConfig, ms=(1, 2, 4, 8, 16), n_locations: int = 5000, epochs: int = 2) -> list:
    df = build(n_locations, cfg)
    ds = KGDataset(df, freqs=cfg.train.fourier_freqs, knn=cfg.data.knn_neighbors)
    onto = solar_ontology(cfg.data.pv_range)
    keys = list(ds.candidates("test"))
    pool = {r: list(g["tail"].unique()) for r, g in ds.split("train").groupby("relation")}
    tc = ExperimentConfig().train
    tc.epochs, tc.patience, tc.fanout = epochs, epochs, (10, 5)
    rows = []
    for m in ms:
        parts = partition_agents(ds.df, m, cfg.reason.overlap, 0)
        t = time.perf_counter()
        agents = []
        for i, rows_i in enumerate(parts):
            local = ds.df.loc[rows_i]
            local_train = local[local["split"] == "train"]
            model = KGModel("rgcn", "bilinear", ds, ds.build_graph(local_train), tc)
            train_model(model, ds, local_train, tc, seed=i)
            agents.append(Agent(i, rows_i, ModelScorer(model, ds), 1.0))
        train_s = time.perf_counter() - t
        t = time.perf_counter()
        algorithm3_multiagent_reasoning(agents, ds.df, onto, keys, pool=pool)
        rows.append({"agents": m, "train_s": train_s, "train_s_per_epoch": train_s / epochs,
                     "inference_s": time.perf_counter() - t})
        print(f"  m={m}: train {train_s:.1f}s, inference {rows[-1]['inference_s']:.2f}s")
    return rows


def naive_vs_local(max_conflicts: int = 7) -> list:
    """Exhaustive MC(K) enumeration vs. per-key clusters on tiny KBs."""
    onto = Ontology(functional={"p": 1.0})
    rows = []
    for c in range(1, max_conflicts + 1):
        triples = [(f"e{i}", "p", f"v{j}") for i in range(c) for j in range(2)]
        t = time.perf_counter()
        mcs = naive_maximal_consistent_subsets(triples, onto)
        naive = time.perf_counter() - t
        t = time.perf_counter()
        kb = ParaconsistentKB(triples, onto)
        _ = [kb.credulous((f"e{i}", "p")) for i in range(c)]
        local = time.perf_counter() - t
        rows.append({"conflict_keys": c, "triples": len(triples), "fragments": len(mcs),
                     "naive_s": naive, "local_s": local})
    return rows


def run_scalability(cfg: ExperimentConfig, sizes=(2500, 5000, 10000, 19757, 50000, 100000, 200000)) -> dict:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    out = {"hardware": {"cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                        "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1)},
           "size": [], "agents": [], "naive_vs_local": naive_vs_local()}
    for n in sizes:
        row = time_size(n, cfg)
        out["size"].append(row)
        print(f"  {row['triples']:>7} triples: fragments {row['fragments_s']:.3f}s, "
              f"epoch {row['train_s_per_epoch']:.1f}s, alg2 {row['alg2_inference_s']:.2f}s, "
              f"alg3 {row['alg3_inference_s']:.2f}s, rss {row['peak_rss_mb']:.0f}MB")
    out["agents"] = time_agents(cfg)
    k = np.log([r["triples"] for r in out["size"]])
    for col in ("fragments_s", "train_s_per_epoch", "alg2_inference_s", "alg3_inference_s"):
        y = np.log([max(r[col], 1e-6) for r in out["size"]])
        out.setdefault("loglog_slope", {})[col] = float(np.polyfit(k, y, 1)[0])
    (RESULTS_DIR / "scalability.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
