"""Additional analyses on SolarKG-VN (ablation questions of R2-1 / R2-3 / R2-8).

A. Scorer choice under stress (`python main.py robustness`)
   Settings: noise_triples in {300, 800, 1600} with the full training split
   (about 5.8%, 15% and 30% injected triples), and training fraction
   {0.5, 0.25} at 300 noise triples (label scarcity).
   Methods: Algorithm 1; Algorithm 2 with three scorers that differ only in
   the encoder - ID embedding (classic shallow KGE: no features, no message
   passing), feature MLP (inductive features, no message passing) and R-GCN
   (proposed: features + message passing over nearTo / KG edges); the R-GCN
   scorer without the paraconsistent layer; Algorithm 3.

B. Heterogeneous agents (`python main.py agents`)
   One of the m = 3 agents observes an unreliable source: for a fraction q of
   its triples it additionally holds a corrupted copy whose value lies inside
   the valid range, so the ontology cannot reject it. The corrupted copies
   are seen only by that agent (and therefore enter K = ∪ B_i).
   Agent weights π_i compared: equal; validation accuracy of the agent's
   scorer (Algorithm 3 as in the paper); its log-odds log(acc / (1 - acc))
   (optimal weighting of independent voters, Nitzan & Paroush 1982); and the
   log-odds of the agent's source precision (share of its validation-split
   assertions that are true), which measures data quality rather than model
   quality.

Every threshold is tuned on the validation split, exactly as in the main run.
"""
from __future__ import annotations

import copy
import json
import math
import time

import numpy as np
import pandas as pd

from algorithms.algorithm3_multiagent import Agent, partition_agents
from config import ANALYSIS_DIR as RESULTS_DIR
from config import KG_CSV, REL_POTENTIAL, ExperimentConfig, ensure_dirs
from experiments.evaluation_metrics import aggregate
from experiments.pipeline import Task
from kg.data import KGDataset, load_triples, prepare_solar
from kg.ontology import normalize_value, solar_ontology

A1, A2, A3 = "Algorithm 1 (paraconsistent)", "Algorithm 2 (learning-assisted)", "Algorithm 3 (multi-agent)"
SCORERS = [("Alg. 2 + ID embedding (no features, no GNN)", "id"),
           ("Alg. 2 + feature MLP (no GNN)", "embed"),
           ("Alg. 2 + R-GCN (proposed)", "rgcn")]
GNN_ONLY = "R-GCN only (no paraconsistent layer)"
NOISE_LEVELS = (300, 800, 1600)
TRAIN_FRACTIONS = (0.5, 0.25)
KEYS = ("accuracy", "f1", "correctness", "conflict_resolution_acc")


def _dataset(cfg: ExperimentConfig, seed: int, noise: int, frac: float):
    dcfg = copy.deepcopy(cfg.data)
    dcfg.noise_triples = noise
    df = prepare_solar(load_triples(KG_CSV), dcfg, seed, noisy=True)
    if frac < 1.0:
        train = df.index[df["split"] == "train"]
        rng = np.random.default_rng(seed + 101)
        drop = rng.choice(train, size=int(round((1 - frac) * len(train))), replace=False)
        df = df.drop(drop).reset_index(drop=True)
    return df


def _unseen_heads(ds: KGDataset) -> float:
    train_heads = set(ds.split("train")["head"])
    test_heads = set(ds.split("test")["head"])
    return len(test_heads - train_heads) / max(1, len(test_heads))


# ------------------------------------------------------------------ A
def robustness_setting(cfg, seed, noise, frac) -> dict:
    df = _dataset(cfg, seed, noise, frac)
    onto = solar_ontology(cfg.data.pv_range)
    ds = KGDataset(df, freqs=cfg.train.fourier_freqs, knn=cfg.data.knn_neighbors)
    task = Task(ds, onto, cfg, seed)
    res = {A1: task.alg1()}
    gnn = None
    for label, enc in SCORERS:
        scorer, _ = task.train(enc, "bilinear")
        res[label], _ = task.alg2(scorer)
        if enc == "rgcn":
            gnn = scorer
    res[GNN_ONLY], _ = task.neural_only(gnn)
    agents, _ = task.make_agents(cfg.reason.num_agents)
    res[A3], _ = task.alg3(agents)
    stats = {"triples": len(df), "train_triples": int((df["split"] == "train").sum()),
             "conflict_keys": len(task.kb.conflict_keys()), "test_heads_unseen_in_train": _unseen_heads(ds)}
    return {"metrics": res, "stats": stats}


def run_robustness(cfg: ExperimentConfig, seeds=(0, 1, 2)) -> dict:
    ensure_dirs()
    settings = [("noise", n, n, 1.0) for n in NOISE_LEVELS] + \
               [("train_fraction", f, NOISE_LEVELS[0], f) for f in TRAIN_FRACTIONS]
    out = {"seeds": list(seeds), "settings": []}
    for kind, value, noise, frac in settings:
        runs = []
        for seed in seeds:
            t0 = time.perf_counter()
            runs.append(robustness_setting(cfg, seed, noise, frac))
            m = runs[-1]["metrics"]
            print(f"  {kind}={value} seed {seed} [{cfg.train.device}] {time.perf_counter() - t0:.0f}s  " +
                  "  ".join(f"{k[:22]}: acc={v['accuracy']:.4f} cr={v['conflict_resolution_acc']:.3f}"
                            for k, v in m.items()), flush=True)
        out["settings"].append({
            "kind": kind, "value": value, "noise_triples": noise, "train_fraction": frac,
            "stats": {k: float(np.mean([r["stats"][k] for r in runs])) for k in runs[0]["stats"]},
            "metrics": {m: aggregate([r["metrics"][m] for r in runs]) for m in runs[0]["metrics"]}})
        (RESULTS_DIR / "analysis_robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_markdown()
    return out


# ------------------------------------------------------------------ B
def corrupt_agent(df: pd.DataFrame, rows: pd.Index, q: float, pv_range, seed: int):
    """Corrupted in-range copies of a fraction q of the hasSolarPotential triples in `rows`."""
    rng = np.random.default_rng(seed + 202)
    cand = [i for i in rows if df.at[i, "relation"] == REL_POTENTIAL and df.at[i, "label"] == 1]
    pick = rng.choice(cand, size=int(round(q * len(cand))), replace=False) if cand else []
    lo, hi = int(pv_range[0]), int(pv_range[1])
    new = []
    for i in pick:
        true = df.at[i, "tail"]
        v = normalize_value(int(rng.integers(lo, hi + 1)))
        while v == true:
            v = normalize_value(int(rng.integers(lo, hi + 1)))
        row = df.loc[i].copy()
        row["tail"], row["label"], row["source"] = v, 0, "agent_noise"
        new.append(row)
    extra = pd.DataFrame(new)
    extra.index = range(len(df), len(df) + len(extra))
    return pd.concat([df, extra]), extra.index


def logodds(p: float) -> float:
    p = min(max(p, 0.5 + 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


def source_precision(df: pd.DataFrame, rows) -> float:
    """Data reliability of an agent: share of its validation-split assertions that are true."""
    local = df.loc[rows]
    val = local[local["split"] == "val"]
    return float(val["label"].mean()) if len(val) else 1.0


WEIGHTINGS = (("equal", "Alg. 3, equal weights"),
              ("accuracy", "Alg. 3, π = scorer val. accuracy (paper)"),
              ("logodds", "Alg. 3, π = log-odds of scorer val. accuracy"),
              ("source", "Alg. 3, π = log-odds of source precision on val."))


def weight(acc: float, prec: float, scheme: str) -> float:
    return {"equal": 1.0, "accuracy": acc, "logodds": logodds(acc), "source": logodds(prec)}[scheme]


def agents_setting(cfg, seed, q, unreliable=0) -> dict:
    onto = solar_ontology(cfg.data.pv_range)
    df = prepare_solar(load_triples(KG_CSV), cfg.data, seed, noisy=True)
    m = cfg.reason.num_agents
    parts = partition_agents(df, m, cfg.reason.overlap, seed)
    if q > 0:
        df, extra = corrupt_agent(df, parts[unreliable], q, cfg.data.pv_range, seed)
        parts[unreliable] = parts[unreliable].append(extra)
    ds = KGDataset(df, freqs=cfg.train.fourier_freqs, knn=cfg.data.knn_neighbors)
    task = Task(ds, onto, cfg, seed)
    res = {A1: task.alg1()}
    gnn, _ = task.train("rgcn", "bilinear")
    res["Alg. 2 (one scorer on all data)"], _ = task.alg2(gnn)
    trained = []
    for rows in parts:
        local = ds.df.loc[rows]
        scorer, info = task.train("rgcn", "bilinear", train_df=local[local["split"] == "train"])
        trained.append((rows, scorer, info["best_val_acc"], source_precision(ds.df, rows)))
    accs = [a for _, _, a, _ in trained]
    precs = [p for _, _, _, p in trained]
    res_w = {}
    for scheme, label in WEIGHTINGS:
        agents = [Agent(i, rows, sc, weight(a, p, scheme)) for i, (rows, sc, a, p) in enumerate(trained)]
        res[label], _ = task.alg3(agents)
        res_w[scheme] = [weight(a, p, scheme) for a, p in zip(accs, precs)]
    stats = {"agent_val_acc": accs, "agent_source_precision": precs, "weights": res_w, "corrupted_copies": int((df["source"] == "agent_noise").sum()),
             "conflict_keys": len(task.kb.conflict_keys())}
    return {"metrics": res, "stats": stats}


def run_agents(cfg: ExperimentConfig, seeds=(0, 1, 2), q_values=(0.0, 0.5)) -> dict:
    ensure_dirs()
    out = {"seeds": list(seeds), "settings": []}
    for q in q_values:
        runs = []
        for seed in seeds:
            t0 = time.perf_counter()
            runs.append(agents_setting(cfg, seed, q))
            r = runs[-1]
            print(f"  q={q} seed {seed} [{cfg.train.device}] {time.perf_counter() - t0:.0f}s  agent acc="
                  f"{[round(a, 3) for a in r['stats']['agent_val_acc']]} source precision="
                  f"{[round(a, 3) for a in r['stats']['agent_source_precision']]}  " +
                  "  ".join(f"{k[:28]}: acc={v['accuracy']:.4f} cr={v['conflict_resolution_acc']:.3f}"
                            for k, v in r["metrics"].items()), flush=True)
        out["settings"].append({
            "unreliable_fraction": q,
            "stats": [r["stats"] for r in runs],
            "metrics": {m: aggregate([r["metrics"][m] for r in runs]) for m in runs[0]["metrics"]}})
        (RESULTS_DIR / "analysis_agents.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_markdown()
    return out


# ------------------------------------------------------------------ report
def _fmt(v):
    return f"{v['mean']:.4f} ± {v['std']:.4f}"


def write_markdown():
    lines = ["# Additional analyses (SolarKG-VN)", ""]
    p = RESULTS_DIR / "analysis_robustness.json"
    if p.exists():
        r = json.loads(p.read_text(encoding="utf-8"))
        lines += [f"## A. Scorer choice under noise and label scarcity (seeds {r['seeds']})", ""]
        for key, title in (("accuracy", "Accuracy"), ("conflict_resolution_acc", "Conflict resolution acc.")):
            cols = [f"{s['kind']}={s['value']}" for s in r["settings"]]
            lines += [f"### {title}", "", "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for m in r["settings"][0]["metrics"]:
                lines.append(f"| {m} | " + " | ".join(_fmt(s["metrics"][m][key]) for s in r["settings"]) + " |")
            lines.append("")
        lines += ["| Setting | triples | train triples | conflict keys | test heads unseen in train |",
                  "|---|---|---|---|---|"]
        for s in r["settings"]:
            st = s["stats"]
            lines.append(f"| {s['kind']}={s['value']} | {st['triples']:.0f} | {st['train_triples']:.0f} | "
                         f"{st['conflict_keys']:.0f} | {100 * st['test_heads_unseen_in_train']:.1f}% |")
        lines.append("")
        from experiments import plots
        plots.analysis_robustness(r)
    p = RESULTS_DIR / "analysis_agents.json"
    if p.exists():
        r = json.loads(p.read_text(encoding="utf-8"))
        lines += [f"## B. Heterogeneous agents (seeds {r['seeds']})", "",
                  "q = fraction of the unreliable agent's hasSolarPotential triples that receive an "
                  "in-range corrupted copy (seen only by that agent).", ""]
        cols = [f"q={s['unreliable_fraction']}" for s in r["settings"]]
        for key, title in (("accuracy", "Accuracy"), ("conflict_resolution_acc", "Conflict resolution acc.")):
            lines += [f"### {title}", "", "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for m in r["settings"][0]["metrics"]:
                lines.append(f"| {m} | " + " | ".join(_fmt(s["metrics"][m][key]) for s in r["settings"]) + " |")
            lines.append("")
        for s in r["settings"]:
            accs = np.array([st["agent_val_acc"] for st in s["stats"]])
            precs = np.array([st["agent_source_precision"] for st in s["stats"]])
            lines.append(f"- q={s['unreliable_fraction']}: per-agent scorer validation accuracy "
                         f"{np.round(accs.mean(0), 4).tolist()}, source precision on validation "
                         f"{np.round(precs.mean(0), 4).tolist()} (mean over seeds)")
        lines.append("")
    (RESULTS_DIR / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
