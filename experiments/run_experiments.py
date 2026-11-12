"""Main SolarKG-VN experiment: Tables 1-3 and 7-10, Figures 1-3, training dynamics."""
from __future__ import annotations

import json
import time

from config import MAIN_DIR as RESULTS_DIR
from config import KG_CSV, ExperimentConfig, ensure_dirs
from experiments import plots
from experiments.evaluation_metrics import aggregate
from experiments.pipeline import NEURAL_BASELINES, Task, run_seed
from kg.data import KGDataset, count_conflicts, load_triples, prepare_solar
from kg.ontology import solar_ontology

ALGS = plots.ALGS


def dataset_stats(clean, noisy, onto) -> dict:
    return {
        "clean_triples": len(clean),
        "locations": int(clean["head"].nunique()),
        "relations": sorted(clean["relation"].unique()),
        "clean_conflicts": count_conflicts(clean),
        "split": noisy[noisy["source"] != "noise"].groupby("split").size().to_dict(),
        "noisy_triples": len(noisy),
        "injected": int((noisy["source"] == "noise").sum()),
        "noisy_conflicts": count_conflicts(noisy),
        "noisy_conflict_rate": count_conflicts(noisy) / len(clean),
        "injected_rejected_by_ontology": int(sum(onto.violation(h, r, t) is not None for h, r, t in
                                                 noisy[noisy["source"] == "noise"][["head", "relation", "tail"]]
                                                 .itertuples(index=False))),
        "ontology_axioms": onto.describe(),
    }


def fmt(m: dict, key: str, digits=4) -> str:
    v = m[key]
    return f"{v['mean']:.{digits}f} ± {v['std']:.{digits}f}"


def table(rows: dict, cols: list) -> str:
    head = "| Method | " + " | ".join(c[1] for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    return head + "".join(f"| {name} | " + " | ".join(fmt(m, c[0]) for c in cols) + " |\n"
                          for name, m in rows.items())


SHARDS = RESULTS_DIR / "shards"


def run_main(cfg: ExperimentConfig, seeds=None, quick=False, verbose=False, summarize_after=True):
    """Run the given seeds; each seed is saved to results/main/shards/seed_<s>.json so
    several processes (e.g. one per GPU) can share the work. Then summarize."""
    ensure_dirs()
    SHARDS.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds or cfg.seeds)
    clean = load_triples(KG_CSV)
    onto = solar_ontology(cfg.data.pv_range)
    for seed in seeds:
        t0 = time.perf_counter()
        df = prepare_solar(clean, cfg.data, seed, noisy=True)
        stats = dataset_stats(clean, df, onto)
        ds = KGDataset(df, freqs=cfg.train.fourier_freqs, knn=cfg.data.knn_neighbors)
        noisy = run_seed(Task(ds, onto, cfg, seed), include_baselines=not quick,
                         include_ablations=not quick, verbose=verbose)
        dfc = prepare_solar(clean, cfg.data, seed, noisy=False)
        dsc = KGDataset(dfc, freqs=cfg.train.fourier_freqs, knn=cfg.data.knn_neighbors)
        cleanr = run_seed(Task(dsc, onto, cfg, seed), include_baselines=False, include_ablations=False)
        shard = {"seed": seed, "quick": quick, "device": cfg.train.device, "stats": stats,
                 "noisy": noisy["metrics"], "clean": cleanr["metrics"], "params": noisy["params"],
                 "kb": noisy["kb"], "timing": noisy["timing"],
                 "history": noisy["logs"]["proposed_scorer"]["history"]}
        (SHARDS / f"seed_{seed}.json").write_text(json.dumps(shard, indent=2, default=str), encoding="utf-8")
        print(f"seed {seed} [{cfg.train.device}]: {time.perf_counter() - t0:.0f}s  " + "  ".join(
            f"{plots.SHORT[a]} acc={noisy['metrics'][a]['accuracy']:.4f}" for a in ALGS), flush=True)
    return summarize(cfg) if summarize_after else None


def summarize(cfg: ExperimentConfig) -> dict:
    """Aggregate every seed shard in results/main/shards into tables, JSON and figures."""
    shards = sorted((json.loads(p.read_text(encoding="utf-8")) for p in SHARDS.glob("seed_*.json")),
                    key=lambda r: r["seed"])
    if not shards:
        raise SystemExit("no shards in results/main/shards - run `python main.py run` first")
    quick = any(r["quick"] for r in shards)
    common = set.intersection(*(set(r["noisy"]) for r in shards))
    noisy = {m: aggregate([r["noisy"][m] for r in shards]) for m in shards[0]["noisy"] if m in common}
    clean_s = {m: aggregate([r["clean"][m] for r in shards]) for m in shards[0]["clean"]}
    timing = {k: aggregate([{"s": r["timing"][k]} for r in shards])["s"] for k in shards[0]["timing"]}
    result = {"config": {"seeds": [r["seed"] for r in shards], "devices": sorted({r["device"] for r in shards}),
                         "data": vars(cfg.data), "train": vars(cfg.train), "reason": vars(cfg.reason)},
              "dataset": shards[0]["stats"],
              "noisy_conflicts_per_seed": [r["stats"]["noisy_conflicts"] for r in shards],
              "clean": clean_s, "noisy": noisy, "timing_seconds": timing,
              "per_seed": [{k: r[k] for k in ("seed", "noisy", "clean", "params", "kb", "timing")} for r in shards],
              "training_history_seed0": shards[0]["history"]}
    (RESULTS_DIR / "main_results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    write_tables(result, quick)
    plots.performance(noisy)
    plots.robustness(clean_s, noisy)
    plots.training(result["training_history_seed0"])
    if not quick:
        methods = [b[0] for b in NEURAL_BASELINES] + ["Majority vote", "Rule-based (spatial kNN rules)",
                                                      "Ablation: GNN scorer, no paraconsistent layer"] + ALGS
        plots.sota(noisy, methods)
    return result


def write_tables(r: dict, quick: bool):
    d = r["dataset"]
    lines = ["# SolarKG-VN results", "",
             f"Seeds: {r['config']['seeds']} (mean ± std). All metrics on the test split.", "",
             "## Dataset (Tables 1-3)", "",
             f"- clean triples: {d['clean_triples']}, locations: {d['locations']}, relations: {d['relations']}",
             f"- split (clean triples): {d['split']}",
             f"- noisy KG: {d['noisy_triples']} triples, {d['injected']} injected, "
             f"conflict keys per seed: {r['noisy_conflicts_per_seed']} "
             f"({100 * d['noisy_conflict_rate']:.1f}% of clean triples, seed {r['config']['seeds'][0]})",
             f"- injected triples rejected by ontology range/domain axioms: {d['injected_rejected_by_ontology']}",
             "", "## Table 7 - clean KG", "",
             table({k: r["clean"][k] for k in ALGS}, [("correctness", "Correctness"), ("accuracy", "Accuracy"),
                                                     ("support_ratio", "Support ratio")]),
             "## Table 8 - noisy KG", "",
             table({k: r["noisy"][k] for k in ALGS},
                   [("correctness", "Correctness"), ("accuracy", "Accuracy"), ("f1", "F1"),
                    ("support_ratio", "Support ratio"), ("conflict_mitigation", "Conflict mitigation"),
                    ("conflict_resolution_acc", "Conflict resolution acc.")])]
    if not quick:
        abl = [k for k in r["noisy"] if k.startswith("Ablation")]
        lines += ["## Table 9 - ablation (noisy KG)", "",
                  table({k: r["noisy"][k] for k in ALGS + abl},
                        [("accuracy", "Accuracy"), ("correctness", "Correctness"), ("f1", "F1"),
                         ("conflict_resolution_acc", "Conflict resolution acc.")]),
                  "## Table 10 - comparison with baselines (noisy KG, same split/features)", "",
                  "Neural baselines decide each candidate with a threshold tuned on validation; "
                  "they share the encoders/features of the proposed scorer and differ only in the reasoning layer. "
                  "TGformer and RL-based reasoning were not re-implemented.", "",
                  table({k: r["noisy"][k] for k in [b[0] for b in NEURAL_BASELINES] +
                         ["Majority vote", "Rule-based (spatial kNN rules)"] + ALGS},
                        [("accuracy", "Accuracy"), ("f1", "F1"), ("correctness", "Correctness"),
                         ("conflict_resolution_acc", "Conflict resolution acc.")])]
    lines += ["## Wall-clock time per seed (s)", "",
              "| Component | Seconds |\n|---|---|\n" +
              "".join(f"| {k} | {v['mean']:.1f} ± {v['std']:.1f} |\n" for k, v in r["timing_seconds"].items())]
    (RESULTS_DIR / "tables.md").write_text("\n".join(lines), encoding="utf-8")
