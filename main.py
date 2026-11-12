"""Entry point.

    python main.py run            # main SolarKG-VN experiment (Tables 7-10, Figures)
    python main.py run --quick    # Algorithms 1-3 only, one seed
    python main.py scalability    # runtime / scalability study
    python main.py benchmarks     # UMLS, Kinship, Nations, CoDEx-S (+ --datasets WN18RR FB15k-237)
    python main.py llm --backend anthropic            # LLM baselines with Claude
    python main.py llm --backend ollama --model qwen2.5:7b-instruct
    python main.py extract [--backend ...]            # LLM-assisted KG construction + extraction quality
    python main.py build          # rebuild solar_dataset.csv and SolarKG-VN.csv from the raster
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import KG_CSV, RASTER_PATH, RESULTS_DIR, SOLAR_CSV, ExperimentConfig, ensure_dirs  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, nargs="*")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--epochs", type=int)
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--no-summary", action="store_true", help="only write per-seed shards (multi-GPU)")
    sub.add_parser("summarize", help="aggregate results/shards into tables and figures")
    s = sub.add_parser("scalability")
    s.add_argument("--sizes", type=int, nargs="*")
    b = sub.add_parser("benchmarks")
    b.add_argument("--datasets", nargs="*", default=["umls", "kinship", "nations", "codex-s"])
    b.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    for name in ("llm", "extract"):
        q = sub.add_parser(name)
        q.add_argument("--backend", choices=["anthropic", "deepseek", "ollama", "none"], default="none" if name == "extract" else "anthropic")
        q.add_argument("--model")
        q.add_argument("--max-queries", type=int)
        q.add_argument("--max-docs", type=int, default=500)
        q.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("build")
    g.add_argument("--raster", default=str(RASTER_PATH))
    g.add_argument("--samples", type=int, default=20000)
    args = p.parse_args(argv)

    ensure_dirs()
    cfg = ExperimentConfig()

    if args.cmd == "run":
        from experiments.run_experiments import run_main
        if args.epochs:
            cfg.train.epochs = args.epochs
        seeds = args.seeds if args.seeds else ([0] if args.quick else cfg.seeds)
        run_main(cfg, seeds, quick=args.quick, verbose=args.verbose, summarize_after=not args.no_summary)
        if not args.no_summary:
            print((RESULTS_DIR / "tables.md").read_text(encoding="utf-8"))

    elif args.cmd == "summarize":
        from experiments.run_experiments import summarize
        summarize(cfg)
        print((RESULTS_DIR / "tables.md").read_text(encoding="utf-8"))

    elif args.cmd == "scalability":
        from experiments import plots
        from experiments.scalability import run_scalability
        out = run_scalability(cfg, sizes=tuple(args.sizes)) if args.sizes else run_scalability(cfg)
        plots.scalability(out)
        print(json.dumps(out["loglog_slope"], indent=2))

    elif args.cmd == "benchmarks":
        from experiments.benchmarks import run_benchmark
        for name in args.datasets:
            summary = run_benchmark(name, cfg, args.seeds)
            print(name, json.dumps({m: {k: round(v["mean"], 4) for k, v in s.items() if k in ("accuracy", "f1")}
                                    for m, s in summary["reasoning"].items()}, indent=1))

    elif args.cmd == "llm":
        from experiments.llm_baselines import run_llm_baselines
        from experiments.pipeline import Task
        from kg.data import KGDataset, load_triples, prepare_solar
        from kg.llm import make_backend
        from kg.ontology import solar_ontology
        df = prepare_solar(load_triples(KG_CSV), cfg.data, args.seed)
        task = Task(KGDataset(df, knn=cfg.data.knn_neighbors), solar_ontology(cfg.data.pv_range), cfg, args.seed)
        lo, hi = cfg.data.pv_range
        desc = f"an 8-bit PVOUT colour index between {lo:g} and {hi:g}" if hi <= 255 else \
            f"the PVOUT in kWh/kWp per year, physically between {lo:g} and {hi:g}"
        res = run_llm_baselines(task, make_backend(args.backend, args.model, tag="conflict"), desc, args.max_queries)
        print(json.dumps(res, indent=2))

    elif args.cmd == "extract":
        import pandas as pd
        from kg.llm import make_backend
        from kg.llm_extraction import (LLMExtractor, RuleExtractor, evaluate_extraction, gold_triples,
                                       make_documents)
        samples = pd.read_csv(SOLAR_CSV).head(args.max_docs)
        docs = make_documents(samples, seed=args.seed)
        if args.backend == "none":
            extractor = RuleExtractor()
        else:
            extractor = LLMExtractor(make_backend(args.backend, args.model, tag="extraction"))
        pred = extractor.extract(docs)
        metrics = evaluate_extraction(pred, gold_triples(samples))
        if args.backend != "none":
            metrics["backend"] = extractor.backend.describe()
            metrics["usage"] = vars(extractor.backend.usage)
        from kg.llm import safe_name
        out = RESULTS_DIR / f"extraction_{safe_name(extractor.name)}.json"
        out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        pred.to_csv(RESULTS_DIR / "extracted_triples.csv", index=False)
        print(json.dumps(metrics, indent=2))

    elif args.cmd == "build":
        from kg.construction import build_kg, raster_to_samples
        samples = raster_to_samples(args.raster, args.samples, out=SOLAR_CSV)
        build_kg(samples, cfg.data, out=KG_CSV)


if __name__ == "__main__":
    main()
