# Learning-Based Neuro-Symbolic Reasoning for Inconsistent Energy Knowledge Graphs with LLMs

This repository contains the implementation and data for the paper of the same title. The paper covers:

- the SolarKG-VN knowledge graph, built from Global Solar Atlas PV-potential data for Vietnam;
- fragment-based **paraconsistent reasoning** (Algorithm 1);
- a **learned triple-reliability scorer**: R-GCN encoder, bilinear decoder, contrastive training (Algorithm 2);
- **multi-agent** belief aggregation (Algorithm 3);
- **LLM-assisted** triple extraction, plus LLM-based reasoning baselines (Claude or local Qwen/Llama models).

## Repository structure

```
config.py                      all hyperparameters (Sec. 7.4) and paths
main.py                        command-line entry point
kg/
  ontology.py                  ontology O, axiom generation G (Sec. 4.3), Phi: O -> T_ont (Sec. 4.4)
  construction.py              raster -> solar_dataset.csv -> SolarKG-VN.csv
  data.py                      split (Table 2), noise injection (Sec. 7.5), features, message-passing graph
  llm.py                       LLM backends (Anthropic Claude, Ollama), on-disk cache, token/latency accounting
  llm_extraction.py            extraction operator L (Sec. 4.2): rule-based and LLM extractors, P/R/F1
models/kge.py                  encoders (embed / R-GCN / CompGCN), decoders (bilinear, DistMult,
                               ComplEx, TransE, RotatE), contrastive training, neighbour sampling
algorithms/
  paraconsistent.py            MC(K) by local decomposition (Proposition 1) + brute-force check
  algorithm1_paraconsistent.py Algorithm 1
  algorithm2_learning.py       learning operator U and Algorithm 2
  algorithm3_multiagent.py     agent partitioning, reliability-weighted aggregation, Algorithm 3
experiments/
  pipeline.py                  every method on one seed (proposed, ablations, baselines)
  run_experiments.py           Tables 7-10, Figures 1-3, training dynamics
  evaluation_metrics.py        correctness, support ratio, conflict mitigation, accuracy, F1
  llm_baselines.py             zero-shot and KG-RAG LLM baselines, with and without the symbolic layer
  scalability.py               runtime vs. |K|, vs. number of agents, naive vs. local fragments
  benchmarks.py                UMLS / Kinship / Nations (+ WN18RR, FB15k-237): reasoning + link prediction
  plots.py                     figures
prompts/                       every LLM prompt, verbatim
tests/                         unit tests (python -m unittest discover -s tests)
THEORY.md                      Proposition 1, proofs of Theorems 1-3, complexity
results/                       generated tables (tables.md), JSON results, figures
```

## Setup

```bash
pip install -r requirements.txt          # Python >= 3.10; CPU is enough
python -m unittest discover -s tests     # 11 tests
```

## Reproducing the experiments

```bash
python main.py run            # Tables 7-10 + figures, 5 seeds    -> results/tables.md, results/figures/
python main.py run --quick    # Algorithms 1-3 only, seed 0
python main.py scalability    # runtime study                     -> results/scalability.json
python main.py benchmarks     # UMLS, Kinship, Nations (3 seeds)  -> results/benchmark_*.json
python main.py benchmarks --datasets WN18RR FB15k-237   # larger; uses neighbour sampling
```

### LLM components

The LLM components need an Anthropic or DeepSeek API key, or a local Ollama server. The Kaggle notebook `kaggle_run.ipynb` uses DeepSeek.

```bash
# Claude (default model claude-opus-5-5, effort "low", JSON-schema output)
pip install anthropic && export ANTHROPIC_API_KEY=...
python main.py llm --backend anthropic                    # zero-shot and KG-RAG baselines
python main.py extract --backend anthropic --max-docs 500 # LLM extraction quality

# DeepSeek V4.1 Flash (OpenAI-compatible API, temperature 0, JSON mode)
export DEEPSEEK_API_KEY=...                       # never commit the key
export DEEPSEEK_BASE_URL=https://api.deepseek.com # or your gateway's URL
python main.py llm --backend deepseek --model deepseek-flash
python main.py extract --backend deepseek --model deepseek-flash --max-docs 500

# Local open-weight model through Ollama (temperature 0)
ollama pull qwen2.5:7b-instruct
python main.py llm --backend ollama --model qwen2.5:7b-instruct
```

LLM responses are cached under `results/llm_cache/`, so a rerun is free. Each result file
records the model, the number of calls, the token counts, the latency and (for Claude) the cost.

## Data

- `dataset/solar_dataset.csv`: 19,769 points sampled from `PVOUT.tif` (raster row `x`, column `y`, value).
- `dataset/SolarKG-VN.csv`: 5,000 triples over 4,683 locations. Relations are `hasSolarPotential` and `produces`.
- The noisy KG is produced deterministically at run time. For each seed, 300 conflicting
  assertions are injected, which gives about 5.8% conflicting keys (291 for seed 0).
- Split: 3,200 / 800 / 1,000 triples (fixed seed). Each injected triple goes to the split of the triple it contradicts.

> **Note on `PVOUT.tif`.** The bundled file is the Global Solar Atlas *poster map*
> (RGBA, 8-bit, no georeference). Its band 1 is the red colour channel, so the
> sampled `pv_potential` values are colour indices in [0, 255] (81% equal 255),
> not kWh/kWp. The ontology's range axiom uses this [0, 255] scale
> (`DataConfig.pv_range`). For physical units, download the GIS GeoTIFF from
> https://globalsolaratlas.info/download/vietnam, then run
> `python main.py build --raster <PVOUT.tif>` with `pv_range` set to e.g. (500, 2500).
> `main.py build` overwrites both CSV files.

## Metrics (Sec. 7.6)

All metrics are computed on the test-split queries. A query is a key (h, r) together with its asserted candidate tails.

| Metric | Definition |
|---|---|
| Correctness | fraction of answered keys whose accepted tails are jointly consistent with O |
| Accuracy | precision of accepted triples |
| Recall / F1 | true asserted triples recovered |
| Support ratio | mean number of supporting statements per accepted triple (assertions, agent copies, agreeing predictions) |
| Conflict mitigation | 1 − (conflict keys whose answer still holds >1 tail) / (conflict keys) |
| Conflict resolution acc. | conflict keys answered with exactly the true tail |

Thresholds τ and δ, and the decision thresholds of the baselines, are tuned on the validation split only.

## License

See `LICENSE`.
