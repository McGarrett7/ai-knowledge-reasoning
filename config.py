"""Central configuration. Every hyperparameter reported in the paper lives here.

Section references point to the manuscript
"Learning-Based Neuro-Symbolic Reasoning for Inconsistent Energy Knowledge Graphs with LLMs".
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_device() -> str:
    if os.environ.get("KG_DEVICE"):
        return os.environ["KG_DEVICE"]
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MAIN_DIR = RESULTS_DIR / "main"              # SolarKG-VN main experiment (Tables 7-10)
BENCH_RESULTS_DIR = RESULTS_DIR / "benchmarks"
ANALYSIS_DIR = RESULTS_DIR / "analysis"      # robustness / heterogeneous agents
SCALABILITY_DIR = RESULTS_DIR / "scalability"
LLM_DIR = RESULTS_DIR / "llm"                # LLM baselines, extraction, response cache
LOG_DIR = RESULTS_DIR / "logs"
PROMPTS_DIR = ROOT / "prompts"
BENCHMARK_DIR = DATASET_DIR / "benchmarks"

RASTER_PATH = ROOT / "PVOUT.tif"
SOLAR_CSV = DATASET_DIR / "solar_dataset.csv"
KG_CSV = DATASET_DIR / "SolarKG-VN.csv"
KG_NOISY_CSV = DATASET_DIR / "SolarKG-VN_noisy.csv"

REL_POTENTIAL = "hasSolarPotential"
REL_PRODUCES = "produces"
REL_NEAR = "nearTo"            # derived spatial relation (T_ont, Sec. 4.4)
SOLAR_ENERGY = "solar_energy"


@dataclass
class DataConfig:
    raster_samples: int = 20000          # Sec. 7.2 spatial sampling
    kg_size: int = 5000                  # Table 1
    split: tuple = (0.64, 0.16, 0.20)    # Table 2: 3200 / 800 / 1000
    noise_triples: int = 300             # Sec. 7.5 (~5.8% conflicts)
    noise_low: int = 100
    noise_high: int = 300
    # Range axiom for hasSolarPotential. The bundled PVOUT.tif is an 8-bit
    # poster map, so values are 8-bit indices in [0, 255]. For the GIS
    # GeoTIFF (kWh/kWp per year) use e.g. (500.0, 2500.0).
    pv_range: tuple = (0.0, 255.0)
    knn_neighbors: int = 8               # nearTo edges per location


@dataclass
class TrainConfig:
    # Sec. 7.4 "Training Configuration"
    dim: int = 128
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 128
    negatives: int = 5
    patience: int = 10                   # early stopping on validation accuracy
    gnn_layers: int = 2
    num_bases: int = 30                  # R-GCN basis decomposition
    fourier_freqs: int = 8               # spatial / numeric feature encoding
    # Neighbour sampling for GNN training: None = exact full-graph message
    # passing per batch (used for the reported accuracy, |K| = 5k); a tuple
    # such as (10, 5) samples that many in-edges per node and layer, which
    # makes the cost of one epoch linear in |K| (scalability study).
    fanout: tuple | None = None
    # Training objective:
    #   "negsampling" - k tail-corrupted negatives per positive, BCE (Sec. 7.4; SolarKG-VN)
    #   "kvsall"      - 1-N scoring (ConvE-style): every (h, r) and (?, r, t) query is
    #                   scored against all entities with multi-label BCE; standard for
    #                   link prediction on the benchmarks
    objective: str = "negsampling"
    select_metric: str = "acc"           # model selection on validation: "acc" or "mrr"
    eval_every: int = 1                  # epochs between validation evaluations
    label_smoothing: float = 0.0         # kvsall only
    emb_dropout: float = 0.0             # kvsall only: dropout on the query embedding (ConvE-style)
    weight_decay: float = 0.0
    # "cuda" when a GPU is visible (e.g. Kaggle), otherwise "cpu"; override with KG_DEVICE
    device: str = field(default_factory=lambda: _default_device())


@dataclass
class ReasonConfig:
    # Belief update operator U (Sec. 5): mu' = lambda*mu + (1-lambda)*w'
    lam: float = 0.5
    # tau: admission threshold for B^(1); gamma: threshold for T_pred.
    # Decision thresholds on the learned reliability w' (shared by every method):
    # the neural baselines accept a candidate iff w' >= θ; Algorithm 2/3 admit it
    # iff μ' >= τ with μ' = 0.5 + 0.5 w' (prior μ = 1, λ = 0.5), so τ = 0.5 + 0.5 θ
    # gives both the same resolution (fair comparison).
    theta_grid: tuple = (0.01, 0.02, 0.03) + tuple(round(k / 20, 2) for k in range(1, 20))
    tau_grid: tuple = (0.0,) + tuple(round(0.5 + 0.5 * t, 4) for t in
                                     (0.01, 0.02, 0.03) + tuple(round(k / 20, 2) for k in range(1, 20)))
    gamma: float = 0.5
    # delta: minimum weight margin needed to commit to one fragment
    delta_grid: tuple = (0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
    num_agents: int = 3
    overlap: float = 0.3                 # prob. a triple is also seen by another agent


@dataclass
class ExperimentConfig:
    seeds: tuple = (0, 1, 2, 3, 4)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reason: ReasonConfig = field(default_factory=ReasonConfig)


def ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR, BENCHMARK_DIR, MAIN_DIR, MAIN_DIR / "shards",
              BENCH_RESULTS_DIR, BENCH_RESULTS_DIR / "shards", ANALYSIS_DIR,
              SCALABILITY_DIR, LLM_DIR, LLM_DIR / "cache", LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
