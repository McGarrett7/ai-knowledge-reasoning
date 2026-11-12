"""Knowledge-graph data handling: loading, the train/val/test split, noise
injection (Sec. 7.5), the vocabulary, entity features, and the message-passing
graph used by the GNN encoders.

All randomness takes an explicit seed so that every number can be reproduced.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from config import REL_NEAR, DataConfig
from kg.ontology import is_location, is_number, location_coords, normalize_value

COLUMNS = ["head", "relation", "tail"]


# --------------------------------------------------------------------- I/O
def load_triples(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)[COLUMNS]
    df["tail"] = df["tail"].map(normalize_value)
    return df.reset_index(drop=True)


def count_conflicts(df: pd.DataFrame, functional=None) -> int:
    """Number of (h, r) keys with more than one distinct tail."""
    sub = df if functional is None else df[df["relation"].isin(functional)]
    n_tails = sub.groupby(["head", "relation"])["tail"].nunique()
    return int((n_tails > 1).sum())


# ------------------------------------------------------------- split/noise
def assign_split(df: pd.DataFrame, fractions=(0.64, 0.16, 0.20), seed: int = 42) -> pd.DataFrame:
    """Disjoint split performed after KG construction (Sec. 7.3)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    split = np.empty(n, dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train:n_train + n_val]] = "val"
    split[perm[n_train + n_val:]] = "test"
    out = df.copy()
    out["split"] = split
    return out


def inject_noise(df: pd.DataFrame, n: int, low: int, high: int, seed: int) -> pd.DataFrame:
    """Add n conflicting assertions (h, r, v2) for randomly chosen (h, r, v1).

    Mirrors the original protocol (random PV value in [low, high]) but is
    seeded. Each noisy row keeps the split of the row it was derived from, so
    a conflict never straddles train and test. `label` is 1 when the asserted
    tail equals the ground-truth tail.
    """
    rng = np.random.default_rng(seed)
    clean = df.copy()
    clean["label"] = 1
    clean["source"] = "sensor"
    idx = rng.integers(0, len(df), size=n)
    vals = rng.integers(low, high + 1, size=n)
    noise = df.iloc[idx].copy()
    noise["tail"] = [normalize_value(v) for v in vals]
    noise["label"] = (noise["tail"].values == df.iloc[idx]["tail"].values).astype(int)
    noise["source"] = "noise"
    return pd.concat([clean, noise], ignore_index=True)


def prepare_solar(df_clean: pd.DataFrame, cfg: DataConfig, seed: int, noisy: bool = True) -> pd.DataFrame:
    split_df = assign_split(df_clean, cfg.split, seed=42)       # split fixed as in the paper
    if not noisy:
        out = split_df.copy()
        out["label"] = 1
        out["source"] = "sensor"
        return out
    return inject_noise(split_df, cfg.noise_triples, cfg.noise_low, cfg.noise_high, seed)


# ------------------------------------------------------------------ dataset
def fourier(x: np.ndarray, freqs: int) -> np.ndarray:
    """[sin(2^k π x), cos(2^k π x)] for k < freqs, x in [0, 1]. Shape (n, 2*freqs*d)."""
    x = np.atleast_2d(x.T).T
    bands = (2.0 ** np.arange(freqs)) * math.pi
    ang = x[:, :, None] * bands[None, None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=2).reshape(len(x), -1)


@dataclass
class Graph:
    """Typed edge list for message passing (train triples, inverses, nearTo)."""
    src: torch.Tensor
    dst: torch.Tensor
    rel: torch.Tensor
    num_relations: int          # including inverse / derived relations
    train_edge_of: torch.Tensor  # i-th source triple -> its two edge ids (2*i, 2*i+1)
    row_index: list             # dataframe index of the source triples

    def to(self, device) -> "Graph":
        return Graph(self.src.to(device), self.dst.to(device), self.rel.to(device), self.num_relations,
                     self.train_edge_of, self.row_index)


class KGDataset:
    """Vocabulary, ground truth and tensors for one (possibly noisy) KG."""

    def __init__(self, df: pd.DataFrame, freqs: int = 8, knn: int = 0,
                 value_scale: float | None = None, truth: dict | None = None):
        self.df = df.reset_index(drop=True)
        ents = pd.unique(pd.concat([self.df["head"], self.df["tail"]]))
        self.entities = list(ents)
        self.ent2id = {e: i for i, e in enumerate(self.entities)}
        self.relations = sorted(self.df["relation"].unique())
        self.rel2id = {r: i for i, r in enumerate(self.relations)}
        self.knn = knn
        if truth is None:
            truth = {}
            for h, r, t, lab in self.df[["head", "relation", "tail", "label"]].itertuples(index=False):
                if lab == 1:
                    truth.setdefault((h, r), set()).add(t)
        self.truth = truth                                    # key -> set of true tails
        self.features = self._features(freqs, value_scale)

    # ---------------------------------------------------------------- views
    def split(self, name: str) -> pd.DataFrame:
        return self.df[self.df["split"] == name]

    def encode(self, df: pd.DataFrame) -> torch.Tensor:
        h = df["head"].map(self.ent2id).values
        r = df["relation"].map(self.rel2id).values
        t = df["tail"].map(self.ent2id).values
        return torch.tensor(np.stack([h, r, t], axis=1), dtype=torch.long)

    def candidates(self, split: str) -> dict:
        """key (h, r) -> {tail: number of assertions} for one split."""
        out = defaultdict(lambda: defaultdict(int))
        for h, r, t in self.split(split)[["head", "relation", "tail"]].itertuples(index=False):
            out[(h, r)][t] += 1
        return {k: dict(v) for k, v in out.items()}

    def tail_pool(self, split: str = "train") -> dict:
        """relation id -> tensor of tail ids observed with it (type-consistent negatives)."""
        sub = self.split(split)
        pool = {}
        for r, grp in sub.groupby("relation"):
            pool[self.rel2id[r]] = torch.tensor(grp["tail"].map(self.ent2id).unique(), dtype=torch.long)
        return pool

    # ------------------------------------------------------------- features
    def _features(self, freqs: int, value_scale):
        locs = [e for e in self.entities if is_location(e)]
        nums = [e for e in self.entities if is_number(e)]
        if not locs and not nums:
            self.has_features = torch.zeros(len(self.entities), dtype=torch.bool)
            return None
        n = len(self.entities)
        coords = np.zeros((n, 2))
        val = np.zeros((n, 1))
        flags = np.zeros((n, 3))
        if locs:
            xy = np.array([location_coords(e) for e in locs], dtype=float)
            self.coord_max = xy.max(axis=0) + 1.0
        if value_scale is None:
            value_scale = max((abs(float(e)) for e in nums), default=1.0) or 1.0
        self.value_scale = value_scale
        for i, e in enumerate(self.entities):
            if is_location(e):
                coords[i] = np.array(location_coords(e)) / self.coord_max
                flags[i, 0] = 1
            elif is_number(e):
                val[i, 0] = float(e) / value_scale
                flags[i, 1] = 1
            else:
                flags[i, 2] = 1
        feat = np.concatenate([coords, val, fourier(coords, freqs) * flags[:, :1],
                               fourier(val, freqs) * flags[:, 1:2], flags], axis=1)
        # Entities with features are encoded inductively (no free embedding),
        # so unseen test locations still get a meaningful representation.
        self.has_features = torch.tensor(flags[:, 0] + flags[:, 1] > 0)
        return torch.tensor(feat, dtype=torch.float32)

    # --------------------------------------------------------------- graph
    def build_graph(self, triples: pd.DataFrame | None = None) -> Graph:
        """Message-passing graph from `triples` (default: the training split)."""
        triples = self.split("train") if triples is None else triples
        tr = self.encode(triples)
        n_rel = len(self.relations)
        src = torch.stack([tr[:, 0], tr[:, 2]], dim=1).reshape(-1)
        dst = torch.stack([tr[:, 2], tr[:, 0]], dim=1).reshape(-1)
        rel = torch.stack([tr[:, 1], tr[:, 1] + n_rel], dim=1).reshape(-1)
        num_rel = 2 * n_rel
        if self.knn > 0:
            s, d = self.knn_edges(self.knn)
            if len(s):
                src = torch.cat([src, s])
                dst = torch.cat([dst, d])
                rel = torch.cat([rel, torch.full_like(s, num_rel)])
                num_rel += 1
        edge_of = torch.arange(2 * len(tr)).reshape(-1, 2)
        return Graph(src, dst, rel, num_rel, edge_of, list(triples.index))

    def knn_edges(self, k: int):
        """nearTo(u, v): v among the k nearest locations of u (Phi-derived, w = 1)."""
        from scipy.spatial import cKDTree

        ids = [i for i, e in enumerate(self.entities) if is_location(e)]
        if len(ids) <= 1:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        xy = np.array([location_coords(self.entities[i]) for i in ids], dtype=float)
        k_eff = min(k, len(ids) - 1)
        _, nn = cKDTree(xy).query(xy, k=k_eff + 1)
        ids = np.array(ids)
        src = ids[nn[:, 1:]].reshape(-1)          # neighbour sends a message ...
        dst = np.repeat(ids, k_eff)               # ... to the location
        return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)

    def near_triples(self):
        s, d = self.knn_edges(self.knn) if self.knn else ([], [])
        return [(self.entities[int(a)], REL_NEAR, self.entities[int(b)]) for a, b in zip(s, d)]
