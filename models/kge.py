"""Triple scoring models f_theta(h, r, t) and their contrastive training (Sec. 4.6, 7.4).

A model is an entity *encoder* plus a relation-specific *decoder*:

  encoders  embed    - Xavier-initialised entity table; entities that carry
                       features (locations: coordinates, values: magnitude)
                       are embedded with an MLP of those features instead,
                       so unseen test locations still get a representation
            rgcn     - R-GCN (Schlichtkrull et al. 2018) on top of `embed`
            compgcn  - CompGCN-style encoder (multiplicative composition,
                       direction-specific weights), simplified re-implementation
  decoders  bilinear (RESCAL-style h^T W_r t), distmult, complex, transe, rotate

The proposed scorer is rgcn + bilinear. The SOTA baselines in Table 10 reuse
the same encoders/decoders, so they differ from the proposed framework only
in the reasoning layer.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainConfig
from kg.data import Graph, KGDataset


# ------------------------------------------------------------------ encoders
class BaseEmbedding(nn.Module):
    def __init__(self, num_entities: int, dim: int, features: torch.Tensor | None, has_features: torch.Tensor):
        super().__init__()
        self.table = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.table.weight)
        self.register_buffer("has_features", has_features.view(-1, 1))
        if features is not None:
            self.register_buffer("features", features)
            self.mlp = nn.Sequential(nn.Linear(features.shape[1], dim), nn.ReLU(), nn.Linear(dim, dim))
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
        else:
            self.features = None

    def forward(self, nodes: torch.Tensor | None = None) -> torch.Tensor:
        if nodes is None:
            x = self.table.weight
            if self.features is not None:
                x = torch.where(self.has_features, self.mlp(self.features), x)
            return x
        x = self.table(nodes)
        if self.features is not None:
            x = torch.where(self.has_features[nodes], self.mlp(self.features[nodes]), x)
        return x


class RGCNLayer(nn.Module):
    def __init__(self, dim: int, num_relations: int, num_bases: int):
        super().__init__()
        self.num_relations = num_relations
        self.self_loop = nn.Linear(dim, dim, bias=False)
        if num_relations > num_bases:
            self.bases = nn.Parameter(torch.empty(num_bases, dim, dim))
            self.coef = nn.Parameter(torch.empty(num_relations, num_bases))
            nn.init.xavier_uniform_(self.coef)
            nn.init.xavier_uniform_(self.bases)
        else:
            self.bases = None
            self.weight = nn.Parameter(torch.empty(num_relations, dim, dim))
            nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.self_loop.weight)

    def rel_weight(self, r: int) -> torch.Tensor:
        if self.bases is None:
            return self.weight[r]
        return torch.einsum("b,bde->de", self.coef[r], self.bases)

    def forward(self, x, g: Graph, groups, mask):
        out = self.self_loop(x)
        for r, idx in groups:
            src, dst = g.src[idx], g.dst[idx]
            m = mask[idx] if mask is not None else torch.ones(len(idx), device=x.device)
            deg = torch.zeros(x.shape[0], device=x.device).index_add_(0, dst, m).clamp(min=1.0)
            msg = (x @ self.rel_weight(r))[src] * (m / deg[dst]).unsqueeze(1)
            out = out.index_add(0, dst, msg)
        return out


class CompGCNLayer(nn.Module):
    def __init__(self, dim: int, num_relations: int, num_base_rel: int):
        super().__init__()
        self.num_base_rel = num_base_rel
        self.w_out = nn.Linear(dim, dim, bias=False)
        self.w_in = nn.Linear(dim, dim, bias=False)
        self.w_self = nn.Linear(dim, dim, bias=False)
        self.w_rel = nn.Linear(dim, dim, bias=False)
        for lin in (self.w_out, self.w_in, self.w_self, self.w_rel):
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, x, z, g: Graph, mask):
        m = mask if mask is not None else torch.ones(len(g.src), device=x.device)
        deg = torch.zeros(x.shape[0], device=x.device).index_add_(0, g.dst, m).clamp(min=1.0)
        comp = x[g.src] * z[g.rel]                                   # phi(x_u, z_r) = x_u * z_r
        inverse = ((g.rel >= self.num_base_rel) & (g.rel < 2 * self.num_base_rel)).unsqueeze(1)
        msg = torch.where(inverse, self.w_in(comp), self.w_out(comp)) * (m / deg[g.dst]).unsqueeze(1)
        out = self.w_self(x).index_add(0, g.dst, msg)
        return out, self.w_rel(z)


class Encoder(nn.Module):
    def __init__(self, kind: str, ds: KGDataset, graph: Graph | None, cfg: TrainConfig):
        super().__init__()
        self.kind = kind
        self.base = BaseEmbedding(len(ds.entities), cfg.dim, ds.features, ds.has_features)
        self.graph = graph
        self.dropout = nn.Dropout(0.1)
        if kind == "embed":
            return
        assert graph is not None, f"{kind} encoder needs a graph"
        self.groups = [(r, (graph.rel == r).nonzero(as_tuple=True)[0]) for r in range(graph.num_relations)]
        self.groups = [(r, idx) for r, idx in self.groups if len(idx)]
        if kind == "rgcn":
            self.layers = nn.ModuleList([RGCNLayer(cfg.dim, graph.num_relations, cfg.num_bases)
                                         for _ in range(cfg.gnn_layers)])
        elif kind == "compgcn":
            self.rel_emb = nn.Embedding(graph.num_relations, cfg.dim)
            nn.init.xavier_uniform_(self.rel_emb.weight)
            self.layers = nn.ModuleList([CompGCNLayer(cfg.dim, graph.num_relations, len(ds.relations))
                                         for _ in range(cfg.gnn_layers)])
        else:
            raise ValueError(kind)

    def forward(self, edge_mask: torch.Tensor | None = None, sub=None) -> torch.Tensor:
        """Full-graph embeddings, or embeddings of `sub = (nodes, local_graph)` only."""
        nodes, graph = (None, self.graph) if sub is None else sub
        x0 = self.base(nodes)
        if self.kind == "embed":
            return x0
        groups = self.groups if sub is None else \
            [(r, idx) for r in graph.rel.unique().tolist() for idx in [(graph.rel == r).nonzero(as_tuple=True)[0]]]
        x = x0
        z = self.rel_emb.weight if self.kind == "compgcn" else None
        for i, layer in enumerate(self.layers):
            if self.kind == "rgcn":
                x = layer(x, graph, groups, edge_mask)
            else:
                x, z = layer(x, z, graph, edge_mask)
            if i < len(self.layers) - 1:
                x = self.dropout(F.relu(x))
        return x + x0                                               # residual keeps entity identity


# ------------------------------------------------------------------ decoders
class Decoder(nn.Module):
    def __init__(self, kind: str, num_relations: int, dim: int):
        super().__init__()
        self.kind = kind
        if kind == "bilinear":
            self.W = nn.Parameter(torch.empty(num_relations, dim, dim))
            nn.init.xavier_uniform_(self.W)
        elif kind in ("distmult", "complex", "transe"):
            self.rel = nn.Embedding(num_relations, dim)
            nn.init.xavier_uniform_(self.rel.weight)
        elif kind == "rotate":
            self.phase = nn.Embedding(num_relations, dim // 2)
            nn.init.uniform_(self.phase.weight, -math.pi, math.pi)
        else:
            raise ValueError(kind)
        if kind in ("transe", "rotate"):
            self.gamma = nn.Parameter(torch.tensor(4.0))
            self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Logits for aligned (h, r, t); h/t may broadcast over a candidate axis."""
        if self.kind == "bilinear":
            if h.dim() == 2:
                return torch.einsum("bd,bde,be->b", h, self.W[r], t)
            return torch.einsum("bnd,bde,bne->bn", h, self.W[r], t)
        if self.kind == "distmult":
            rr = self.rel(r) if h.dim() == 2 else self.rel(r).unsqueeze(1)
            return (h * rr * t).sum(-1)
        if self.kind == "complex":
            rr = self.rel(r) if h.dim() == 2 else self.rel(r).unsqueeze(1)
            hr, hi = h.chunk(2, -1)
            rr_, ri = rr.chunk(2, -1)
            tr, ti = t.chunk(2, -1)
            return (hr * rr_ * tr + hi * rr_ * ti + hr * ri * ti - hi * ri * tr).sum(-1)
        if self.kind == "transe":
            rr = self.rel(r) if h.dim() == 2 else self.rel(r).unsqueeze(1)
            dist = (h + rr - t).norm(p=2, dim=-1)
            return self.scale * (self.gamma - dist)
        # rotate
        ph = self.phase(r) if h.dim() == 2 else self.phase(r).unsqueeze(1)
        hr, hi = h.chunk(2, -1)
        tr, ti = t.chunk(2, -1)
        c, s = torch.cos(ph), torch.sin(ph)
        dr, di = hr * c - hi * s - tr, hr * s + hi * c - ti
        dist = torch.sqrt(dr ** 2 + di ** 2 + 1e-9).sum(-1) / math.sqrt(h.shape[-1])
        return self.scale * (self.gamma - dist)


class KGModel(nn.Module):
    def __init__(self, encoder: str, decoder: str, ds: KGDataset, graph: Graph | None, cfg: TrainConfig):
        super().__init__()
        self.name = f"{encoder}+{decoder}"
        self.device = torch.device(cfg.device)
        self.cpu_graph = graph
        self.encoder = Encoder(encoder, ds, graph.to(self.device) if graph is not None else None, cfg)
        self.decoder = Decoder(decoder, len(ds.relations), cfg.dim)
        self.to(self.device)

    def logits(self, triples: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        triples = triples.to(emb.device)
        return self.decoder(emb[triples[:, 0]], triples[:, 1], emb[triples[:, 2]])

    @torch.no_grad()
    def embed(self) -> torch.Tensor:
        self.eval()
        return self.encoder(None)

    @torch.no_grad()
    def prob(self, triples: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.embed() if emb is None else emb
        return torch.sigmoid(self.logits(triples, emb)).cpu()

    @torch.no_grad()
    def tail_scores(self, h: torch.Tensor, r: torch.Tensor, emb: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        """Logits of (h, r, e) for every entity e. Shape (B, N)."""
        return self._all_scores(h, r, emb, side="tail", chunk=chunk)

    @torch.no_grad()
    def head_scores(self, t: torch.Tensor, r: torch.Tensor, emb: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        return self._all_scores(t, r, emb, side="head", chunk=chunk)

    def _all_scores(self, fixed, r, emb, side, chunk):
        out = []
        n = emb.shape[0]
        for s in range(0, len(fixed), chunk):
            f = emb[fixed[s:s + chunk].to(emb.device)].unsqueeze(1).expand(-1, n, -1)
            allx = emb.unsqueeze(0).expand(f.shape[0], -1, -1)
            rr = r[s:s + chunk].to(emb.device)
            h, t = (f, allx) if side == "tail" else (allx, f)
            out.append(self.decoder(h, rr, t).cpu())
        return torch.cat(out)


# ------------------------------------------------------------------ sampling
class NeighborSampler:
    """Uniform in-edge sampling with a fixed fanout per layer (GraphSAGE-style).
    The cost of one batch is bounded by batch x prod(fanouts), independent of |K|."""

    def __init__(self, g: Graph, num_nodes: int, fanouts):
        self.g, self.num_nodes, self.fanouts = g, num_nodes, tuple(fanouts)
        self.order = torch.argsort(g.dst, stable=True)
        self.deg = torch.bincount(g.dst, minlength=num_nodes)
        self.indptr = torch.zeros(num_nodes + 1, dtype=torch.long)
        self.indptr[1:] = self.deg.cumsum(0)

    def subgraph(self, targets: torch.Tensor, gen: torch.Generator, edge_mask=None):
        g = self.g
        frontier = targets.unique()
        nodes, picked = frontier, []
        for f in self.fanouts:
            v = frontier[self.deg[frontier] > 0]
            if not len(v):
                break
            v = v.repeat_interleave(f)
            off = (torch.rand(len(v), generator=gen) * self.deg[v]).long()
            e = self.order[self.indptr[v] + off].unique()
            if edge_mask is not None:
                e = e[edge_mask[e] > 0]
            picked.append(e)
            frontier = g.src[e].unique()
            nodes = torch.cat([nodes, frontier]).unique()
        e = torch.cat(picked).unique() if picked else torch.empty(0, dtype=torch.long)
        remap = torch.full((self.num_nodes,), -1, dtype=torch.long)
        remap[nodes] = torch.arange(len(nodes))
        local = Graph(remap[g.src[e]], remap[g.dst[e]], g.rel[e], g.num_relations,
                      torch.empty(0, 2, dtype=torch.long), [])
        return nodes, local, remap


# ------------------------------------------------------------------ training
def corrupt_tails(pos: torch.Tensor, k: int, pool: dict, num_entities: int, gen: torch.Generator) -> torch.Tensor:
    """k tail-corrupted negatives per positive, drawn from the relation's tail pool."""
    neg = pos.repeat_interleave(k, dim=0).clone()
    for r in neg[:, 1].unique().tolist():
        sel = (neg[:, 1] == r).nonzero(as_tuple=True)[0]
        cand = pool.get(r)
        if cand is None or len(cand) < 2:
            neg[sel, 2] = torch.randint(0, num_entities, (len(sel),), generator=gen)
        else:
            neg[sel, 2] = cand[torch.randint(0, len(cand), (len(sel),), generator=gen)]
    same = neg[:, 2] == pos.repeat_interleave(k, dim=0)[:, 2]
    neg[same, 2] = torch.randint(0, num_entities, (int(same.sum()),), generator=gen)
    return neg


def validation_set(ds: KGDataset, seed: int):
    """Clean validation positives plus one tail-corrupted negative each."""
    val = ds.split("val")
    val = val[val["label"] == 1].drop_duplicates(["head", "relation", "tail"])
    pos = ds.encode(val)
    gen = torch.Generator().manual_seed(seed + 7)
    neg = corrupt_tails(pos, 1, ds.tail_pool("train"), len(ds.entities), gen)
    return pos, neg


def train_model(model: KGModel, ds: KGDataset, train_df, cfg: TrainConfig, seed: int,
                verbose: bool = False) -> dict:
    """Contrastive training (Sec. 7.4): maximise log σ(f(h,r,t)) + Σ log(1-σ(f(h,r,t'))).

    `train_df` are the triples the model may observe (the noisy training split,
    or an agent's local share of it). For GNN encoders the positives of each
    batch are removed from message passing, so the model cannot read a target
    edge off the graph.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    pos_all = ds.encode(train_df)
    edge_ids = None
    if model.encoder.kind != "embed":
        # map each training triple to its two edges in the encoder graph
        g = model.encoder.graph
        position = {idx: i for i, idx in enumerate(g.row_index)}
        edge_ids = g.train_edge_of[[position[i] for i in train_df.index]]
    pool = ds.tail_pool("train")
    val_pos, val_neg = validation_set(ds, seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best = (-1.0, None, 0)
    history = []
    n_edges = len(model.encoder.graph.src) if edge_ids is not None else 0
    sampler = None
    if edge_ids is not None and cfg.fanout:
        sampler = NeighborSampler(model.cpu_graph, len(ds.entities), cfg.fanout)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(pos_all), generator=gen)
        total = 0.0
        for s in range(0, len(perm), cfg.batch_size):
            b = perm[s:s + cfg.batch_size]
            pos = pos_all[b]
            neg = corrupt_tails(pos, cfg.negatives, pool, len(ds.entities), gen)
            mask = None
            if edge_ids is not None:
                mask = torch.ones(n_edges)
                mask[edge_ids[b].reshape(-1)] = 0.0
            dev = model.device
            if sampler is None:
                emb = model.encoder(mask.to(dev) if mask is not None else None)
                lp, ln = model.logits(pos, emb), model.logits(neg, emb)
            else:
                targets = torch.cat([pos[:, 0], pos[:, 2], neg[:, 2]])
                nodes, local, remap = sampler.subgraph(targets, gen, mask)
                emb = model.encoder(None, sub=(nodes.to(dev), local.to(dev)))
                pl, nl = remap[pos].to(dev), remap[neg].to(dev)
                lp = model.decoder(emb[pl[:, 0]], pos[:, 1].to(dev), emb[pl[:, 2]])
                ln = model.decoder(emb[nl[:, 0]], neg[:, 1].to(dev), emb[nl[:, 2]])
            loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + \
                F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        emb = model.embed()
        acc = float(((model.prob(val_pos, emb) >= 0.5).float().mean() +
                     (model.prob(val_neg, emb) < 0.5).float().mean()) / 2)
        history.append({"epoch": epoch, "loss": total / len(perm), "val_acc": acc,
                        "seconds": time.perf_counter() - t0})
        if verbose:
            print(f"  [{model.name}] epoch {epoch:02d} loss={total / len(perm):.4f} val_acc={acc:.4f}")
        if acc > best[0]:
            best = (acc, copy.deepcopy(model.state_dict()), epoch)
        elif epoch - best[2] >= cfg.patience:
            break
    model.load_state_dict(best[1])
    model.eval()
    return {"history": history, "best_epoch": best[2], "best_val_acc": best[0],
            "train_seconds": float(np.sum([h["seconds"] for h in history]))}
