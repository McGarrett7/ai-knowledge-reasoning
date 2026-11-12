"""Triple scoring models f_theta(h, r, t) and their contrastive training (Sec. 4.6, 7.4).

A model is an entity *encoder* plus a relation-specific *decoder*:

  encoders  id       - plain Xavier-initialised entity table (classic shallow
                       KGE; no features, no message passing)
            embed    - entity table; entities that carry
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
        if kind == "id":
            self.base = BaseEmbedding(len(ds.entities), cfg.dim, None, torch.zeros(len(ds.entities), dtype=torch.bool))
        else:
            self.base = BaseEmbedding(len(ds.entities), cfg.dim, ds.features, ds.has_features)
        self.graph = graph
        self.dropout = nn.Dropout(0.1)
        if kind in ("embed", "id"):
            self.graph = None
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
        if self.graph is None:
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

    def score_all(self, x: torch.Tensor, r: torch.Tensor, E: torch.Tensor, side: str = "tail") -> torch.Tensor:
        """Logits of (x, r, e) (side="tail") or (e, r, x) (side="head") for every row e of E. Shape (B, N).
        Multiplicative decoders use one matrix product; distance decoders broadcast."""
        if self.kind == "bilinear":
            q = torch.einsum("bd,bde->be", x, self.W[r]) if side == "tail" else torch.einsum("bde,be->bd", self.W[r], x)
            return q @ E.T
        if self.kind == "distmult":
            return (x * self.rel(r)) @ E.T
        if self.kind == "complex":
            xr, xi = x.chunk(2, -1)
            rr, ri = self.rel(r).chunk(2, -1)
            er, ei = E.chunk(2, -1)
            if side == "tail":       # coefficients of t_re / t_im in Re(<h, r, conj t>)
                return (xr * rr - xi * ri) @ er.T + (xi * rr + xr * ri) @ ei.T
            return (rr * xr + ri * xi) @ er.T + (rr * xi - ri * xr) @ ei.T
        n = E.shape[0]
        f = x.unsqueeze(1).expand(-1, n, -1)
        allx = E.unsqueeze(0).expand(x.shape[0], -1, -1)
        return self.forward(f, r, allx) if side == "tail" else self.forward(allx, r, f)


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
        for s in range(0, len(fixed), chunk):
            x = emb[fixed[s:s + chunk].to(emb.device)]
            out.append(self.decoder.score_all(x, r[s:s + chunk].to(emb.device), emb, side).cpu())
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


@torch.no_grad()
def rank_metrics(model: KGModel, ds: KGDataset, triples_df, truth: dict, emb=None) -> dict:
    """Filtered MRR / Hits@{1,3,10} over head and tail prediction (Bordes et al. 2013).
    `truth` maps (h, r) to every known true tail (train, valid and test)."""
    emb = model.embed() if emb is None else emb
    x = ds.encode(triples_df)
    heads_of = {}
    for (h, r), ts in truth.items():
        for t in ts:
            heads_of.setdefault((t, r), set()).add(h)
    ranks = []
    rows = list(triples_df[["head", "relation", "tail"]].itertuples(index=False))
    for side in ("tail", "head"):
        fixed = x[:, 0] if side == "tail" else x[:, 2]
        scores = model._all_scores(fixed, x[:, 1], emb, side, chunk=128)
        for i, (h, r, t) in enumerate(rows):
            target = int(x[i, 2] if side == "tail" else x[i, 0])
            known = truth.get((h, r), ()) if side == "tail" else heads_of.get((t, r), ())
            s = scores[i]
            target_score = s[target].item()
            filt = [ds.ent2id[e] for e in known if e in ds.ent2id and ds.ent2id[e] != target]
            if filt:
                s = s.clone()
                s[filt] = -float("inf")
            ranks.append(1 + int((s > target_score).sum()))
    ranks = np.array(ranks, dtype=float)
    return {"MRR": float(np.mean(1 / ranks)), "Hits@1": float(np.mean(ranks <= 1)),
            "Hits@3": float(np.mean(ranks <= 3)), "Hits@10": float(np.mean(ranks <= 10))}


def _kvsall_queries(pos_all: torch.Tensor):
    """1-N queries: (side, fixed entity, relation, target ids, row positions)."""
    groups = {}
    for i, (h, r, t) in enumerate(pos_all.tolist()):
        for side, x, y in (("tail", h, t), ("head", t, h)):
            g = groups.setdefault((side, x, r), ([], []))
            g[0].append(y)
            g[1].append(i)
    return [(side, x, r, ys, pos) for (side, x, r), (ys, pos) in groups.items()]


def train_model(model: KGModel, ds: KGDataset, train_df, cfg: TrainConfig, seed: int,
                verbose: bool = False) -> dict:
    """Training of f_θ on the triples the model may observe (`train_df`: the
    noisy training split, or an agent's local share of it).

    objective "negsampling" (Sec. 7.4): maximise log σ(f(h,r,t)) + Σ log(1-σ(f(h,r,t'))).
    objective "kvsall": every query (h, r, ?) and (?, r, t) is scored against all
    entities; multi-label BCE with the observed answers as positives.
    For GNN encoders the positives of each batch are removed from message
    passing, so the model cannot read a target edge off the graph.
    Model selection: validation accuracy (one corrupted negative per positive)
    or filtered validation MRR (cfg.select_metric).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    pos_all = ds.encode(train_df)
    edge_ids = None
    if model.encoder.graph is not None:
        # map each training triple to its two edges in the encoder graph
        g = model.encoder.graph
        position = {idx: i for i, idx in enumerate(g.row_index)}
        edge_ids = g.train_edge_of[[position[i] for i in train_df.index]]
    pool = ds.tail_pool("train")
    val_pos, val_neg = validation_set(ds, seed)
    val_df = ds.split("val")
    val_df = val_df[val_df["label"] == 1].drop_duplicates(["head", "relation", "tail"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = (-1.0, None, 0, 0.0)
    history = []
    n_edges = len(model.encoder.graph.src) if edge_ids is not None else 0
    kvsall = cfg.objective == "kvsall"
    sampler = None
    if edge_ids is not None and cfg.fanout and not kvsall:
        sampler = NeighborSampler(model.cpu_graph, len(ds.entities), cfg.fanout)
    queries = _kvsall_queries(pos_all) if kvsall else None
    n_ent = len(ds.entities)
    dev = model.device
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        total = 0.0
        if kvsall:
            perm = torch.randperm(len(queries), generator=gen)
            for s in range(0, len(perm), cfg.batch_size):
                batch = [queries[i] for i in perm[s:s + cfg.batch_size].tolist()]
                mask = None
                if edge_ids is not None:
                    mask = torch.ones(n_edges)
                    rows = torch.tensor([p for q in batch for p in q[4]])
                    mask[edge_ids[rows].reshape(-1)] = 0.0
                emb = model.encoder(mask.to(dev) if mask is not None else None)
                loss = 0.0
                for side in ("tail", "head"):
                    part = [q for q in batch if q[0] == side]
                    if not part:
                        continue
                    x = torch.tensor([q[1] for q in part], device=dev)
                    r = torch.tensor([q[2] for q in part], device=dev)
                    q = F.dropout(emb[x], cfg.emb_dropout, training=True) if cfg.emb_dropout else emb[x]
                    logits = model.decoder.score_all(q, r, emb, side)
                    target = torch.zeros(len(part), n_ent, device=dev)
                    ri = torch.tensor([j for j, q in enumerate(part) for _ in q[3]], device=dev)
                    ci = torch.tensor([y for q in part for y in q[3]], device=dev)
                    target[ri, ci] = 1.0
                    if cfg.label_smoothing:
                        target = target * (1 - cfg.label_smoothing) + cfg.label_smoothing / n_ent
                    loss = loss + F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / n_ent
                loss = loss / len(batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(batch)
            n_items = len(queries)
        else:
            perm = torch.randperm(len(pos_all), generator=gen)
            n_items = len(perm)
        for s in (range(0, len(perm), cfg.batch_size) if not kvsall else ()):
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
        seconds = time.perf_counter() - t0
        if epoch % cfg.eval_every and epoch != cfg.epochs:
            history.append({"epoch": epoch, "loss": total / n_items, "seconds": seconds})
            continue
        emb = model.embed()
        acc = float(((model.prob(val_pos, emb) >= 0.5).float().mean() +
                     (model.prob(val_neg, emb) < 0.5).float().mean()) / 2)
        rec = {"epoch": epoch, "loss": total / n_items, "val_acc": acc, "seconds": seconds}
        score = acc
        if cfg.select_metric == "mrr":
            rec["val_mrr"] = score = rank_metrics(model, ds, val_df, ds.truth, emb)["MRR"]
        history.append(rec)
        if verbose:
            print(f"  [{model.name}] epoch {epoch:03d} loss={rec['loss']:.4f} val_acc={acc:.4f}"
                  + (f" val_mrr={rec['val_mrr']:.4f}" if "val_mrr" in rec else ""), flush=True)
        if score > best[0]:
            best = (score, copy.deepcopy(model.state_dict()), epoch, acc)
        elif epoch - best[2] >= cfg.patience:
            break
    model.load_state_dict(best[1])
    model.eval()
    return {"history": history, "best_epoch": best[2], "best_val_acc": best[3],
            "best_val_score": best[0], "select_metric": cfg.select_metric,
            "train_seconds": float(np.sum([h["seconds"] for h in history]))}
