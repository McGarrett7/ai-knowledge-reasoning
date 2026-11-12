"""Ontology O (Sec. 3.2), axiom generation G (Sec. 4.3) and the transformation
Phi: O -> T_ont (Sec. 4.4).

Only the axiom forms the reasoner needs are modelled:
  * class assertions          C(a)
  * subclass axioms           C ⊑ D
  * functional roles          Func(r)            (each head has one value)
  * domain / range axioms     dom(r) = C, ran(r) = D or a numeric interval
A triple that violates a domain/range axiom is inconsistent with O on its
own. Two triples (h, r, t1), (h, r, t2) with t1 != t2 and Func(r) are jointly
inconsistent. Every inconsistency is therefore local to one key (h, r);
algorithms/paraconsistent.py relies on this to stay linear in |K|.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from config import REL_NEAR, REL_POTENTIAL, REL_PRODUCES, SOLAR_ENERGY

TYPE = "type"
SUBCLASS = "subClassOf"


def is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def is_location(entity: str) -> bool:
    return isinstance(entity, str) and entity.startswith("location_")


def location_coords(entity: str) -> tuple[int, int]:
    _, x, y = entity.split("_")
    return int(x), int(y)


def normalize_value(value) -> str:
    """Canonical string form so that 255, '255' and 255.0 are one entity."""
    if is_number(value):
        return f"{float(value):.1f}"
    return str(value)


@dataclass
class RangeAxiom:
    kind: str                     # "numeric" or "class"
    lo: float | None = None
    hi: float | None = None
    cls: str | None = None
    conf: float = 1.0


@dataclass
class Ontology:
    individuals: dict = field(default_factory=dict)     # entity -> class
    subclass: list = field(default_factory=list)        # (C, D)
    functional: dict = field(default_factory=dict)      # relation -> confidence
    domains: dict = field(default_factory=dict)         # relation -> class
    ranges: dict = field(default_factory=dict)          # relation -> RangeAxiom

    # ------------------------------------------------------------------ types
    def entity_class(self, entity: str) -> str:
        if entity in self.individuals:
            return self.individuals[entity]
        if is_location(entity):
            return "Location"
        if is_number(entity):
            return "PVValue"
        return "Thing"

    def is_functional(self, relation: str) -> bool:
        return relation in self.functional

    # ------------------------------------------------------------ consistency
    def violation(self, h: str, r: str, t: str) -> str | None:
        """Reason why {(h, r, t)} ∪ O is inconsistent, or None."""
        dom = self.domains.get(r)
        if dom is not None and self.entity_class(h) != dom:
            return f"domain({r})={dom}"
        rng = self.ranges.get(r)
        if rng is None:
            return None
        if rng.kind == "numeric":
            if not is_number(t):
                return f"range({r}) numeric"
            v = float(t)
            if (rng.lo is not None and v < rng.lo) or (rng.hi is not None and v > rng.hi):
                return f"range({r})=[{rng.lo},{rng.hi}]"
            return None
        if self.entity_class(t) != rng.cls:
            return f"range({r})={rng.cls}"
        return None

    def conflicting(self, a: tuple, b: tuple) -> bool:
        return a[0] == b[0] and a[1] == b[1] and a[2] != b[2] and self.is_functional(a[1])

    # ------------------------------------------------------------- Phi (4.4)
    def to_triples(self) -> list[tuple[str, str, str]]:
        triples = [(a, TYPE, c) for a, c in self.individuals.items()]
        triples += [(c, SUBCLASS, d) for c, d in self.subclass]
        return triples

    def describe(self) -> list[str]:
        lines = [f"{c}({a})" for a, c in self.individuals.items()]
        lines += [f"{c} ⊑ {d}" for c, d in self.subclass]
        lines += [f"Func({r})  [conf={c:.3f}]" for r, c in self.functional.items()]
        lines += [f"dom({r}) = {c}" for r, c in self.domains.items()]
        for r, ax in self.ranges.items():
            if ax.kind == "numeric":
                lines.append(f"ran({r}) = [{ax.lo}, {ax.hi}]")
            else:
                lines.append(f"ran({r}) = {ax.cls}")
        return lines

    # ------------------------------------------------------------- G (4.3)
    def extend(self, candidates: "Ontology", tau: float) -> "Ontology":
        """O' = O ∪ Γ_tau: keep candidate axioms whose confidence >= tau."""
        out = Ontology(dict(self.individuals), list(self.subclass), dict(self.functional),
                       dict(self.domains), dict(self.ranges))
        for r, conf in candidates.functional.items():
            if conf >= tau:
                out.functional.setdefault(r, conf)
        for r, ax in candidates.ranges.items():
            if ax.conf >= tau:
                out.ranges.setdefault(r, ax)
        return out


def solar_ontology(pv_range=(0.0, 255.0), with_ranges: bool = True) -> Ontology:
    """Hand-written ontology for SolarKG-VN."""
    onto = Ontology(
        individuals={SOLAR_ENERGY: "EnergySource"},
        subclass=[("EnergySource", "Thing"), ("Location", "Thing"), ("PVValue", "Thing")],
        functional={REL_POTENTIAL: 1.0, REL_PRODUCES: 1.0},
        domains={REL_POTENTIAL: "Location", REL_PRODUCES: "Location", REL_NEAR: "Location"},
    )
    if with_ranges:
        onto.ranges = {
            REL_POTENTIAL: RangeAxiom("numeric", pv_range[0], pv_range[1]),
            REL_PRODUCES: RangeAxiom("class", cls="EnergySource"),
        }
    return onto


def mine_axioms(triples, min_keys: int = 5) -> Ontology:
    """Candidate axioms Γ mined from data (statistical instance of G).

    conf(Func(r)) = fraction of (h, r) keys that have exactly one tail.
    The noise in the training data lowers this confidence, which is why the
    threshold tau (Sec. 4.3) must be below 1.
    """
    tails = defaultdict(set)
    for h, r, t in triples:
        tails[(h, r)].add(t)
    per_rel = defaultdict(lambda: [0, 0])
    for (h, r), ts in tails.items():
        per_rel[r][0] += 1
        per_rel[r][1] += int(len(ts) == 1)
    functional = {r: single / n for r, (n, single) in per_rel.items() if n >= min_keys}
    return Ontology(functional=functional)
