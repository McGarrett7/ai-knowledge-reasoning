"""Fragment-based paraconsistent semantics over K = O ∪ T (Sec. 3.5, 6.1).

Under the axiom fragment of kg/ontology.py every minimal inconsistent subset
of K is either
  (a) a single triple violating a domain/range axiom, or
  (b) a pair {(h,r,t1), (h,r,t2)} with Func(r) and t1 != t2.
Both are local to a key (h, r). Hence (Proposition 1, THEORY.md)

    MC(K) = { C ∪ {choice(k) : k ∈ conflict keys} }

where C is the conflict-free part and choice(k) picks one consistent tail of
key k. So a query (h, r, t) is credulously entailed (∃F ∈ MC(K), F ⊨ φ) iff t
is a consistent candidate of (h, r), and skeptically entailed (∀F) iff it is
the only one. Building all clusters is one pass over K, O(|K|) expected time,
instead of enumerating the up to ∏_k |cand(k)| fragments.
`naive_maximal_consistent_subsets` enumerates MC(K) by brute force; the
tests use it to check this decomposition on random small knowledge bases.
"""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from dataclasses import dataclass, field

from kg.ontology import Ontology


@dataclass
class Conclusion:
    """Answer of a reasoning algorithm for one query key (h, r)."""
    tails: list                                   # accepted tails (>1 for a functional r = unresolved)
    support: dict = field(default_factory=dict)   # tail -> |S_i|, number of supporting statements
    weight: dict = field(default_factory=dict)    # tail -> belief weight used for the decision


class ParaconsistentKB:
    def __init__(self, triples, ontology: Ontology):
        """`triples`: iterable of (h, r, t). Repeated assertions count as separate evidence."""
        self.ontology = ontology
        self.evidence = defaultdict(lambda: defaultdict(int))   # key -> tail -> #assertions
        self.rejected = defaultdict(int)                         # triple -> #assertions
        for h, r, t in triples:
            if ontology.violation(h, r, t) is None:
                self.evidence[(h, r)][t] += 1
            else:
                self.rejected[(h, r, t)] += 1

    # ---------------------------------------------------------- fragments
    def candidates(self, key) -> dict:
        return dict(self.evidence.get(key, {}))

    def is_conflict(self, key) -> bool:
        return self.ontology.is_functional(key[1]) and len(self.evidence.get(key, ())) > 1

    def fragments(self, key) -> list[frozenset]:
        """Restriction of MC(K) to key (h, r)."""
        tails = list(self.evidence.get(key, {}))
        if not tails:
            return []
        if self.ontology.is_functional(key[1]):
            return [frozenset([t]) for t in tails]
        return [frozenset(tails)]

    def credulous(self, key) -> set:
        """{t : ∃F ∈ MC(K) with (h, r, t) ∈ F}  (Definition 5)."""
        return set().union(*self.fragments(key)) if key in self.evidence else set()

    def skeptical(self, key) -> set:
        frags = self.fragments(key)
        return set(frozenset.intersection(*frags)) if frags else set()

    def entails(self, h, r, t, mode: str = "credulous") -> bool:
        tails = self.credulous((h, r)) if mode == "credulous" else self.skeptical((h, r))
        return t in tails

    def conflict_keys(self) -> set:
        return {k for k in self.evidence if self.is_conflict(k)}

    def log10_num_fragments(self) -> float:
        """log10 |MC(K)|: what exhaustive enumeration would have to visit."""
        return sum(math.log10(len(v)) for k, v in self.evidence.items() if self.is_conflict(k))


def naive_maximal_consistent_subsets(triples, ontology: Ontology) -> list[frozenset]:
    """Brute-force MC(K) over the triple part of K (exponential; small inputs only)."""
    triples = list(dict.fromkeys(triples))
    usable = [t for t in triples if ontology.violation(*t) is None]

    def consistent(subset):
        return not any(ontology.conflicting(a, b) for a, b in itertools.combinations(subset, 2))

    consistent_sets = [frozenset(s) for n in range(len(usable) + 1)
                       for s in itertools.combinations(usable, n) if consistent(s)]
    return [s for s in consistent_sets if not any(s < o for o in consistent_sets)]


def decide(weights: dict, delta: float, functional: bool) -> list:
    """Preferred-fragment selection: keep the highest-weight candidate when it
    beats the runner-up by at least delta; otherwise keep every candidate
    (credulous answer, the conflict stays visible)."""
    if not weights:
        return []
    ranked = sorted(weights, key=weights.get, reverse=True)
    if not functional or len(ranked) == 1:
        return ranked
    if weights[ranked[0]] - weights[ranked[1]] >= delta:
        return ranked[:1]
    return ranked
