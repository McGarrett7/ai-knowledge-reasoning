"""Algorithm 1 - Standard Paraconsistent Reasoning (Sec. 6.1).

Input  : ontology O, triple set T', query keys (h, r, ?)
Output : for each query, the tails entailed by at least one admissible
         fragment F ∈ MC(O ∪ T')  (credulous entailment, Definition 5).

No conflict is resolved: if two consistent tails compete, both are returned.
Range/domain axioms of O still remove triples that are inconsistent on their own.
"""
from __future__ import annotations

from algorithms.paraconsistent import Conclusion, ParaconsistentKB


def algorithm1_paraconsistent_reasoning(kb: ParaconsistentKB, queries) -> dict:
    out = {}
    for key in queries:
        tails = kb.credulous(key)
        if not tails:
            continue
        ev = kb.candidates(key)
        out[key] = Conclusion(tails=sorted(tails), support={t: ev[t] for t in tails},
                              weight={t: 1.0 for t in tails})
    return out
