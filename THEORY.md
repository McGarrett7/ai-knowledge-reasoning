# Formal results behind the implementation

This note covers reviewer comments R2-4 (complexity justification) and R2-5 (rigour of
Theorems 1–3). Section numbers refer to the manuscript. Every statement here is implemented
in `algorithms/paraconsistent.py` and checked by `tests/test_reasoning.py`.

## Setting

$K = O \cup T$. Here $T$ is a finite set of ground triples and $O$ contains only axioms of these forms:

| Axiom | Meaning |
|---|---|
| $C(a)$ | class assertion |
| $C \sqsubseteq D$ | subclass |
| $C \sqcap D \sqsubseteq \bot$ | disjointness, for the base classes `Location`, `PVValue`, `EnergySource` |
| $\mathrm{Func}(r)$ | functional role: $r(a,b) \wedge r(a,c) \rightarrow b = c$ |
| $\mathrm{dom}(r) = C$, $\mathrm{ran}(r) = D$ | domain and range (class or numeric interval $[l,u]$) |

The unique name assumption holds. Every individual has exactly one base class: it is
asserted for named entities and syntactic for locations and numeric literals.

For a triple $\tau = (h,r,t)$, write $\mathrm{viol}(\tau)$ if $\{\tau\} \cup O$ is
inconsistent. For a key $k = (h,r)$, write $\mathrm{cand}(k) = \{t : (h,r,t) \in T, \neg\mathrm{viol}(h,r,t)\}$.

## Proposition 1 (locality of inconsistency)

$S \subseteq T$ is consistent with $O$ **iff**

1. no $\tau \in S$ satisfies $\mathrm{viol}(\tau)$, and
2. there are no $(h,r,t_1), (h,r,t_2) \in S$ with $\mathrm{Func}(r)$ and $t_1 \ne t_2$.

*Proof.*
(⇒) Suppose (1) fails for $\tau = (h,r,t)$. Then $\mathrm{ran}(r)$ forces $t$ into a class
disjoint from its own base class, or into a numeric interval that excludes it, or
$\mathrm{dom}(r)$ does the same for $h$. No interpretation satisfies this, so $S$ is
inconsistent. Suppose (2) fails. Then $\mathrm{Func}(r)$ forces $t_1 = t_2$, which
contradicts the unique name assumption.

(⇐) Build the canonical interpretation $I$:
- the domain is the set of individuals;
- $C^I$ is the set of individuals whose base class is a subclass of $C$;
- $r^I = \{(h,t) : (h,r,t) \in S\}$.

Class and subclass axioms hold by construction. Disjointness holds because base classes
are unique. Domain and range axioms hold by (1). Functionality holds by (2). ∎

**Corollary 1 (structure of $MC(K)$).** A set $F \subseteq T$ is a maximal consistent subset
iff it contains:
- every non-violating triple of every key that is not a conflict key, and
- exactly one triple from $\mathrm{cand}(k)$ for every conflict key $k$, i.e. a key with
  $\mathrm{Func}(r)$ and $|\mathrm{cand}(k)| > 1$.

Hence $|MC(K)| = \prod_k |\mathrm{cand}(k)|$.

*Proof.* By Proposition 1, consistency is checked per key. Maximality forces every triple to be
included whenever adding it keeps the set consistent. For a conflict key, (2) allows at most one
tail and maximality requires at least one. ∎

**Corollary 2 (entailment is local).**

- Credulous entailment ($\exists F \in MC(K): F \models \tau$, Definition 5) holds iff $t \in \mathrm{cand}(h,r)$.
- Skeptical entailment ($\forall F$) holds iff $t \in \mathrm{cand}(h,r)$ and either
  $r$ is not functional or $|\mathrm{cand}(h,r)| = 1$.

The test `test_local_decomposition_equals_brute_force` checks both corollaries against
exhaustive enumeration of $MC(K)$ on 40 random knowledge bases.

## Complexity (replaces the "approximately $O(|K|^2)$" claim in Sec. 6.5)

**Fragment construction.** One pass over $K$ with a hash map keyed by $(h,r)$. Each axiom
check takes $O(1)$ for a fixed $O$, so the pass costs **$O(|K|)$ expected time and space**.
No fragment is ever enumerated.

On the noisy SolarKG-VN, $|MC(K)| \approx 10^{33}$, so exhaustive enumeration is infeasible.
The local decomposition answers all queries in milliseconds: see `results/scalability/scalability.json`,
fields `log10_fragments` and `naive_vs_local`.

**Algorithm 1.** $O(|K| + \sum_{q \in Q} |\mathrm{cand}(q)|)$.

**Algorithm 2.** $O(C_U + |Q|\,|V_r|\,d^2 + \sum_q |\mathrm{cand}(q)|)$, where $V_r$ is the
tail pool of relation $r$ used for prediction.

- With exact full-graph message passing per mini-batch, one training epoch costs
  $O(\frac{|T|}{b}\,(|E|+|V|)\,d^2)$, which is quadratic in $|K|$.
- With neighbour sampling (fanouts $f_1, f_2$), one epoch costs $O(|T|\, f_1 f_2\, d^2)$,
  which is linear in $|K|$.

The reported accuracy uses exact message passing ($|K| = 5.3$k). The scalability study uses
sampling (`TrainConfig.fanout`).

**Algorithm 3.** Agent $i$ trains on its home region plus overlap $\rho$. The total training
volume is $(1+\rho(m-1))\,|T|$, so with sampling the training cost is
$O((1+\rho(m-1))\,|T|\, f_1 f_2 d^2)$. Aggregation costs $O(m \sum_q |\mathrm{cand}(q)|)$.

The cost is linear in $m$ for a fixed $\rho$, and constant in $m$ when $\rho = 0$. Empirical
log–log slopes are reported in `results/scalability/scalability.json`.

## Theorem 1 (soundness of Algorithm 1)

Algorithm 1 returns $t$ for a query $(h,r)$ iff $\exists F \in MC(K)$ with $(h,r,t) \in F$.

*Proof.* It returns $\mathrm{cand}(h,r)$. Apply Corollary 2. ∎

## Theorem 2 (preservation under learning)

Let the learning operator $U$ of Sec. 5 be built exactly as in `learning_operator`:
- it removes triples whose weight $\mu' < \tau$;
- it adds a predicted triple only to keys whose belief set is empty, and only if that triple is
  not violating;
- it changes weights.

Then $U$ creates no new minimal inconsistent subset. Every conclusion of Algorithm 2 is either
- the set of tails of one fragment $F^* = \arg\max_{F \in MC(B')} \sum_{\tau \in F} \mu'(\tau)$
  (a preferred maximal consistent subset), when the margin condition $\delta$ holds, or
- the credulous answer of Theorem 1 on $B'$, otherwise.

So Algorithm 2 is sound with respect to paraconsistent entailment on $B'$.

*Proof.*
- Removing triples maps consistent sets to consistent sets.
- A triple added to an empty key cannot violate condition (2) of Proposition 1, and it is
  checked against condition (1).
- Reweighting does not change consistency.

So $MC(B')$ keeps the product form of Corollary 1. Choosing the maximum-weight element of
each $\mathrm{cand}(k)$ independently maximises the sum over $F$, because the sum decomposes
over keys. ∎

## Theorem 3 (multi-agent soundness)

$K = \bigcup_i B_i$ is again a triple set over the same $O$, so Proposition 1 applies to it.
Algorithm 3 answers each key with one or more tails from $\mathrm{cand}_K(k)$, ranked by the
aggregated weight $\bar\mu$. By the argument of Theorem 2, a committed answer is a fragment of
$MC(K)$ and an uncommitted answer is the credulous answer. Aggregation can add conflicts but
never yields an answer outside some admissible fragment. ∎

## What is new compared with existing techniques (reviewer R2-1)

1. **Learned preference over admissible fragments, with abstention.** Classical preferred
   subtheories use fixed priorities. Here the preference is the learned reliability
   $\mu'$, and the reasoner commits only when the preferred fragment beats the runner-up by
   $\delta$ (tuned on validation). Otherwise it keeps the credulous answer, so an ambiguous
   conflict stays visible instead of being silently resolved.
2. **Locality theorem for KG ontologies.** For functional, domain and range axioms,
   paraconsistent entailment is decided per key in linear time (Proposition 1). This is
   what makes fragment-based semantics usable on knowledge graphs.
3. **Reliability-weighted multi-agent aggregation.** Each agent learns locally. Beliefs are
   combined as $\bar\mu = \sum_i \pi_i \mu'_i / \sum_i \pi_i$, where $\pi_i$ is the agent's
   validation accuracy. Support is counted over agents and predictions.
