# SolarKG-VN results

Seeds: [0, 1, 2, 3, 4] (mean ± std). All metrics on the test split.

## Dataset (Tables 1-3)

- clean triples: 5000, locations: 4683, relations: ['hasSolarPotential', 'produces']
- split (clean triples): {'test': 1000, 'train': 3200, 'val': 800}
- noisy KG: 5300 triples, 300 injected, conflict keys per seed: [291, 292, 289, 287, 291] (5.8% of clean triples, seed 0)
- injected triples rejected by ontology range/domain axioms: 190

## Table 7 - clean KG

| Method | Correctness | Accuracy | Support ratio |
|---|---|---|---|
| Algorithm 1 (paraconsistent) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Algorithm 2 (learning-assisted) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.8996 ± 0.0037 |
| Algorithm 3 (multi-agent) | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 4.2902 ± 0.0202 |

## Table 8 - noisy KG

| Method | Correctness | Accuracy | F1 | Support ratio | Conflict mitigation | Conflict resolution acc. |
|---|---|---|---|---|---|---|
| Algorithm 1 (paraconsistent) | 0.9790 ± 0.0027 | 0.9785 ± 0.0023 | 0.9891 ± 0.0012 | 1.0000 ± 0.0000 | 0.6347 ± 0.0263 | 0.6347 ± 0.0263 |
| Algorithm 2 (learning-assisted) | 1.0000 ± 0.0000 | 0.9990 ± 0.0006 | 0.9990 ± 0.0006 | 1.9012 ± 0.0023 | 1.0000 ± 0.0000 | 0.9816 ± 0.0140 |
| Algorithm 3 (multi-agent) | 1.0000 ± 0.0000 | 0.9988 ± 0.0007 | 0.9988 ± 0.0007 | 4.2912 ± 0.0126 | 1.0000 ± 0.0000 | 0.9794 ± 0.0123 |

## Table 9 - ablation (noisy KG)

| Method | Accuracy | Correctness | F1 | Conflict resolution acc. |
|---|---|---|---|---|
| Algorithm 1 (paraconsistent) | 0.9785 ± 0.0023 | 0.9790 ± 0.0027 | 0.9891 ± 0.0012 | 0.6347 ± 0.0263 |
| Algorithm 2 (learning-assisted) | 0.9990 ± 0.0006 | 1.0000 ± 0.0000 | 0.9990 ± 0.0006 | 0.9816 ± 0.0140 |
| Algorithm 3 (multi-agent) | 0.9988 ± 0.0007 | 1.0000 ± 0.0000 | 0.9988 ± 0.0007 | 0.9794 ± 0.0123 |
| Ablation: Alg.1 without range axioms | 0.9436 ± 0.0056 | 0.9424 ± 0.0069 | 0.9710 ± 0.0030 | 0.0000 ± 0.0000 |
| Ablation: GNN scorer, no paraconsistent layer | 0.9915 ± 0.0029 | 0.9919 ± 0.0027 | 0.9881 ± 0.0028 | 0.8499 ± 0.0366 |
| Ablation: Alg.2 with shallow scorer (no GNN) | 0.9996 ± 0.0005 | 1.0000 ± 0.0000 | 0.9996 ± 0.0005 | 0.9935 ± 0.0080 |
| Ablation: Alg.3 with equal agent weights | 0.9988 ± 0.0007 | 1.0000 ± 0.0000 | 0.9988 ± 0.0007 | 0.9794 ± 0.0123 |

## Table 10 - comparison with baselines (noisy KG, same split/features)

Neural baselines decide each candidate with a threshold tuned on validation; they share the encoders/features of the proposed scorer and differ only in the reasoning layer. TGformer and RL-based reasoning were not re-implemented.

| Method | Accuracy | F1 | Correctness | Conflict resolution acc. |
|---|---|---|---|---|
| TransE | 0.9936 ± 0.0034 | 0.9898 ± 0.0027 | 0.9935 ± 0.0035 | 0.8905 ± 0.0590 |
| DistMult | 0.9918 ± 0.0021 | 0.9927 ± 0.0017 | 0.9919 ± 0.0023 | 0.8616 ± 0.0322 |
| ComplEx | 0.9916 ± 0.0048 | 0.9917 ± 0.0017 | 0.9915 ± 0.0049 | 0.8546 ± 0.0709 |
| RotatE | 0.9914 ± 0.0040 | 0.9907 ± 0.0015 | 0.9913 ± 0.0041 | 0.8486 ± 0.0504 |
| R-GCN | 0.9943 ± 0.0022 | 0.9876 ± 0.0024 | 0.9945 ± 0.0021 | 0.8998 ± 0.0365 |
| CompGCN | 0.9910 ± 0.0037 | 0.9881 ± 0.0029 | 0.9911 ± 0.0036 | 0.8420 ± 0.0658 |
| Majority vote | 0.9436 ± 0.0056 | 0.9710 ± 0.0030 | 0.9424 ± 0.0069 | 0.0000 ± 0.0000 |
| Rule-based (spatial kNN rules) | 0.9688 ± 0.0024 | 0.9688 ± 0.0024 | 1.0000 ± 0.0000 | 0.4552 ± 0.0279 |
| Algorithm 1 (paraconsistent) | 0.9785 ± 0.0023 | 0.9891 ± 0.0012 | 0.9790 ± 0.0027 | 0.6347 ± 0.0263 |
| Algorithm 2 (learning-assisted) | 0.9990 ± 0.0006 | 0.9990 ± 0.0006 | 1.0000 ± 0.0000 | 0.9816 ± 0.0140 |
| Algorithm 3 (multi-agent) | 0.9988 ± 0.0007 | 0.9988 ± 0.0007 | 1.0000 ± 0.0000 | 0.9794 ± 0.0123 |

## Wall-clock time per seed (s)

| Component | Seconds |
|---|---|
| Algorithm 1 | 0.0 ± 0.0 |
| Algorithm 2 | 10.1 ± 0.8 |
| Algorithm 3 | 19.0 ± 2.2 |
| TransE | 4.0 ± 0.9 |
| DistMult | 2.5 ± 0.3 |
| ComplEx | 3.4 ± 0.6 |
| RotatE | 7.3 ± 0.4 |
| R-GCN | 8.2 ± 0.9 |
| CompGCN | 43.5 ± 6.4 |
