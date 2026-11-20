# Additional analyses (SolarKG-VN)

## A. Scorer choice under noise and label scarcity (seeds [0, 1, 2])

### Accuracy

| Method | noise=300 | noise=800 | noise=1600 | train_fraction=0.5 | train_fraction=0.25 |
|---|---|---|---|---|---|
| Algorithm 1 (paraconsistent) | 0.9791 ± 0.0027 | 0.9464 ± 0.0062 | 0.8960 ± 0.0129 | 0.9791 ± 0.0027 | 0.9791 ± 0.0027 |
| Alg. 2 + ID embedding (no features, no GNN) | 0.9889 ± 0.0096 | 0.9649 ± 0.0179 | 0.9296 ± 0.0047 | 0.9830 ± 0.0063 | 0.9813 ± 0.0056 |
| Alg. 2 + feature MLP (no GNN) | 0.9993 ± 0.0005 | 0.9950 ± 0.0042 | 0.9885 ± 0.0040 | 0.9993 ± 0.0005 | 0.9977 ± 0.0019 |
| Alg. 2 + R-GCN (proposed) | 0.9983 ± 0.0005 | 0.9957 ± 0.0034 | 0.9963 ± 0.0012 | 0.9990 ± 0.0000 | 0.9970 ± 0.0029 |
| R-GCN only (no paraconsistent layer) | 0.9916 ± 0.0028 | 0.9895 ± 0.0026 | 0.9898 ± 0.0015 | 0.9810 ± 0.0049 | 0.9769 ± 0.0103 |
| Algorithm 3 (multi-agent) | 0.9990 ± 0.0000 | 0.9960 ± 0.0029 | 0.9960 ± 0.0016 | 0.9990 ± 0.0008 | 0.9970 ± 0.0021 |

### Conflict resolution acc.

| Method | noise=300 | noise=800 | noise=1600 | train_fraction=0.5 | train_fraction=0.25 |
|---|---|---|---|---|---|
| Algorithm 1 (paraconsistent) | 0.6431 ± 0.0312 | 0.6398 ± 0.0521 | 0.6326 ± 0.0291 | 0.6431 ± 0.0312 | 0.6431 ± 0.0312 |
| Alg. 2 + ID embedding (no features, no GNN) | 0.8164 ± 0.1360 | 0.7635 ± 0.1121 | 0.7565 ± 0.0275 | 0.7083 ± 0.0707 | 0.6840 ± 0.0877 |
| Alg. 2 + feature MLP (no GNN) | 0.9891 ± 0.0078 | 0.9640 ± 0.0322 | 0.9638 ± 0.0115 | 0.9891 ± 0.0078 | 0.9674 ± 0.0231 |
| Alg. 2 + R-GCN (proposed) | 0.9696 ± 0.0106 | 0.9714 ± 0.0224 | 0.9865 ± 0.0049 | 0.9819 ± 0.0027 | 0.9537 ± 0.0458 |
| R-GCN only (no paraconsistent layer) | 0.8491 ± 0.0223 | 0.9186 ± 0.0263 | 0.9487 ± 0.0024 | 0.6632 ± 0.0545 | 0.6092 ± 0.1256 |
| Algorithm 3 (multi-agent) | 0.9819 ± 0.0027 | 0.9739 ± 0.0190 | 0.9878 ± 0.0034 | 0.9797 ± 0.0179 | 0.9551 ± 0.0302 |

| Setting | triples | train triples | conflict keys | test heads unseen in train |
|---|---|---|---|---|
| noise=300 | 5300 | 3396 | 112 | 92.1% |
| noise=800 | 5800 | 3713 | 287 | 92.1% |
| noise=1600 | 6600 | 4227 | 547 | 92.1% |
| train_fraction=0.5 | 3602 | 1698 | 59 | 96.1% |
| train_fraction=0.25 | 2753 | 849 | 40 | 98.3% |

## B. Heterogeneous agents (seeds [0, 1, 2])

q = fraction of the unreliable agent's hasSolarPotential triples that receive an in-range corrupted copy (seen only by that agent).

### Accuracy

| Method | q=0.0 | q=0.5 |
|---|---|---|
| Algorithm 1 (paraconsistent) | 0.9791 ± 0.0027 | 0.8782 ± 0.0040 |
| Alg. 2 (one scorer on all data) | 0.9983 ± 0.0005 | 0.9950 ± 0.0016 |
| Alg. 3, equal weights | 0.9987 ± 0.0005 | 0.9957 ± 0.0005 |
| Alg. 3, π = scorer val. accuracy (paper) | 0.9987 ± 0.0005 | 0.9957 ± 0.0005 |
| Alg. 3, π = log-odds of scorer val. accuracy | 0.9987 ± 0.0005 | 0.9957 ± 0.0005 |
| Alg. 3, π = log-odds of source precision on val. | 0.9987 ± 0.0005 | 0.9967 ± 0.0009 |

### Conflict resolution acc.

| Method | q=0.0 | q=0.5 |
|---|---|---|
| Algorithm 1 (paraconsistent) | 0.6431 ± 0.0312 | 0.2085 ± 0.0225 |
| Alg. 2 (one scorer on all data) | 0.9696 ± 0.0106 | 0.9694 ± 0.0117 |
| Alg. 3, equal weights | 0.9768 ± 0.0053 | 0.9740 ± 0.0019 |
| Alg. 3, π = scorer val. accuracy (paper) | 0.9768 ± 0.0053 | 0.9740 ± 0.0019 |
| Alg. 3, π = log-odds of scorer val. accuracy | 0.9768 ± 0.0053 | 0.9740 ± 0.0019 |
| Alg. 3, π = log-odds of source precision on val. | 0.9768 ± 0.0053 | 0.9803 ± 0.0047 |

- q=0.0: per-agent scorer validation accuracy [0.9735, 0.9746, 0.9748], source precision on validation [0.9404, 0.9481, 0.947] (mean over seeds)
- q=0.5: per-agent scorer validation accuracy [0.976, 0.976, 0.9775], source precision on validation [0.7588, 0.9481, 0.947] (mean over seeds)
