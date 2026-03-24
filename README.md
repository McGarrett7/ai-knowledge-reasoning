This repository contains the implementation and experimental datasets used in the paper:
Learning-Based Neuro-Symbolic Reasoning for Inconsistent Energy Knowledge Graphs with LLMs

# Overview
This work addresses knowledge graph reasoning in renewable energy systems under noisy and inconsistent conditions. The proposed framework integrates:
- Knowledge graph construction from real-world solar energy data (GeoTIFF)
- Paraconsistent reasoning for handling logical inconsistency
- Learning-based reasoning for reliability estimation
- LLM-guided semantic scoring for conflict resolution (LLMs are used as a semantic guidance component, as described in the paper)
- Multi-agent reasoning for distributed knowledge aggregation
All components are provided to ensure full reproducibility.

# Repository Structure
- dataset/
    - PVOUT.tif: GeoTIFF raster dataset representing photovoltaic potential
    - solar_dataset.csv
    - SolarKG-VN.csv
- algorithms/
Implementation of  algorithms:
    - algorithm1_paraconsistent.py
    - algorithm2_learning.py
    - algorithm3_multiagent.py
- experiments/
    - Dataset splitting
    - Experiment pipeline and evaluation metrics
- results/
    - Experimental results
- main.py
    - Main script to run experiments
- requirements.txt
    - Required dependencies

#Running Experiments
## Install dependencies
```pip install -r requirements.txt```
## Run experiments
```python main.py```

# Outputs
The system produces:
- Correctness score (Algorithm 1)
- Accuracy (Algorithm 2)
- Support ratio (Algorithm 3)
- Result summaries stored in /results

# Reproducibility
All experiments in the paper

