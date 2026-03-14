import pandas as pd
from algorithms.algorithm1_paraconsistent import algorithm1_paraconsistent_reasoning
from algorithms.algorithm2_learning import algorithm2_learning_reasoning
from algorithms.algorithm3_multiagent import algorithm3_multiagent_reasoning
from .evaluation_metrics import evaluate_results

def load_kg(file_path):
    kg = pd.read_csv(file_path)
    return kg

def run_experiment():
    print("Loading dataset...")
    kg = load_kg("../dataset/renewable_kg.csv")
    print("Total triples:", len(kg))

    print("\nRunning Algorithm 1...")
    res1 = algorithm1_paraconsistent_reasoning(kg)
    print(res1)

    print("\nRunning Algorithm 2...")
    res2 = algorithm2_learning_reasoning(kg)
    print(res2)

    print("\nRunning Algorithm 3...")
    res3 = algorithm3_multiagent_reasoning(kg)
    print(res3)

    results = evaluate_results(res1, res2, res3)

    print("\nFinal Results:")
    print(results)

    return results

