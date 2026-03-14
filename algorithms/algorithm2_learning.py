from collections import defaultdict
# ============================================
# Algorithm 2
# Learning-Assisted Reasoning
# ============================================

def algorithm2_learning_reasoning(kg):

    beliefs = defaultdict(list)

    for _, row in kg.iterrows():
        beliefs[(row["head"], row["relation"])].append(row["tail"])

    learned_beliefs = {}

    for key in beliefs:

        values = beliefs[key]

        # simple learning rule: choose most frequent
        freq = defaultdict(int)

        for v in values:
            freq[v] += 1

        best = max(freq, key=freq.get)

        learned_beliefs[key] = best

    accuracy = len(learned_beliefs) / len(beliefs)

    return {
        "learned_rules": len(learned_beliefs),
        "accuracy": accuracy
    }