from collections import defaultdict
# ============================================
# Algorithm 1
# Standard Paraconsistent Reasoning
# ============================================

def algorithm1_paraconsistent_reasoning(kg):

    beliefs = defaultdict(set)

    for _, row in kg.iterrows():
        h = row["head"]
        r = row["relation"]
        t = row["tail"]

        beliefs[(h, r)].add(t)

    # check conflicts
    conflicts = 0

    for key in beliefs:
        if len(beliefs[key]) > 1:
            conflicts += 1

    total = len(beliefs)

    correctness = (total - conflicts) / total

    return {
        "correctness": correctness,
        "conflicts": conflicts,
        "total_rules": total
    }