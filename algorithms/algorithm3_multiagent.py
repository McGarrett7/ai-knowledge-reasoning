from collections import defaultdict
# ============================================
# Algorithm 3
# Multi-Agent Learning-Based Reasoning
# ============================================

def algorithm3_multiagent_reasoning(kg, num_agents=3):

    agents = []

    split = len(kg) // num_agents

    for i in range(num_agents):
        start = i * split
        end = (i + 1) * split
        agents.append(kg.iloc[start:end])

    agent_models = []

    for agent_data in agents:

        beliefs = defaultdict(list)

        for _, row in agent_data.iterrows():
            beliefs[(row["head"], row["relation"])].append(row["tail"])

        agent_models.append(beliefs)

    # knowledge sharing
    global_beliefs = defaultdict(set)

    for model in agent_models:
        for key in model:
            for v in model[key]:
                global_beliefs[key].add(v)

    support_ratio = sum(len(v) for v in global_beliefs.values()) / len(global_beliefs)

    return {
        "agents": num_agents,
        "global_rules": len(global_beliefs),
        "support_ratio": support_ratio
    }
