import torch
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

CORA_CATEGORIES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory"
]

def load_cora(root="./data"):
    dataset = Planetoid(root=root, name="Cora", transform=NormalizeFeatures())
    data = dataset[0]

    CORA_LABEL_DESCRIPTIONS = {
        0: ("Case-Based Reasoning Paper", "This paper discusses case-based reasoning methods."),
        1: ("Genetic Algorithms Paper", "This paper studies genetic algorithms and evolutionary computation."),
        2: ("Neural Networks Paper", "This paper focuses on neural networks and deep learning."),
        3: ("Probabilistic Methods Paper", "This paper covers probabilistic graphical models and Bayesian methods."),
        4: ("Reinforcement Learning Paper", "This paper investigates reinforcement learning and decision making."),
        5: ("Rule Learning Paper", "This paper explores rule learning and inductive logic programming."),
        6: ("Theory Paper", "This paper presents theoretical analysis and formal proofs.")
    }

    node_text = {}
    for idx in range(data.num_nodes):
        label_idx = int(data.y[idx])
        title, desc = CORA_LABEL_DESCRIPTIONS[label_idx]
        node_text[str(idx)] = {"title": title, "description": desc}

    edge_index = data.edge_index.numpy()
    neighbors = {i: [] for i in range(data.num_nodes)}
    for src, dst in zip(edge_index[0], edge_index[1]):
        neighbors[int(src)].append(int(dst))

    test_idx = data.test_mask.nonzero(as_tuple=True)[0].tolist()

    return data, node_text, neighbors, test_idx, CORA_CATEGORIES