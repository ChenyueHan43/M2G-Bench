import torch
import json
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

def load_cora(root="./data", corpus_path=None):
    # 加载图结构
    dataset = Planetoid(root=root, name="Cora", transform=NormalizeFeatures())
    data = dataset[0]

    # 加载真实文本
    if corpus_path is None:
        corpus_path = os.path.expanduser("~/Desktop/GS_DATASET/cora_corpus.json")
    
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    # 构建节点文本字典
    node_text = {}
    for idx in range(data.num_nodes):
        entry = corpus.get(str(idx), {})
        text = entry.get("text", f"Paper {idx}")
        # text 格式是 "Title: xxx ; Abstract: xxx"
        # 拆成 title 和 description
        if " ; Abstract:" in text:
            parts = text.split(" ; Abstract:", 1)
            title = parts[0].replace("Title:", "").strip()
            description = parts[1].strip()
        else:
            title = f"Paper {idx}"
            description = text
        
        node_text[str(idx)] = {
            "title": title,
            "description": description
        }

    # 构建邻居字典
    edge_index = data.edge_index.numpy()
    neighbors = {i: [] for i in range(data.num_nodes)}
    for src, dst in zip(edge_index[0], edge_index[1]):
        neighbors[int(src)].append(int(dst))

    # 测试集 ID
    test_idx = data.test_mask.nonzero(as_tuple=True)[0].tolist()

    return data, node_text, neighbors, test_idx, CORA_CATEGORIES

PRODUCTS_CATEGORIES = None  # 从数据集动态读取

def load_products_gs(gs_dir="/scratch/ch5085/GS_DATASET"):
    import json, os
    
    # 加载节点文本
    with open(os.path.join(gs_dir, "products_corpus.json")) as f:
        corpus = json.load(f)
    
    # 加载测试集 ID
    with open(os.path.join(gs_dir, "products_test_ids.txt")) as f:
        test_idx = [int(line.strip()) for line in f if line.strip()]
    
    # 构建节点文本字典
    node_text = {}
    for node_id, entry in corpus.items():
        text = entry.get("text", "")
        if " ; Abstract:" in text:
            parts = text.split(" ; Abstract:", 1)
            title = parts[0].replace("Title:", "").strip()
            description = parts[1].strip()
        elif "Title:" in text:
            title = text.replace("Title:", "").strip()[:200]
            description = text.strip()
        else:
            title = f"Product {node_id}"
            description = text.strip()
        node_text[node_id] = {"title": title, "description": description}
    
    # 收集所有类别
    categories = list(set(
        entry.get("label", entry.get("category", "Unknown"))
        for entry in corpus.values()
        if entry.get("label") or entry.get("category")
    ))
    categories = sorted([c for c in categories if c])
    
    # 构建邻居字典（从 corpus 里的 edge 信息）
    neighbors = {node_id: [] for node_id in corpus.keys()}
    for node_id, entry in corpus.items():
        for nb in entry.get("neighbors", []):
            neighbors[node_id].append(str(nb))
    
    return node_text, neighbors, test_idx, categories