import json
import os

def load_products_gs(gs_dir=None):
    if gs_dir is None:
        gs_dir = os.path.expanduser("~/Desktop/GS_DATASET")
    
    corpus_path = os.path.join(gs_dir, "products_corpus.json")
    test_ids_path = os.path.join(gs_dir, "products_test_ids.txt")
    
    print("Loading products corpus...")
    with open(corpus_path) as f:
        corpus = json.load(f)
    
    # 加载测试集 ID
    with open(test_ids_path) as f:
        content = f.read().strip()
        test_idx = [int(x.strip()) for x in content.split(',') if x.strip()]
    
    # 构建节点文本字典
    node_text = {}
    for node_id, entry in corpus.items():
        text = entry.get("text", "")
        if "; Description:" in text:
            parts = text.split("; Description:", 1)
            title = parts[0].replace("Title:", "").strip()
            description = parts[1].strip()
        else:
            title = text[:100].replace("Title:", "").strip()
            description = text.strip()
        node_text[str(node_id)] = {
            "title": title,
            "description": description,
            "degree": entry.get("degree", 0),
            "avg_degree": entry.get("dataset_avg_degree", 2.68),
            "label": entry.get("label", "Unknown")
        }
    
    # 构建邻居字典（local neighbors）
    neighbors = {}
    for node_id, entry in corpus.items():
        neighbors[str(node_id)] = [str(nb) for nb in entry.get("neighbors", [])]
    
    # 构建 PPR 邻居字典（global neighbors）
    ppr_neighbors = {}
    for node_id, entry in corpus.items():
        ppr_neighbors[str(node_id)] = [str(nb) for nb in entry.get("ppr_neighbors", [])]
    
    # 收集类别
    categories = sorted(set(
        entry["label"] for entry in corpus.values()
        if entry.get("label")
    ))
    
    print(f"Loaded {len(node_text)} nodes, {len(test_idx)} test nodes, {len(categories)} categories")
    return node_text, neighbors, ppr_neighbors, test_idx, categories