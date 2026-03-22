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

def load_gs_dataset(dataset_name, gs_dir=None):
    if gs_dir is None:
        gs_dir = os.path.expanduser("~/Desktop/GS_DATASET")
    
    corpus_path = os.path.join(gs_dir, f"{dataset_name}_corpus.json")
    test_ids_path = os.path.join(gs_dir, f"{dataset_name}_test_ids.txt")
    
    print(f"Loading {dataset_name} corpus...")
    with open(corpus_path) as f:
        corpus = json.load(f)
    
    with open(test_ids_path) as f:
        content = f.read().strip()
        test_idx = [int(x.strip()) for x in content.split(',') if x.strip()]
    
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
            "avg_degree": entry.get("dataset_avg_degree", 0),
            "label": entry.get("label", "Unknown")
        }
    
    neighbors = {}
    ppr_neighbors = {}
    for node_id, entry in corpus.items():
        neighbors[str(node_id)] = [str(nb) for nb in entry.get("neighbors", [])]
        ppr_neighbors[str(node_id)] = [str(nb) for nb in entry.get("ppr_neighbors", [])]
    
    categories = sorted(set(
        entry["label"] for entry in corpus.values()
        if entry.get("label")
    ))
    
    print(f"Loaded {len(node_text)} nodes, {len(test_idx)} test nodes, {len(categories)} categories")
    return node_text, neighbors, ppr_neighbors, test_idx, categories

from retriever import GraphAwareRetriever

class ProductsRetriever(GraphAwareRetriever):
    def __init__(self, node_text, neighbors, ppr_neighbors):
        super().__init__(node_text, neighbors)
        self.ppr_neighbors = ppr_neighbors
    
    def retrieve(self, anchor_id, query_text, mode="local", hop=1, top_k=3, alpha=1.0):
        anchor_id = str(anchor_id)
        if mode == "global":
            candidates = self.ppr_neighbors.get(anchor_id, [])
            if not candidates:
                candidates = self.get_local_neighbors(anchor_id, hop=1)
        else:
            candidates = self.get_local_neighbors(anchor_id, hop=hop)
        
        if not candidates:
            return []
        
        anchor_text = self.get_node_text_by_id(anchor_id)
        anchor_emb = self._embed(anchor_text)
        query_emb = self._embed(query_text)
        
        scored = []
        for cand_id in candidates:
            cand_text = self.get_node_text_by_id(str(cand_id))
            cand_emb = self._embed(cand_text)
            score = (alpha * self.cosine_sim(cand_emb, anchor_emb) +
                     (1 - alpha) * self.cosine_sim(cand_emb, query_emb))
            scored.append((score, cand_id, cand_text))
        
        scored.sort(reverse=True)
        return [(str(cid), ctxt) for _, cid, ctxt in scored[:top_k]]
    
    def get_node_text_by_id(self, node_id):
        node_id = str(node_id)
        info = self.node_text.get(node_id, {})
        return f"Title: {info.get('title', 'N/A')}\nDescription: {info.get('description', 'N/A')}"