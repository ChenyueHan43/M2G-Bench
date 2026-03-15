from sentence_transformers import SentenceTransformer
import numpy as np

class GraphAwareRetriever:
    def __init__(self, node_text, neighbors, encoder_name="all-MiniLM-L6-v2"):
        self.node_text = node_text
        self.neighbors = neighbors
        self.encoder = SentenceTransformer(encoder_name)
        self._embed_cache = {}
    
    def _embed(self, text):
        if text not in self._embed_cache:
            self._embed_cache[text] = self.encoder.encode(
                text, normalize_embeddings=True
            )
        return self._embed_cache[text]
    
    def cosine_sim(self, a, b):
        return float(np.dot(a, b))
    
    def get_local_neighbors(self, anchor_id, hop=1):
        frontier = {anchor_id}
        visited = {anchor_id}
        for _ in range(hop):
            next_frontier = set()
            for node in frontier:
                for nb in self.neighbors.get(node, []):
                    if nb not in visited:
                        next_frontier.add(nb)
                        visited.add(nb)
            frontier = next_frontier
        return list(frontier)
    
    def retrieve(self, anchor_id, query_text, mode="local", hop=1, top_k=3, alpha=1.0):
        if mode == "local":
            candidates = self.get_local_neighbors(anchor_id, hop=hop)
        else:
            import random
            all_nodes = list(self.node_text.keys())
            candidates = random.sample(all_nodes, min(200, len(all_nodes)))
            candidates = [int(c) for c in candidates]
        
        if not candidates:
            return []
        
        anchor_text = self.get_node_text_by_id(anchor_id)
        anchor_emb = self._embed(anchor_text)
        query_emb = self._embed(query_text)
        
        scored = []
        for cand_id in candidates:
            cand_text = self.get_node_text_by_id(cand_id)
            cand_emb = self._embed(cand_text)
            score = (alpha * self.cosine_sim(cand_emb, anchor_emb) +
                     (1 - alpha) * self.cosine_sim(cand_emb, query_emb))
            scored.append((score, cand_id, cand_text))
        
        scored.sort(reverse=True)
        return [(cid, ctxt) for _, cid, ctxt in scored[:top_k]]
    
    def get_node_text_by_id(self, node_id):
        info = self.node_text.get(str(node_id), {})
        return f"Title: {info.get('title', 'N/A')}\nDescription: {info.get('description', 'N/A')}"