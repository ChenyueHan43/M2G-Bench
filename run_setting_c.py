import json, os
import torch
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader_products import load_gs_dataset, ProductsRetriever
from query_planner import parse_search_query, parse_answer

VLLM_HOST = os.getenv("VLLM_HOST", "gh109")
client = OpenAI(api_key="EMPTY", base_url=f"http://{VLLM_HOST}:8000/v1")
MODEL = "/scratch/ch5085/models/Qwen2.5-32B-Instruct"

node_text, neighbors, ppr_neighbors, test_idx, categories = load_gs_dataset(
    "products", gs_dir="/scratch/ch5085/GS_DATASET"
)

# 加载 CLIP embeddings
with open('/scratch/ch5085/graphsearch/data/clip_embeddings.json') as f:
    clip_embeddings = json.load(f)

print(f"Loaded {len(clip_embeddings)} CLIP embeddings")

def cosine_sim_clip(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    return (a @ b / (a.norm() * b.norm())).item()

class CLIPRetriever(ProductsRetriever):
    def __init__(self, node_text, neighbors, ppr_neighbors, clip_embeddings):
        super().__init__(node_text, neighbors, ppr_neighbors)
        self.clip_embeddings = clip_embeddings

    def retrieve(self, anchor_id, query_text, mode="local", hop=1, top_k=3, alpha=0.5, beta=0.3):
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
        anchor_clip = self.clip_embeddings.get(anchor_id)

        scored = []
        for cand_id in candidates:
            cand_id = str(cand_id)
            cand_text = self.get_node_text_by_id(cand_id)
            cand_emb = self._embed(cand_text)

            text_score = alpha * self.cosine_sim(cand_emb, anchor_emb)
            query_score = (1 - alpha - beta) * self.cosine_sim(cand_emb, query_emb)

            # 视觉相似度
            cand_clip = self.clip_embeddings.get(cand_id)
            if anchor_clip and cand_clip:
                visual_score = beta * cosine_sim_clip(anchor_clip, cand_clip)
            else:
                visual_score = 0.0
                text_score = alpha * self.cosine_sim(cand_emb, anchor_emb)
                query_score = (1 - alpha) * self.cosine_sim(cand_emb, query_emb)

            score = text_score + visual_score + query_score
            scored.append((score, cand_id, cand_text))

        scored.sort(reverse=True)
        return [(cid, ctxt) for _, cid, ctxt in scored[:top_k]]

retriever = CLIPRetriever(node_text, neighbors, ppr_neighbors, clip_embeddings)

test_with_clip = [str(nid) for nid in test_idx if str(nid) in clip_embeddings]
print(f"Test nodes with CLIP embeddings: {len(test_with_clip)}")

SYSTEM_PROMPT = """You are a reasoning assistant for node classification on an Amazon product graph.
Your goal is to select the most likely category for the target node from the provided list.

Tools:
- To perform a search, use this schema exactly:
  <search> mode={local|global}, hop={1|2}, query={your query with keywords} </search>
  * mode=local: recall co-purchase neighbors within 1-2 hops
  * mode=global: recall globally relevant products via PageRank
- The graph retriever returns results inside <information>...</information>.

Reasoning protocol:
- Begin with <think>...</think> to assess the product attributes.
- You MUST perform at least one search before giving your final answer. Do NOT skip this step.
- After receiving neighbor information, reason inside <think>...</think>.
- Output ONLY the category name inside <answer>...</answer>, no extra explanation.

Example:
<think>I need to search for co-purchase neighbors to confirm the category.</think>
<search> mode=local, hop=1, query=USB cable electronics accessories </search>
<information>...retriever results...</information>
<think>Neighbors are electronics accessories, so this is Electronics.</think>
<answer>Electronics</answer>"""

def process_node(args):
    i, node_id = args
    node_id = str(node_id)
    info = node_text[node_id]
    true_label = info.get("label", "Unknown")

    user_prompt = f"""Node classification task:
- Title: {info['title']}
- Description: {info['description'][:200]}
- Graph info: degree={info['degree']}, avg_degree={info['avg_degree']:.2f}
- Categories: {'; '.join(categories)}

Remember: you MUST search at least once before answering. Predict the category."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    answer = None
    hop_count = 0

    for step in range(8):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.7, max_tokens=1024
        )
        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})

        answer = parse_answer(assistant_text)
        if answer and hop_count >= 1:
            answer = answer.split("</think>")[-1].strip().split("<")[0].strip()
            if answer:
                break

        query_info = parse_search_query(assistant_text)
        if query_info:
            hop_count += 1
            results = retriever.retrieve(
                anchor_id=node_id,
                query_text=query_info["query"],
                mode=query_info["mode"],
                hop=query_info.get("hop", 1),
                top_k=3, alpha=0.5, beta=0.3
            )
            info_text = "\n\n".join([f"Node {cid}:\n{ctxt}" for cid, ctxt in results]) or "No neighbors found."
            messages.append({"role": "user", "content": f"<information>\n{info_text}\n</information>"})
        else:
            if answer:
                break

    predicted = answer or "N/A"
    is_correct = true_label.lower() in predicted.lower()
    return {"node_id": node_id, "true_label": true_label, "predicted": predicted,
            "correct": is_correct, "hops": hop_count}

print(f"Running Setting C on {len(test_with_clip)} nodes...")
results = []
correct = 0

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_node, (i, nid)): i
               for i, nid in enumerate(test_with_clip)}
    for future in tqdm(as_completed(futures), total=len(test_with_clip)):
        r = future.result()
        results.append(r)
        if r["correct"]:
            correct += 1

sr = correct / len(results) * 100
avg_hops = sum(r["hops"] for r in results) / len(results)
print(f"\n=== Setting C Results ===")
print(f"Success Rate : {sr:.1f}%")
print(f"Avg Hops     : {avg_hops:.2f}")
print(f"Nodes        : {len(results)}")

with open("/scratch/ch5085/graphsearch/results_setting_c_v2.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved → results_setting_c_v2.json")
