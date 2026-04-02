import json, os
import torch
from openai import OpenAI
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader_products import load_gs_dataset, ProductsRetriever
from query_planner import parse_search_query, parse_answer

VLLM_HOST = os.getenv("VLLM_HOST", "gh117")
client = OpenAI(api_key="EMPTY", base_url=f"http://{VLLM_HOST}:8000/v1")
MODEL = "/scratch/ch5085/models/Qwen2.5-32B-Instruct"

# 加载 CLIP 模型
print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

node_text, neighbors, ppr_neighbors, test_idx, categories = load_gs_dataset(
    "products", gs_dir="/scratch/ch5085/GS_DATASET"
)

with open('/scratch/ch5085/graphsearch/data/clip_embeddings.json') as f:
    clip_embeddings = json.load(f)

print(f"Loaded {len(clip_embeddings)} CLIP embeddings")

def get_text_clip_embedding(text):
    """用 CLIP 文字编码器把文字转成向量"""
    inputs = clip_processor(text=[text[:77]], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb[0].tolist()

def cosine_sim_vectors(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    return (a @ b / (a.norm() * b.norm())).item()

class CrossModalRetriever(ProductsRetriever):
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

        # 用 CLIP 文字编码器编码查询
        query_clip_emb = get_text_clip_embedding(query_text)

        scored = []
        for cand_id in candidates:
            cand_id = str(cand_id)
            cand_text = self.get_node_text_by_id(cand_id)
            cand_emb = self._embed(cand_text)
            cand_clip = self.clip_embeddings.get(cand_id)

            text_score = alpha * self.cosine_sim(cand_emb, anchor_emb)

            if cand_clip:
                # 跨模态：文字查询 × 邻居图片
                cross_score = beta * cosine_sim_vectors(query_clip_emb, cand_clip)
                query_score = (1 - alpha - beta) * self.cosine_sim(cand_emb, self._embed(query_text))
                score = text_score + cross_score + query_score
            else:
                # 没有图片，退回纯文字
                score = text_score + (1 - alpha) * self.cosine_sim(cand_emb, self._embed(query_text))

            scored.append((score, cand_id, cand_text))

        scored.sort(reverse=True)
        return [(cid, ctxt) for _, cid, ctxt in scored[:top_k]]

retriever = CrossModalRetriever(node_text, neighbors, ppr_neighbors, clip_embeddings)

# 只用有图片的测试节点
with open('/scratch/ch5085/graphsearch/data/captions_all.json') as f:
    captions = json.load(f)
test_with_images = [str(nid) for nid in test_idx if str(nid) in captions]
print(f"Test nodes: {len(test_with_images)}")

SYSTEM_PROMPT = """You are a reasoning assistant for node classification on an Amazon product graph.
Your goal is to select the most likely category for the target node from the provided list.

Tools:
- To perform a search, use this schema exactly:
  <search> mode={local|global}, hop={1|2}, query={your query with keywords} </search>
- Results returned inside <information>...</information>

Rules:
- Reason inside <think>...</think>
- You MUST perform at least one search before giving your final answer
- Output ONLY the category name inside <answer>...</answer>

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

print(f"Running Setting C v2 (cross-modal) on {len(test_with_images)} nodes...")
results = []
correct = 0

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_node, (i, nid)): i
               for i, nid in enumerate(test_with_images)}
    for future in tqdm(as_completed(futures), total=len(test_with_images)):
        r = future.result()
        results.append(r)
        if r["correct"]:
            correct += 1

sr = correct / len(results) * 100
avg_hops = sum(r["hops"] for r in results) / len(results)
print(f"\n=== Setting C v2 (Cross-Modal) Results ===")
print(f"Success Rate : {sr:.1f}%")
print(f"Avg Hops     : {avg_hops:.2f}")
print(f"Nodes        : {len(results)}")

with open("/scratch/ch5085/graphsearch/results/results_setting_c_crossmodal.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved → results_setting_c_crossmodal.json")
