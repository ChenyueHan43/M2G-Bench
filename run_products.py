import json, random, os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader_products import load_products_gs
from retriever import GraphAwareRetriever
from agent import run_graphsearch

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

print("Loading Products dataset...")
node_text, neighbors, ppr_neighbors, test_idx, categories = load_products_gs()

# 升级 Retriever，支持 PPR global 模式
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

retriever = ProductsRetriever(node_text, neighbors, ppr_neighbors)

# 跑 100 个验证
sample = random.sample(test_idx, 100)

SYSTEM_PROMPT_PRODUCTS = """You are a research assistant for node classification on an Amazon product graph.
Your goal is to select the most likely category for the target product from the provided list.

Tools:
- To perform a search, use this schema exactly:
  <search> mode={local|global}, hop={1|2}, query={your query with keywords} </search>
  • mode=local: recall co-purchase neighbors within 1-2 hops
  • mode=global: recall globally relevant products via PageRank
- The graph retriever returns results inside <information>...</information>.

Reasoning protocol:
- Begin with <think>...</think> to assess if product attributes are sufficient.
- Search for co-purchase neighbors to find related products and infer category.
- Output ONLY the category name inside <answer>...</answer>.
- Must be exactly one category from the list."""

def process_node(args):
    i, node_id = args
    node_id = str(node_id)
    true_label = node_text[node_id].get("label") or "Unknown"
    
    # 如果 node_text 里没有 label，从 corpus 里拿
    info = node_text[node_id]
    degree = info.get("degree", 0)
    avg_degree = info.get("avg_degree", 2.68)
    
    user_prompt = f"""Use the following information for the node classification task:
- The target product's information: Title: {info['title']}
  Description: {info['description']}
- The domain knowledge: Each node represents a product connected through co-purchase relationships. Degree of target node: {degree}, average degree: {avg_degree:.2f}.
- The category list: {'; '.join(categories)}

Please predict the category of the above product."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PRODUCTS},
        {"role": "user", "content": user_prompt}
    ]
    
    from query_planner import parse_search_query, parse_answer
    hop_count = 0
    
    for step in range(8):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )
        assistant_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_text})
        
        answer = parse_answer(assistant_text)
        if answer and hop_count >= 1:
            answer = answer.split("</think>")[-1].strip()
            answer = answer.split("<")[0].strip()
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
                top_k=3,
                alpha=1.0
            )
            info_text = "\n\n".join(
                [f"Node {cid}:\n{ctxt}" for cid, ctxt in results]
            ) or "No relevant neighbors found."
            messages.append({
                "role": "user",
                "content": f"<information>\n{info_text}\n</information>"
            })
        else:
            if answer:
                break
    
    predicted = answer if answer else "N/A"
    is_correct = true_label.lower() in predicted.lower()
    print(f"  [{i+1}/20] true={true_label[:30]} | pred={predicted[:30]} | hops={hop_count}")
    
    return {
        "node_id": node_id,
        "true_label": true_label,
        "predicted": predicted,
        "correct": is_correct,
        "hops": hop_count
    }

print(f"\nRunning GraphSearch on Products (n=20)...")
results = []
correct = 0

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(process_node, (i, node_id)): i
               for i, node_id in enumerate(sample)}
    for future in tqdm(as_completed(futures), total=len(sample)):
        result = future.result()
        results.append(result)
        if result["correct"]:
            correct += 1

sr = correct / len(results) * 100
avg_hops = sum(r["hops"] for r in results) / len(results)
print(f"\n=== Results (Products, n=100) ===")
print(f"Success Rate : {sr:.1f}%  (Paper target: ~71.7%)")
print(f"Avg Hops     : {avg_hops:.2f}")

with open("results_products_100.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved → results_products_20.json")