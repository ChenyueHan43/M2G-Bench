import json, random, os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader_products import load_gs_dataset, ProductsRetriever
from query_planner import parse_search_query, parse_answer

load_dotenv('/scratch/ch5085/graphsearch/.env')

# 连接 vLLM
VLLM_HOST = os.getenv("VLLM_HOST", "gh105")
client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://{VLLM_HOST}:8000/v1"
)
MODEL = "/scratch/ch5085/models/Qwen2.5-32B-Instruct"

# 加载数据
node_text, neighbors, ppr_neighbors, test_idx, categories = load_gs_dataset(
    "products", gs_dir="/scratch/ch5085/GS_DATASET"
)
retriever = ProductsRetriever(node_text, neighbors, ppr_neighbors)

# 加载 caption
with open('/scratch/ch5085/graphsearch/data/captions_all.json') as f:
    captions = json.load(f)

# 只用有 caption 的测试节点
test_with_caption = [str(nid) for nid in test_idx if str(nid) in captions]
print(f"Test nodes with captions: {len(test_with_caption)}")

SYSTEM_PROMPT = """You are a reasoning assistant for node classification on an Amazon product graph.
Your goal is to select the most likely category for the target node from the provided list.

Tools:
- To perform a search, use this schema exactly:
  <search> mode={local|global}, hop={1|2}, query={your query with keywords} </search>
- Results returned inside <information>...</information>

Rules:
- Reason inside <think>...</think>
- You MUST perform at least one search before giving your final answer
- Output ONLY the category name inside <answer>...</answer>"""

def process_node(args):
    i, node_id = args
    node_id = str(node_id)
    info = node_text[node_id]
    true_label = info.get("label", "Unknown")
    caption = captions.get(node_id, "")

    user_prompt = f"""Node classification task:
- Title: {info['title']}
- Description: {info['description'][:200]}
- Visual appearance: {caption}
- Graph info: degree={info['degree']}, avg_degree={info['avg_degree']:.2f}
- Categories: {'; '.join(categories)}

Predict the category."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    answer = None
    hop_count = 0

    for step in range(8):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024
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
                top_k=3, alpha=1.0
            )
            info_text = "\n\n".join([f"Node {cid}:\n{ctxt}" for cid, ctxt in results]) or "No neighbors found."
            messages.append({"role": "user", "content": f"<information>\n{info_text}\n</information>"})
        else:
            if answer:
                break

    predicted = answer or "N/A"
    is_correct = true_label.lower() in predicted.lower()
    return {
        "node_id": node_id,
        "true_label": true_label,
        "predicted": predicted,
        "correct": is_correct,
        "hops": hop_count,
        "has_caption": True
    }

print(f"Running Setting B on {len(test_with_caption)} nodes with captions...")
results = []
correct = 0

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_node, (i, nid)): i
               for i, nid in enumerate(test_with_caption)}
    for future in tqdm(as_completed(futures), total=len(test_with_caption)):
        r = future.result()
        results.append(r)
        if r["correct"]:
            correct += 1

sr = correct / len(results) * 100
avg_hops = sum(r["hops"] for r in results) / len(results)
print(f"\n=== Setting B Results ===")
print(f"Success Rate : {sr:.1f}%")
print(f"Avg Hops     : {avg_hops:.2f}")
print(f"Nodes        : {len(results)}")

with open("/scratch/ch5085/graphsearch/results_setting_b.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved → results_setting_b.json")
