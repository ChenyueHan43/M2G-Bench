import json, os, random
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader_products import load_gs_dataset, ProductsRetriever
from query_planner import parse_search_query, parse_answer

VLLM_HOST = os.getenv("VLLM_HOST", "gh111")
client = OpenAI(api_key="EMPTY", base_url=f"http://{VLLM_HOST}:8000/v1")
MODEL = "/scratch/ch5085/models/Qwen2.5-32B-Instruct"

node_text, neighbors, ppr_neighbors, test_idx, categories = load_gs_dataset(
    "reddits", gs_dir="/scratch/ch5085/GS_DATASET"
)
retriever = ProductsRetriever(node_text, neighbors, ppr_neighbors)

print(f"Reddit: {len(node_text)} nodes, {len(test_idx)} test nodes, {len(categories)} categories")

SYSTEM_PROMPT = """You are a reasoning assistant for node classification on a Reddit community graph.
Your goal is to select the most likely subreddit category for the target post from the provided list.

Tools:
- <search> mode={local|global}, hop={1|2}, query={your query} </search>
- Results returned inside <information>...</information>

Rules:
- Reason inside <think>...</think>
- You MUST perform at least one search before giving your final answer
- Output ONLY the category name inside <answer>...</answer>

Example:
<think>I need to search for related posts to confirm the category.</think>
<search> mode=local, hop=1, query=cute animals pets photos </search>
<information>...results...</information>
<think>Neighbors are about animals and pets.</think>
<answer>aww</answer>"""

def process_node(args):
    i, node_id = args
    node_id = str(node_id)
    info = node_text[node_id]
    true_label = info.get("label", "Unknown")

    user_prompt = f"""Node classification task:
- Text: Title: {info['title']}
  Description: {info['description'][:200]}
- Graph info: degree={info['degree']}, avg_degree={info['avg_degree']:.2f}
- Categories: {'; '.join(categories)}

Predict the subreddit category."""

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
                anchor_id=node_id, query_text=query_info["query"],
                mode=query_info["mode"], hop=query_info.get("hop", 1),
                top_k=3, alpha=1.0
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

print(f"Running Reddit Setting A on {len(test_idx)} nodes...")
results = []
correct = 0

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_node, (i, nid)): i
               for i, nid in enumerate(test_idx)}
    for future in tqdm(as_completed(futures), total=len(test_idx)):
        r = future.result()
        results.append(r)
        if r["correct"]:
            correct += 1

sr = correct / len(results) * 100
avg_hops = sum(r["hops"] for r in results) / len(results)
print(f"\n=== Reddit Setting A ===")
print(f"Success Rate : {sr:.1f}%  (Paper: 67.4%)")
print(f"Avg Hops     : {avg_hops:.2f}")

with open("/scratch/ch5085/graphsearch/results/results_reddit_a.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved → results_reddit_a.json")
