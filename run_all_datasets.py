import json, random, os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader_products import load_gs_dataset, ProductsRetriever
from query_planner import parse_search_query, parse_answer

load_dotenv()
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

SYSTEM_PROMPT = """You are a classifier. Given a node's text and its neighbors in a graph, 
select the most likely category from the provided list.

Tools:
- <search> mode={local|global}, hop={1|2}, query={keywords} </search>
- Results returned inside <information>...</information>

Rules:
- Reason inside <think>...</think>
- Output ONLY the category name inside <answer>...</answer>
- Must be exactly one category from the list"""

def run_dataset(dataset_name, n_sample=50):
    node_text, neighbors, ppr_neighbors, test_idx, categories = load_gs_dataset(dataset_name)
    retriever = ProductsRetriever(node_text, neighbors, ppr_neighbors)
    sample = random.sample(test_idx, min(n_sample, len(test_idx)))
    
    def process_node(args):
        i, node_id = args
        node_id = str(node_id)
        info = node_text[node_id]
        true_label = info.get("label", "Unknown")
        
        user_prompt = f"""Node classification task:
- Text: Title: {info['title']}
  Description: {info['description'][:300]}
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
                model="deepseek-chat",
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
        return {"node_id": node_id, "true_label": true_label, "predicted": predicted,
                "correct": is_correct, "hops": hop_count}
    
    results = []
    correct = 0
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_node, (i, nid)): i for i, nid in enumerate(sample)}
        for future in tqdm(as_completed(futures), total=len(sample), desc=dataset_name):
            r = future.result()
            results.append(r)
            if r["correct"]:
                correct += 1
    
    sr = correct / len(results) * 100
    avg_hops = sum(r["hops"] for r in results) / len(results)
    print(f"\n{dataset_name}: Success Rate={sr:.1f}% | Avg Hops={avg_hops:.2f}")
    
    with open(f"results_{dataset_name}_50.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return dataset_name, sr, avg_hops

# 跑 PubMed 和 Reddit
all_results = {}
for dataset in ["pubmed", "reddits"]:
    name, sr, hops = run_dataset(dataset, n_sample=50)
    all_results[name] = {"success_rate": sr, "avg_hops": hops}

print("\n=== Summary ===")
print(f"{'Dataset':15} {'SR':>8} {'Hops':>8} {'Paper':>8}")
paper = {"pubmed": 89.8, "reddits": 67.4}
for name, res in all_results.items():
    print(f"{name:15} {res['success_rate']:>7.1f}% {res['avg_hops']:>8.2f} {paper.get(name, '?'):>7}%")