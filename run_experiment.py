import json, random, os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader import load_cora
from retriever import GraphAwareRetriever
from agent import run_graphsearch

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

print("Loading Cora dataset...")
data, node_text, neighbors, test_idx, categories = load_cora()
print(f"Loaded: {data.num_nodes} nodes, {len(test_idx)} test nodes")

retriever = GraphAwareRetriever(node_text, neighbors)
sample = random.sample(test_idx, 50)

def process_node(args):
    i, node_id = args
    true_label_idx = int(data.y[node_id])
    true_label = categories[true_label_idx]
    result = run_graphsearch(
        anchor_id=node_id,
        retriever=retriever,
        categories=categories,
        client=client,
        model="deepseek-chat",
        alpha=1.0
    )
    predicted = result["answer"] or "N/A"
    is_correct = true_label.lower() in predicted.lower()
    print(f"  [{i+1}/50] true={true_label} | pred={predicted} | hops={result['hops']}")
    return {
        "node_id": int(node_id),
        "true_label": true_label,
        "predicted": predicted,
        "correct": is_correct,
        "hops": result["hops"]
    }

print("\nRunning GraphSearch (50 nodes, 5 workers)...")
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
print(f"\n=== Results (n=50) ===")
print(f"Success Rate : {sr:.1f}%")
print(f"Avg Hops     : {avg_hops:.2f}")

with open("results_cora_50.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved → results_cora_50.json")