import json, random, os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
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
print(f"Loaded: {data.num_nodes} nodes, {data.num_edges} edges, {len(test_idx)} test nodes")
print(f"Categories: {categories}")

retriever = GraphAwareRetriever(node_text, neighbors)

# 先只跑 5 个节点，确认整条 pipeline 通了
sample = random.sample(test_idx, 5)

results = []
correct = 0

print("\nRunning GraphSearch on 5 nodes (smoke test)...")
for i, node_id in enumerate(sample):
    true_label_idx = int(data.y[node_id])
    true_label = categories[true_label_idx]
    
    print(f"\n[{i+1}/5] Node {node_id} | True label: {true_label}")
    
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
    if is_correct:
        correct += 1
    
    print(f"  Predicted : {predicted}")
    print(f"  Correct   : {is_correct} | Hops: {result['hops']}")
    
    results.append({
        "node_id": int(node_id),
        "true_label": true_label,
        "predicted": predicted,
        "correct": is_correct,
        "hops": result["hops"]
    })

sr = correct / len(results) * 100
avg_hops = sum(r["hops"] for r in results) / len(results)
print(f"\n=== Smoke Test Results (n=5) ===")
print(f"Success Rate : {sr:.0f}%")
print(f"Avg Hops     : {avg_hops:.1f}")

with open("results_smoke_test.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved → results_smoke_test.json")