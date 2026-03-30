import json

with open('/scratch/ch5085/graphsearch/results_products_qwen32b.json', 'r') as f:
    data = json.load(f)

total = len(data)
correct = sum(1 for r in data if r['correct'])
errors = [r for r in data if not r['correct']]

# 统计跳数分布
hop_dist = {}
for r in data:
    h = r['hops']
    hop_dist[h] = hop_dist.get(h, 0) + 1

print(f"--- Analysis Report (n={total}) ---")
print(f"Success Rate: {correct/total*100:.1f}%")
print(f"Hop Distribution: {hop_dist}")

print("\n--- Sample Errors (Top 3) ---")
for i, err in enumerate(errors[:3]):
    print(f"{i+1}. Node: {err['node_id']} | True: {err['true_label']} | Pred: {err['predicted']} | Hops: {err['hops']}")

