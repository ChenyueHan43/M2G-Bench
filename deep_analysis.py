import json
from collections import Counter

with open('/scratch/ch5085/graphsearch/results_products_qwen32b.json', 'r') as f:
    results = json.load(f)

errors = [r for r in results if not r['correct']]
corrects = [r for r in results if r['correct']]

print(f"Total Samples: {len(results)}")
print(f"Total Errors: {len(errors)}")

# 1. 分析最常出错的真实类别 (Ground Truth)
error_gt_counts = Counter([r['true_label'] for r in errors])
print("\n--- Top 5 Hardest Categories (True Label) ---")
for cat, count in error_gt_counts.most_common(5):
    print(f"{cat}: {count} errors")

# 2. 分析混淆对 (True -> Predicted)
confusion_pairs = Counter([(r['true_label'], r['predicted']) for r in errors])
print("\n--- Top 5 Common Confusions (True -> Pred) ---")
for (gt, pred), count in confusion_pairs.most_common(5):
    print(f"{gt}  ==>  {pred}: {count} times")

# 3. 检查 Hop 数与准确率的关系
hop_stats = {}
for r in results:
    h = r['hops']
    if h not in hop_stats: hop_stats[h] = {'c': 0, 't': 0}
    hop_stats[h]['t'] += 1
    if r['correct']: hop_stats[h]['c'] += 1

print("\n--- Accuracy by Hop Count ---")
for h, stat in hop_stats.items():
    print(f"Hops {h}: {stat['c']/stat['t']*100:.1f}% accuracy ({stat['t']} samples)")
