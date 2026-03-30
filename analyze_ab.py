import json
from collections import defaultdict

with open('/scratch/ch5085/graphsearch/data/captions_all.json') as f:
    captions = json.load(f)

with open('/scratch/ch5085/graphsearch/results_setting_a_subset_v3.json') as f:
    a_results = {r['node_id']: r for r in json.load(f)}

with open('/scratch/ch5085/graphsearch/results_setting_b.json') as f:
    b_results = {r['node_id']: r for r in json.load(f)}

b_better = [nid for nid in a_results 
            if not a_results[nid]['correct'] and b_results.get(nid, {}).get('correct')]
a_better = [nid for nid in a_results 
            if a_results[nid]['correct'] and not b_results.get(nid, {}).get('correct')]
both_correct = [nid for nid in a_results 
                if a_results[nid]['correct'] and b_results.get(nid, {}).get('correct')]
both_wrong = [nid for nid in a_results 
              if not a_results[nid]['correct'] and not b_results.get(nid, {}).get('correct')]

print(f"=== Setting A vs B Analysis ===")
print(f"Both correct  : {len(both_correct)}")
print(f"Both wrong    : {len(both_wrong)}")
print(f"Caption helped: {len(b_better)}")
print(f"Caption hurt  : {len(a_better)}")
print()

print("=== Caption helped (B correct, A wrong) ===")
for nid in b_better:
    a = a_results[nid]
    b = b_results[nid]
    cap = captions.get(nid, "")[:80]
    print(f"Node {nid}: true={a['true_label']}")
    print(f"  A predicted: {a['predicted']}")
    print(f"  B predicted: {b['predicted']}")
    print(f"  Caption: {cap}...")
    print()

print("=== Caption hurt (A correct, B wrong) ===")
for nid in a_better:
    a = a_results[nid]
    b = b_results[nid]
    cap = captions.get(nid, "")[:80]
    print(f"Node {nid}: true={a['true_label']}")
    print(f"  A predicted: {a['predicted']}")
    print(f"  B predicted: {b['predicted']}")
    print(f"  Caption: {cap}...")
    print()

# 按类别分析
print("=== Which categories benefit most from captions? ===")
cat_helped = defaultdict(int)
cat_hurt = defaultdict(int)
for nid in b_better:
    cat_helped[a_results[nid]['true_label']] += 1
for nid in a_better:
    cat_hurt[a_results[nid]['true_label']] += 1

print("Helped by category:")
for cat, count in sorted(cat_helped.items(), key=lambda x: -x[1]):
    print(f"  {cat}: +{count}")
print("Hurt by category:")
for cat, count in sorted(cat_hurt.items(), key=lambda x: -x[1]):
    print(f"  {cat}: -{count}")
