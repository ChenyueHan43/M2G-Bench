import json, random, os, re
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader_products import load_gs_dataset, ProductsRetriever

client = OpenAI(api_key="EMPTY", base_url=f"http://{os.getenv('VLLM_HOST', 'gh108')}:8000/v1")
MODEL = "/scratch/ch5085/models/Qwen2.5-32B-Instruct"

def robust_parse_search(text):
    match = re.search(r'<search>(.*?)</search>', text, re.S)
    if match:
        t = match.group(1)
        q = re.search(r'query=(.*?)(?:,|$)', t)
        return {"query": q.group(1).strip() if q else "product"}
    return None

def robust_parse_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.S)
    return match.group(1).strip() if match else None

SYSTEM_PROMPT = """You are a graph-based reasoning assistant. 
1. <think> Analyze the target product. </think>
2. <search> mode=local, hop=1, query=keywords </search>
3. <answer> Category </answer>

Example:
User: Target: "Casio Calculator"
Assistant: <think>Need neighbors.</think> <search>query=Casio</search>
User: <information>Neighbor: Paper</information>
Assistant: <answer>Office Products</answer>"""

def process_node(args, node_text, retriever, categories):
    i, node_id = args
    info = node_text[str(node_id)]
    true_label = info.get("label", "Unknown")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": f"Target: {info['title']}\nCategories: {'; '.join(categories)}"}]
    
    hop_count, final_pred = 0, "N/A"
    for step in range(3):
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.1)
        txt = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": txt})
        s_info = robust_parse_search(txt)
        if s_info and hop_count == 0:
            hop_count += 1
            res = retriever.retrieve(anchor_id=node_id, query_text=s_info['query'], mode='local', hop=1)
            info_msg = "\n".join([f"NB: {c[:100]}" for _, c in res]) or "None"
            messages.append({"role": "user", "content": f"<information>\n{info_msg}\n</information>"})
        else:
            ans = robust_parse_answer(txt)
            if ans: 
                final_pred = ans
                break
    return {"node_id": node_id, "true_label": true_label, "predicted": final_pred, "correct": true_label.lower() in final_pred.lower(), "hops": hop_count}

node_text, neighbors, ppr_neighbors, test_idx, categories = load_gs_dataset("products", gs_dir="/scratch/ch5085/GS_DATASET")
retriever = ProductsRetriever(node_text, neighbors, ppr_neighbors)
sample = random.sample(test_idx, 500) 

results = []
with ThreadPoolExecutor(max_workers=20) as exc:
    futures = [exc.submit(process_node, (i, nid), node_text, retriever, categories) for i, nid in enumerate(sample)]
    for f in tqdm(as_completed(futures), total=500):
        results.append(f.result())

# 存储到独立的文件
with open('/scratch/ch5085/graphsearch/final_success_500.json', 'w') as f:
    json.dump(results, f)

print(f"\nDone! Saved to final_success_500.json")
