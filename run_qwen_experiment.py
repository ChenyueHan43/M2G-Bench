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
To classify the product, you MUST follow this pattern:
1. <think> Analyze the target. </think>
2. <search> mode=local, hop=1, query=keywords </search>
3. <think> Combine target info with neighbor info. </think>
4. <answer> Category </answer>

Example:
User: Target: "Casio MS-80B Calculator"
Assistant: <think>This is a calculator. I need to check its co-purchase items to confirm.</think>
<search> mode=local, hop=1, query=Casio calculator </search>
User: <information> Neighbor 1: HP Financial Calculator; Neighbor 2: Inkjet Paper </information>
Assistant: <think>The neighbors are office and financial tools. The category is Office Products.</think>
<answer> Office Products </answer>"""

def process_node(args, node_text, retriever, categories):
    i, node_id = args
    info = node_text[str(node_id)]
    true_label = info.get("label", "Unknown")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": f"Target: {info['title']}\nCategories: {'; '.join(categories)}"}]
    
    hop_count, final_pred = 0, "N/A"
    for step in range(3):
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.1) # 低温保证稳定
        txt = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": txt})
        
        s_info = robust_parse_search(txt)
        if s_info and hop_count == 0:
            hop_count += 1
            res = retriever.retrieve(anchor_id=node_id, query_text=s_info['query'], mode='local', hop=1)
            info_msg = "\n".join([f"Neighbor: {c[:150]}" for _, c in res]) or "No neighbors."
            messages.append({"role": "user", "content": f"<information>\n{info_msg}\n</information>"})
        else:
            ans = robust_parse_answer(txt)
            if ans: 
                final_pred = ans
                break
            elif hop_count >= 1: # 如果搜过了但没给标准answer标签，尝试兜底解析
                final_pred = txt.split('\n')[-1]
                break
    return {"correct": true_label.lower() in final_pred.lower(), "hops": hop_count}

node_text, neighbors, ppr_neighbors, test_idx, categories = load_gs_dataset("products", gs_dir="/scratch/ch5085/GS_DATASET")
retriever = ProductsRetriever(node_text, neighbors, ppr_neighbors)
sample = random.sample(test_idx, 500) 

results = []
with ThreadPoolExecutor(max_workers=20) as exc:
    futures = [exc.submit(process_node, (i, nid), node_text, retriever, categories) for i, nid in enumerate(sample)]
    for f in tqdm(as_completed(futures), total=500):
        results.append(f.result())

sr = sum(r["correct"] for r in results) / len(results) * 100
avg_h = sum(r["hops"] for r in results) / len(results)
print(f"\n=== Final Reproduction: SR: {sr:.1f}% | Avg Hops: {avg_h:.2f} ===")

# 保存结果
import json as _json
with open("/scratch/ch5085/graphsearch/results_products_qwen32b_n500.json", "w") as f:
    _json.dump(results, f, indent=2)
print("Saved → results_products_qwen32b_n500.json")
