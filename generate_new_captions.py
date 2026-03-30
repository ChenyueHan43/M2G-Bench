import json, os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv('/scratch/ch5085/graphsearch/.env')

client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

with open('/scratch/ch5085/graphsearch/data/new_nodes_to_caption.json') as f:
    new_nodes = json.load(f)

with open('/scratch/ch5085/graphsearch/data/captions_all.json') as f:
    captions = json.load(f)

print(f"Generating captions for {len(new_nodes)} new nodes...")

for node_id, item in tqdm(new_nodes.items()):
    try:
        response = client.chat.completions.create(
            model='qwen-vl-plus',
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': item['image_url'], 'detail': 'low'}},
                    {'type': 'text', 'text': 'Describe this product image in 2-3 sentences. Focus on visual appearance, color, style, material, and product type.'}
                ]
            }],
            max_tokens=150
        )
        captions[node_id] = response.choices[0].message.content
    except Exception as e:
        print(f"  Node {node_id} failed: {e}")

with open('/scratch/ch5085/graphsearch/data/captions_all.json', 'w') as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)

print(f"Total captions: {len(captions)}")
