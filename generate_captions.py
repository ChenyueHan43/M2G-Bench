import json, os, time
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

load_dotenv('/scratch/ch5085/graphsearch/.env')

client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

# 加载图片索引
with open('/scratch/ch5085/graphsearch/data/m2g_image_index.json') as f:
    image_index = json.load(f)

# 只处理测试集里有图片的节点
with open('/scratch/ch5085/GS_DATASET/products_test_ids.txt') as f:
    content = f.read().strip()
    test_ids = set(str(x.strip()) for x in content.split(','))

test_with_images = {k: v for k, v in image_index.items() if k in test_ids}
print(f"Test nodes with images: {len(test_with_images)}")

def generate_caption(node_id, item):
    try:
        response = client.chat.completions.create(
            model='qwen-vl-plus',
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': item['image_url'], 'detail': 'low'}
                    },
                    {
                        'type': 'text',
                        'text': 'Describe this product image in 2-3 sentences. Focus on visual appearance, color, style, material, and product type. Be specific and concise.'
                    }
                ]
            }],
            max_tokens=150
        )
        return node_id, response.choices[0].message.content, None
    except Exception as e:
        return node_id, None, str(e)

# 先跑前10个测试
sample = test_with_images
print(f"\nGenerating captions for {len(sample)} test nodes...")

captions = {}
for node_id, item in tqdm(sample.items()):
    nid, caption, err = generate_caption(node_id, item)
    if caption:
        captions[nid] = caption
        print(f"  Node {nid}: {caption[:80]}...")
    else:
        print(f"  Node {nid}: ERROR - {err}")

print(f"\nGenerated {len(captions)}/{len(sample)} captions successfully")

# 保存结果
with open('/scratch/ch5085/graphsearch/data/captions_all.json', 'w') as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)
print("Saved → captions_all.json")
