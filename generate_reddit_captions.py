import json, os, base64
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv('/scratch/ch5085/graphsearch/.env')

client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

IMAGE_DIR = '/scratch/jl11523/dataset-mgllm/Reddit_here/images'

# 加载测试集
with open('/scratch/ch5085/GS_DATASET/reddits_test_ids.txt') as f:
    content = f.read().strip()
    test_ids = [x.strip() for x in content.split(',')]

print(f"Test nodes: {len(test_ids)}")

def encode_image(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def generate_caption(node_id):
    img_path = f'{IMAGE_DIR}/{node_id}.jpg'
    try:
        img_b64 = encode_image(img_path)
        response = client.chat.completions.create(
            model='qwen-vl-plus',
            messages=[{
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}
                    },
                    {
                        'type': 'text',
                        'text': 'Describe this image in 2-3 sentences. Focus on the main subject, visual appearance, and what type of content it is.'
                    }
                ]
            }],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return None

# 先测试5个
print("Testing 5 nodes...")
captions = {}
for nid in test_ids[:5]:
    caption = generate_caption(nid)
    if caption:
        captions[nid] = caption
        print(f"  Node {nid}: {caption[:80]}...")
    else:
        print(f"  Node {nid}: FAILED")

print(f"\nSuccess: {len(captions)}/5")

# 生成全部 1000 个
print(f"\nGenerating captions for all {len(test_ids)} nodes...")
captions = {}
failed = []

for nid in tqdm(test_ids):
    caption = generate_caption(nid)
    if caption and "not available" not in caption.lower() and "deleted" not in caption.lower():
        captions[nid] = caption
    else:
        failed.append(nid)

with open('/scratch/ch5085/graphsearch/data/reddit_captions.json', 'w') as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)

print(f"Done! Success: {len(captions)}, Failed: {len(failed)}")
print("Saved → reddit_captions.json")
