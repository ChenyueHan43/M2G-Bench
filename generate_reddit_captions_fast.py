import json, os, base64
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv('/scratch/ch5085/graphsearch/.env')

client = OpenAI(
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)

IMAGE_DIR = '/scratch/jl11523/dataset-mgllm/Reddit_here/images'

with open('/scratch/ch5085/GS_DATASET/reddits_test_ids.txt') as f:
    content = f.read().strip()
    test_ids = [x.strip() for x in content.split(',')]

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
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
                    {'type': 'text', 'text': 'Describe this image in 2-3 sentences. Focus on the main subject, visual appearance, and what type of content it is.'}
                ]
            }],
            max_tokens=150
        )
        caption = response.choices[0].message.content
        if "not available" in caption.lower() or "deleted" in caption.lower():
            return node_id, None
        return node_id, caption
    except Exception as e:
        return node_id, None

print(f"Generating captions for {len(test_ids)} nodes with 10 workers...")
captions = {}
failed = []

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(generate_caption, nid): nid for nid in test_ids}
    for future in tqdm(as_completed(futures), total=len(test_ids)):
        nid, caption = future.result()
        if caption:
            captions[nid] = caption
        else:
            failed.append(nid)

with open('/scratch/ch5085/graphsearch/data/reddit_captions.json', 'w') as f:
    json.dump(captions, f, indent=2, ensure_ascii=False)

print(f"Done! Success: {len(captions)}, Failed: {len(failed)}")
