import json, os, requests
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from io import BytesIO

# 加载 CLIP 模型
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

# 加载图片索引
with open('/scratch/ch5085/graphsearch/data/m2g_image_index.json') as f:
    image_index = json.load(f)

# 只处理测试集有图片的节点
with open('/scratch/ch5085/GS_DATASET/products_test_ids.txt') as f:
    content = f.read().strip()
    test_ids = set(str(x.strip()) for x in content.split(','))

test_with_images = {k: v for k, v in image_index.items() if k in test_ids}
print(f"Extracting CLIP embeddings for {len(test_with_images)} nodes...")

embeddings = {}
failed = []

for node_id, item in tqdm(test_with_images.items()):
    try:
        response = requests.get(item['image_url'], timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # normalize
        embeddings[node_id] = emb[0].tolist()
    except Exception as e:
        failed.append(node_id)

print(f"Success: {len(embeddings)}, Failed: {len(failed)}")

with open('/scratch/ch5085/graphsearch/data/clip_embeddings.json', 'w') as f:
    json.dump(embeddings, f)
print("Saved → clip_embeddings.json")
