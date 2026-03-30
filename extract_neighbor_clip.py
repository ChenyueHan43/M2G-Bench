import json, requests, torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from io import BytesIO

print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

with open('/scratch/ch5085/graphsearch/data/neighbors_to_clip.json') as f:
    neighbors = json.load(f)

with open('/scratch/ch5085/graphsearch/data/clip_embeddings.json') as f:
    clip_embs = json.load(f)

print(f"Processing {len(neighbors)} neighbor nodes...")
failed = []

for node_id, item in tqdm(neighbors.items()):
    try:
        response = requests.get(item['image_url'], timeout=10)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        clip_embs[node_id] = emb[0].tolist()
    except Exception as e:
        failed.append(node_id)

with open('/scratch/ch5085/graphsearch/data/clip_embeddings.json', 'w') as f:
    json.dump(clip_embs, f)

print(f"Success: {len(clip_embs)}, Failed: {len(failed)}")
print("Saved → clip_embeddings.json")
