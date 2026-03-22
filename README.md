# M2G-Bench: GraphSearch Text-only Baseline

Reproduction of GraphSearch (arXiv 2601.08621) text-only baseline,
part of the M2G-Bench project at NYU Shanghai.

**Research Question:** Does integrating visual information (images) significantly
improve the reasoning accuracy and efficiency of graph-based AI agents?

## Setup
```bash
conda create -n graphsearch python=3.10
conda activate graphsearch
pip install torch torch-geometric ogb sentence-transformers openai python-dotenv tqdm
```

## Data

Download GS_DATASET from the GraphSearch authors (contact: Qiaoyu Tan's group, NYU Shanghai).
Place it at `~/Desktop/GS_DATASET/` or update the path in the data loader.

## Run
```bash
cp .env.example .env  # add your DeepSeek or DashScope API key
python run_experiment.py      # Cora dataset
python run_products.py        # Amazon-Products dataset
```

## Results (Text-only Baseline, Setting A)

| Dataset | Model | n | Success Rate | Paper (Qwen2.5-32B) |
|---------|-------|---|-------------|---------------------|
| Cora | DeepSeek-chat | 50 | 16% | 65.9% |
| Amazon-Products | DeepSeek-chat | 100 | **70.0%** | 71.7% |

Note: Gap on Cora is due to model capability difference (DeepSeek-chat vs Qwen2.5-32B).
Products baseline is within 1.7% of the paper's reported number.

## Project Roadmap

- [x] Setting A: Text-only baseline (this repo)
- [ ] Setting B: Text + Image Captions (GPT-4o-mini)
- [ ] Setting C: Text + CLIP Visual Embeddings