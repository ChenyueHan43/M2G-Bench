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
cp .env.example .env          # add your DeepSeek API key
python run_experiment.py      # Cora dataset
python run_products.py        # Amazon-Products dataset
python run_all_datasets.py    # PubMed + Reddit datasets
```

## Results (Setting A: Text-only Baseline)

| Dataset | Domain | DeepSeek-chat | Paper (Qwen2.5-32B) | Gap |
|---------|--------|--------------|---------------------|-----|
| Amazon-Products | E-commerce | 70.0% (n=100) | 71.7% | -1.7% |
| PubMed | Citation | 84.0% (n=50) | 89.8% | -5.8% |
| Reddit | Social | **76.0%** (n=50) | 67.4% | **+8.6%** |
| Cora | Citation | 16.0% (n=50) | 65.9% | -49.9% |

**Notes:**
- Cora gap is due to model capability difference (DeepSeek-chat vs Qwen2.5-32B).
  Academic paper classification is more sensitive to model scale.
- Reddit and Products results are competitive with or exceed the paper's numbers.
- Formal experiments with Qwen2.5-32B pending HPC access.

## Project Roadmap

- [x] Setting A: Text-only baseline (this repo)
- [ ] Setting B: Text + Image Captions (GPT-4o-mini)
- [ ] Setting C: Text + CLIP Visual Embeddings

## Citation
```bibtex
@article{liu2026graphsearch,
  title={GraphSearch: Agentic Search-Augmented Reasoning for Zero-Shot Graph Learning},
  author={Liu, Jiajin and Sun, Yuanfu and Fan, Dongzhe and Tan, Qiaoyu},
  journal={arXiv preprint arXiv:2601.08621},
  year={2026}
}
```