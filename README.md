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

## Results (Setting A: Text-only Baseline)

### Qwen2.5-32B (same model as paper)

| Dataset | Ours | Paper | Gap |
|---------|------|-------|-----|
| Amazon-Products | 67.8-69.4% (n=500) | 71.7% | -2.3% |
| PubMed | 81.5% (n=200) | 89.8% | -8.3% |
| Reddit | 64.0% (n=200) | 67.4% | -3.4% |

### DeepSeek-chat (alternative model)

| Dataset | Ours | Paper | Gap |
|---------|------|-------|-----|
| Amazon-Products | 70.0% (n=100) | 71.7% | -1.7% |
| PubMed | 84.0% (n=50) | 89.8% | -5.8% |
| Reddit | 76.0% (n=50) | 67.4% | +8.6% |
| Cora | 16.0% (n=50) | 65.9% | -49.9% |

## Project Roadmap

- [x] Setting A: Text-only baseline
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
