# M2G-Bench: GraphSearch Text-only Baseline

Reproduction of GraphSearch (arXiv 2601.08621) text-only baseline,
part of the M2G-Bench project at NYU Shanghai.

## Setup
```bash
conda create -n graphsearch python=3.10
conda activate graphsearch
pip install torch torch-geometric ogb sentence-transformers openai python-dotenv tqdm
```

## Run
```bash
cp .env.example .env  # add your API key
python run_experiment.py
```

## Smoke Test Results (n=5, Cora)
- Success Rate: 100%
- Avg Hops: 3.4
