# README

> [!NOTE]
> The current version is still in testing. A tested version is in the [raptors](https://github.com/ZIYU-DEEP/bilevel-reasoner/tree/raptors) branch, where we have the benchmark results with Llemma-7b on `bfs_low` and `bfs_low_with_raw_high` methods. The latter is very much underperforming (23 out of 244 problems), likely due to the fact that the raw informal proof is unstructured and the model capacity is bad.

## Installation
```bash
conda create --name rir python=3.10
conda activate rir
pip install -r requirements.txt
```

## Run
```bash
# Low-level search
CONTAINER=native python run.py --config_name dojo_test.yaml --search_method bfs_low

# Low-level search with raw informal proof from minif2f (sanity check; currently worse than low-level)
CONTAINER=native python run.py --config_name dojo_test.yaml --search_method bfs_low_with_raw_high

# Bi-level search
CONTAINER=native python run.py --config_name dojo_test.yaml --search_method bfs_bilevel
```
