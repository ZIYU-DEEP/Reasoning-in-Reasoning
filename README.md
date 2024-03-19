# README

## Installation
```bash
conda create --name rir python=3.10
conda activate rir
pip install -r requirements.txt
```

## Run
```bash
# Low-level search
python run.py --config_name dojo_test.yaml --search_method bfs_low

# Low-level search with raw informal proof from minif2f (sanity check; currently worse than low-level)
python run.py --config_name dojo_test.yaml --search_method bfs_low_with_raw_high

# Bi-level search
python run.py --config_name dojo_test.yaml --search_method bfs_bilevel
```