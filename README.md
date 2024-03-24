# README

> [!NOTE]
> The current version is still in testing. A tested version is in the [raptors](https://github.com/ZIYU-DEEP/bilevel-reasoner/tree/raptors) branch, where we have the benchmark results with Llemma-7b on `bfs_low` and `bfs_low_with_raw_high` methods. The latter is very much underperforming (23 out of 244 problems), likely due to the fact that the raw informal proof is unstructured and the model capacity is bad. A proof of concept colab with GPT-4 can be found at [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BI3u6NwVtefTpWzQNj-OwPf6C3ONuPFn?usp=sharing).

## Installation

```bash
conda create --name rir python=3.10
conda activate rir
pip install -r requirements.txt
```

## Run

```bash
# Low-level search
python run.py --config_name dojo_test_mew.yaml --search_method bfs_low

# Low-level search with raw informal proof from minif2f (sanity check)
python run.py --config_name dojo_test_mew.yaml --search_method bfs_low_with_raw_high

# Bi-level search
python run.py --config_name dojo_test_mew.yaml --search_method bfs_bilevel
```

You may change the `--gen_method` argument to `openai` (check `./configs/dojo_test_mew_openai.yaml` for more information). In case you haven't set up the API key:
```bash
echo 'export OPENAI_API_KEY="your_api_key"' >> ~/.bashrc
source ~/.bashrc
```

## TODO

- [ ]  Update the current `minif2f-Lean4` repo to match the latest `lean-dojo` package.
- [ ]  Add smt-solver in the low-level search stage.
- [ ]  Double check the benchmark results on bi-level BFS.
- [ ]  Update the single-level and bi-level MCTS and get benchmark results.
- [ ]  Consider the design for trees in trees (how to decide to expand the next hyper node?)
