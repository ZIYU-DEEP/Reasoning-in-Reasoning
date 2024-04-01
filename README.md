# README

<!-- > [!NOTE]
> The current version is still in testing. A tested version is in the [raptors](https://github.com/ZIYU-DEEP/bilevel-reasoner/tree/raptors) branch, where we have the benchmark results with Llemma-7b on `bfs_low` and `bfs_low_with_raw_high` methods. The latter is very much underperforming (23 out of 244 problems), likely due to the fact that the raw informal proof is unstructured and the model capacity is bad. A proof of concept colab with GPT-4 can be found at [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BI3u6NwVtefTpWzQNj-OwPf6C3ONuPFn?usp=sharing). -->

## Setup

```bash
# Setup for the environment
conda create --name rir python=3.10
conda activate rir
pip install -r requirements.txt  # You may ignore the vllm installation if you do not have cuda

# Setup for the paths 
echo "export GITHUB_ACCESS_TOKEN='YOUR GITHUB TOKEN'" >> ~/.bashrc  # Optional - to avoid rate limit issues when setting up the dojo
echo "export RAY_TMPDIR='YOUR TEMP DIR'" >> ~/.bashrc  # Optional - to avoid ray init issues
```

Optional installation guide for SMT is in [`INSTALLATION.md`](https://github.com/ZIYU-DEEP/bilevel-reasoner/blob/main/INSTALLATION.md).

## Run

### Running with OpenAI API

Make sure you have set up the API key in your environment before running.

```bash
echo 'export OPENAI_API_KEY="your_api_key"' >> ~/.bashrc
source ~/.bashrc
```

```bash
# Low-level search
python run.py --config_name bfs_low.yaml --gen_method openai --model_name gpt-4-0125-preview

# Bi-level search
python run.py --config_name bfs_high.yaml --gen_method openai --model_name gpt-4-0125-preview
```

### Running with Open-Source Models

```bash
# Low-level search
python run.py --config_name bfs_low_llemma_7b.yaml --model_name open-web-math/llemma_7b

# Bi-level search
python run.py --config_name bfs_bilevel_llemma_7b.yaml --model_name open-web-math/llemma_7b
```

Notice that for test purpose, you may set `--slice_size 1` to test only on the first theorem.

## TODO
- [ ] Fix the generation issue for bilevel search using `vllm`. The high-level proof is not extacted correctly due to the `stop` setting.
- [ ] Add `smt-solver` as a default tactic.
