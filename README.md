# Reasoning in Reasoning

The current codebase relies on `Lean 4`.


## Installation
First, for the basic environment:
```bash
# Setting up the environment
conda create --name rir python=3.10
conda activate rir
pip install -r requirements.txt

# Using huggingface
huggingface-cli login
```

Then, to install `Lean`:
```bash
# Install Lean with Elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env

# Fix the leanprover
cd ~  # Or your working directory
git clone https://github.com/leanprover-community/repl.git
cd repl

# Adding mathlib dependencies
echo -e '\nrequire mathlib from git "https://github.com/leanprover-community/mathlib4"' >> lakefile.lean
curl https://raw.githubusercontent.com/leanprover-community/mathlib4/master/lean-toolchain -o lean-toolchain

# Update with lake
lake update
lake build
lake build Mathlib

# Add the path
echo "export PATH_TO_LEAN_REPL=\"$(pwd)\"" >> ~/.bashrc
source ~/.bashrc

# Get back to our env
conda activate rir
```

## Run
```bash
python run.py --language Lean4 --problem_name problem_fact
python run.py --language Lean4 --problem_name problem_fact
python run_whole.py --language Lean4 --n_samples 10 --problem_name problem_fact --greedy False 
python run_reflection.py --language Lean4 --problem_name problem_fact
```

## TODO
The current implementation is MCTS where the each step is connected to the low-level verifier. We are implementing an additional high-level planner above this (will use the LightZero approach). 

A intuitive idea is that we first ask the algorithm to do a high-level search on the proof plan.




<!-- Notice the implementation of stepwise reflection can be problematic. -->

<!-- And we should not trigger the verifier at each step when the proof is incomplete. -->


## References
We have referenced the following repositories:
- [LLM Tree Search](https://github.com/waterhorse1/LLM_Tree_Search)
- [LightZero](https://github.com/opendilab/LightZero)
- [LLM Verifier](https://github.com/namin/llm-verified-with-monte-carlo-tree-search/tree/main)
- [Fun Search](https://github.com/google-deepmind/funsearch)
- [pySagredo](https://github.com/zhangir-azerbayev/pySagredo)
- [Monte Carlo Tree Search Basics](https://github.com/ImparaAI/monte-carlo-tree-search)
- [Learning Lean4](https://leanprover-community.github.io/learn.html)
- [Language Agent Tree Search](https://arxiv.org/pdf/2310.04406v2.pdf)
- [Neural Theorem Proving Tutorial](https://github.com/wellecks/ntptutorial/tree/main)
- [GPT-F](https://arxiv.org/pdf/2009.03393.pdf)
- [pylean](https://github.com/yeahrmek/pylean)
- [HyperTree Proof Search](https://openreview.net/pdf?id=J4pX8Q8cxHH) ([slides](https://github.com/tanchongmin/TensorFlow-Implementations/blob/main/Paper_Reviews/Hypertree%20Proof%20Search%20Slides.pdf))
    - This is in essence multi-level tree search.
- [dl4math](https://github.com/lupantech/dl4math)
