# README

## Run
To run with the default config setting (.configs/default.yaml):
```python
# Simple search (for whole proof)
python main.py --search_method simple_search

# Best-first search (step by step, one tactic a time)
python main.py --search_method best_first_search
```


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
# Assuming you have cloned this repository at ~/github/
cd ~/github  # or your working directory

# Install Lean with Elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env

# Fix the leanprover
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

<!-- require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "38dbcd8285bc4b1391619c12f158a7409f3dfc12" -->

## TODO
- [ ] test Llema in pylean env (currently the dequeue part is a bit misbehaving)
- [ ] test Llema in dojo env
- [ ] add the mcts part
- [ ] modify the prompt

