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
# Assuming you have cloned this folder at ~/github/
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
<!-- 
We also need `pylean` as a wrapper to get proof states.
```bash
cd ~

git clone https://github.com/zhangir-azerbayev/repl
cd repl

git checkout bddf452deda0df2240b248e651bcc37fb8e59d01

cd pylean

python setup.py develop 
```
You may need go to `pylean/__init__.py` and overide the path as `path_to_repl = os.environ.get('PATH_TO_LEAN_REPL')`, as we previously defined. -->


## Run
(follows from llm verfier)
```bash
python run.py --language Lean4 --problem_name problem_fact
python run.py --language Lean4 --problem_name problem_fact
python run_whole.py --language Lean4 --n_samples 10 --problem_name problem_fact --greedy False 
python run_reflection.py --language Lean4 --problem_name problem_fact
```
We put exploration code in the `./exploration` folder.

## TODO
- [ ] Make a concrete baseline of existing MCTS with Lean verifier on MiniF2F
- [ ] Apply bi-level search 
- [ ] Add training in the loop

## Some Random Notes
The current implementation uses LLM verifier, where the each and every step is verified formally (which is very inefficient). We are implementing an additional high-level planner above this. 

A rough idea is that we first ask the algorithm to do a high-level search on the proof plan. Each step correspond to a tactic in natural language (while there might be sequential dependency for tactics, we encourage them to be mostly independent such that this division of of maximum information). After this search, we choose the trajectory with the highest value. For each node in this trajectory, we first do formalization, that means each tactic is decomposed into formal subgoals. The subgoals will be recursively expanded, until we reach the termination conditions (solved with empty set of subgoals, or failures).

This is different from [HyperTree](https://openreview.net/pdf?id=J4pX8Q8cxHH) as we handle the state abstraction and the tactic dependency in a more efficient way. 



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
