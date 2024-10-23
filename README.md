# Reasoning in Reasoning

This repository is based on [ReProver](https://github.com/lean-dojo/ReProver).


## 0. Structure
```
root
|== generator
│   └── confs
│       └── ....yaml
│   └── datamodule.py
│   └── model.py
│   └── main.py
|== prover_rir
│   └── search_tree.py
│   └── proof_search.py
│   └── model.py
|== utils
│   └── stats.py
│   └── data_stats.py
│   └── download_data.py
│   └── trace_repo.py
|== scripts
│   └── eval
│       └── ....sh
│   └── train
│       └── ....sh
|== common.py
|== requirements.txt
```

## 1. Setup
### 1.1. Environment
```bash
# Set up the environment
cd RiR
conda create -n dojo python=3.10
conda activate dojo
pip install -r requirements.txt

# Set up elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env

# You may add the following to your .bashrc
echo 'export PATH="$HOME/.elan/bin:$PATH"' >> ~/.bashrc
echo 'export GITHUB_ACCESS_TOKEN="YOUR_GITHUB_ACCESS_TOKEN"' >> ~/.bashrc

# Make them effective
source ~/.bashrc
conda activate dojo
chmod -R +x ./scripts
```

```bash
# Add the project directory to PYTHONPATH
PROJECT_DIR=$(pwd)
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"
```

### 1.2. Download Data
#### Download the LeanDojo Dataset
```bash
python utils/download_data.py  # or python utils/download_data_lambda.py if it fails
python utils/trace_repos.py
```
#### Download the miniF2F dataset
```bash
python utils/dojo_mini.py
python utils/abstract_minif2f.py
```
Now you should have both `leandojo_benchmark_4` and `minif2f` under `$PROJECT_DIR/data`.

### 1.3. Install Git LFS
If you have sudo permissions:
```bash
sudo apt install git-lfs
```

If you do not have sudo permissions:
```bash
# Define the working directory where Git LFS will be installed.
WORKING_DIR=${WORKING_DIR}  # Replace ${WORKING_DIR} with the desired directory
mkdir -p $WORKING_DIR/.local/bin && cd $WORKING_DIR

# Download the Git LFS binary for Linux and extract the files
wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-linux-amd64-v3.4.1.tar.gz && tar xvf git-lfs-linux-amd64-v3.4.1.tar.gz

# Modify the prefix for installation
cd git-lfs-3.4.1/ && chmod +x install.sh && vi install.sh
# Modify the prefix from /usr/local/ to $WORKING_DIR/.local
# This step is manual
# Or sed -i "s|/usr/local|$WORKING_DIR/.local|g" install.sh

# Add the installation directory to PATH and run
echo 'export PATH="'$WORKING_DIR'/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc && ./install.sh

# Verify the Git LFS installation and install for this repo
git-lfs version
```

### 1.4. Download Lightning Checkpoints
```bash
cd $PROJECT_DIR
git lfs install && mkdir ckpts
```
#### Download the Reprover Checkpoints
```bash
cd $PROJECT_DIR/ckpts
git clone https://huggingface.co/kaiyuy/leandojo-pl-ckpts.git
cd leandojo-pl-ckpts
git lfs fetch --all
```

#### Download the RiR Checkpoints
```bash
cd $PROJECT_DIR/ckpts
git clone https://huggingface.co/cat-searcher/rir-pl-ckpts.git
cd rir-pl-ckpts
git lfs fetch --all
```

## 2. Evaluation
Below is an example command. See the `./scripts/` folder for evaluation with different models and settings.
```bash
# Make them executable
cd $PROJECT_DIR
chmod -R +x ./scripts

# Sanity check
./scripts/eval/sanity_check.sh

# Run with RiR
./scripts/eval/leandojo_rir.sh

# Run with default reprover
./scripts/eval/leandojo_default.sh
```

## 3. Training
Below is an example command. See the `./scripts/` folder for more information.
```bash
# Make them executable
cd $PROJECT_DIR
chmod -R +x ./scripts

# Train the goal generator
./scripts/train/planner.sh

# Train the goal-driven tactic generator
./scripts/train/actor.sh

# Train the joint generator
./scripts/train/joint.sh
```
