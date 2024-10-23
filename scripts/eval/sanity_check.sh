#!/bin/bash

# Set the variables
PROJECT_DIR=$(pwd)
num_theorems=1
num_workers=1
num_gpus=1
num_sampled_goals=5
gen_type="goal_driven_tactic"
timeout=600
start_ind=400

# Set the path
data_path="data/leandojo_benchmark_4/random/"
split="test"

# Set the actor path
goal_driven_tactic_path="$PROJECT_DIR/ckpts/rir-pl-ckpts/goal_driven_tactic/checkpoint-epoch=05-step=40000-loss_val=1000.0000-loss_train=0.1496.ckpt"
echo $goal_driven_tactic_path

# Set the planner path (this one is fined from Reprover)
goal_ckpt_path="$PROJECT_DIR/ckpts/rir-pl-ckpts/goal_ckpt/checkpoint-epoch=05-step=41993-loss_val=0.0538-loss_train=0.0217.ckpt"
echo $goal_ckpt_path

# Run
CUDA_VISIBLE_DEVICES=0 python prover_rir/evaluate.py \
    --data-path $data_path \
    --split $split \
    --num-theorems $num_theorems \
    --num-workers $num_workers \
    --num-gpus $num_gpus \
    --ckpt_path $goal_driven_tactic_path \
    --goal-ckpt-path $goal_ckpt_path \
    --num-sampled-goals $num_sampled_goals \
    --gen-type $gen_type \
    --timeout $timeout \
    --start-ind $start_ind
