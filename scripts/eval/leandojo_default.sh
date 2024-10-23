#!/bin/bash

# This is to run with the default baseline settings with reprover ckpt

# Set the variables
PROJECT_DIR=$(pwd) 
start_ind=400   # Adjust this!
num_theorems=1  # Adjust this!
num_workers=1   # 8 for H100-80G usually work
num_gpus=1
gen_type="default"
timeout=600

# Set the path
data_path="data/leandojo_benchmark_4/random/"
split="test"

# Set the actor path
default_path="$PROJECT_DIR/ckpts/leandojo-pl-ckpts/generator_random.ckpt"
echo $default_path

# Run
CUDA_VISIBLE_DEVICES=0 python prover/evaluate.py \
    --data-path  $data_path \
    --split $split \
    --num-theorems $num_theorems  \
    --num-workers $num_workers \
    --num-gpus $num_gpus \
    --ckpt_path $default_path \
    --gen-type $gen_type \
    --start-ind $start_ind 
