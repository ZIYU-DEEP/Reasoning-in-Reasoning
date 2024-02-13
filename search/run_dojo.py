from accelerate import Accelerator
from pylean import LeanServer
import torch
import heapq
import concurrent
import transformers
import os
import vllm
import json
import time
from datetime import datetime
import logging
import argparse
import torch

from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    StoppingCriteria, 
    StoppingCriteriaList,
)
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from lean_dojo import *
import heapq
import subprocess
import yaml

from datetime import datetime
from tqdm import tqdm, trange
from pprint import pprint, pformat
from pathlib import Path

from utils.search_dojo import *
from utils import search_dojo
from utils import misc



# ------------------------------------------------
# Some prep
accelerator = Accelerator()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  
logger = logging.getLogger()
# ------------------------------------------------


# -------------------------------------------------------------------
# Set the parser and config
parser = argparse.ArgumentParser()

# Arguments for dataset
parser.add_argument('-cfgr', '--config_root', type=str, 
                    default='./configs/')
parser.add_argument('-cfg', '--config_name', type=str, 
                    default='dojo_default.yaml')
parser.add_argument('-sm', '--search_method', type=str, 
                    default='', help='Rewrite the search method in config.',
                    choices=['simple_search', 'best_first_search', 'mcts', 'rir'])
parser.add_argument('-re', '--resume_from', type=str, default='', 
                    help='Resume from a specific problem, e.g., amc12_2000_p12.')

# Parse the arguments
args = parser.parse_args()

# Get the config
config_path = Path(args.config_root) / args.config_name
with open(config_path, 'r') as y_file:
    p = yaml.load(y_file, Loader=yaml.FullLoader)
    p = DotDict(p)

# Update the search method if specified
if args.search_method:
    assert args.search_method in ['simple_search', 'best_first_search', 'mcts', 'rir']
    p.search_method = args.search_method

# -------------------------------------------------------------------
# Set the logger
# Record the time
time_now = datetime.now().strftime('%m-%d-%H-%M')
log_folder = Path(p.log_root) / p.search_method
os.makedirs(log_folder, exist_ok=True)

logger = misc.set_logger(log_folder / f'{time_now}.log')
logger.info(f'{p.log_path}')
logger.info(pformat(p, indent=4))

# Set the results folder
results_folder = Path(p.results_root) / p.search_method 
os.makedirs(results_folder, exist_ok=True)
results_path = results_folder / f'{time_now}.json'
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Set the dataset and the model
# Set up the minif2f dataset
repo, data = search_dojo._load_data(dataset_name=p.dataset_name,
                                    dataset_path=p.dataset_path)

if args.resume_from:
    re_ind = [item['full_name'] for item in data].index(args.resume_from)
    data = data[re_ind:] 

# Split data in different shards and use CUDA_VISIBLE_DEVICES to control
# shard_size = len(data) // args.num_shards
# data = data[args.shard * shard_size:(args.shard + 1) * shard_size]
# print("Shard size: %d" % (len(data)))

# Load the model
if p.gen_method == 'vllm':
    model, tokenizer = search_dojo.load_model_vllm(
        model_name=p.model_name,
        tp_degree=p.tp_degree,
        dtype=p.dtype,
        max_num_batched_tokens=p.max_num_batched_tokens)
    stopping_criteria = None  

elif p.gen_method == 'hf':
    model, tokenizer = search_dojo.load_model_hf(model_name=p.model_name,
                                                 accelerator=accelerator)
    stop_words_ids = [torch.tensor([807])]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Run the search
def main():
    results = []

    for example in tqdm(data, total=len(data)):
        
        # Set up the data
        file_path = example['file_path']
        theorem_name = example['full_name']
        theorem = Theorem(repo, file_path, theorem_name)
        
        # Start search
        if p.search_method == 'simple_search':
            raise NotImplementedError('Simple search is not implemented yet.')
        
        if p.search_method == 'best_first_search':
            attempt_results = best_first_search(theorem=theorem,
                                                model=model,
                                                tokenizer=tokenizer,
                                                max_iters=p.max_iters,
                                                temperatures=p.temperatures,
                                                num_samples=p.num_samples,
                                                prompt_fn=search_dojo._prompt_fewshot,
                                                timeout=p.timeout,
                                                early_stop=p.early_stop,
                                                max_tokens=p.max_tokens,
                                                stopping_criteria=stopping_criteria,
                                                gen_method=p.gen_method)
        
        if p.search_method == 'mcts':
            attempt_results = mct_search(theorem=theorem,
                                         model=model,
                                         tokenizer=tokenizer,
                                         max_iters=p.max_iters,
                                         temperatures=p.temperatures,
                                         num_samples=p.num_samples,
                                         prompt_fn=search_dojo._prompt_fewshot,
                                         timeout=p.timeout,
                                         early_stop=p.early_stop,
                                         max_tokens=p.max_tokens,
                                         stopping_criteria=stopping_criteria,
                                         gen_method=p.gen_method)

        if p.search_method == 'rir':
            raise NotImplementedError('Bilevel MCTS is not implemented yet.')

        result = {
            'attempt_results': attempt_results,
            'success': any([x['success'] for x in attempt_results]),
            'example': example
        }
        logger.info(pformat(result, indent=4))
        logger.info('\n-----\n')
        
        # Add ther result to the result
        results.append(result)
        with open(results_path, 'w') as f:
            json.dump({'results': results,
                       'config': p}, f, indent=4)
        search_dojo.print_stats(results)


if __name__ == '__main__':
    main()
# -------------------------------------------------------------------
