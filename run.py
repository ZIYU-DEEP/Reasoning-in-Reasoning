from accelerate import Accelerator
import torch
import heapq
import concurrent
import transformers
import os
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

# from utils.search_dojo import *  # To remove this
from utils.search import *
from utils import misc
from utils import prompts
from utils import llms
from utils import search

try:
    import vllm
except Exception:
    pass


# ------------------------------------------------
# Some prep
try:
    accelerator = Accelerator()
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
except Exception:
    pass
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
                    default='', help='Rewrite the search method in config.')
parser.add_argument('-gm', '--gen_method', type=str,
                    default='', help='Rewrite the llm generation method in config.')
parser.add_argument('-mn', '--model_name', type=str,
                    default='', help='Rewrite the model name in config.')
parser.add_argument('-sp', '--split', type=str,
                    default='', help='Rewrite the split in config.')
parser.add_argument('-re', '--resume_from', type=str, default='',
                    help='Resume from a specific problem, e.g., imo_1969_p2.')
parser.add_argument('-ss', '--slice_size', type=int, default=None,
                    help='The size of the slice of the dataset.')

# Parse the arguments
args = parser.parse_args()

# Get the config
config_path = Path(args.config_root) / args.config_name
with open(config_path, 'r') as y_file:
    p = yaml.load(y_file, Loader=yaml.FullLoader)
    p = misc.DotDict(p)

# Update the search method if specified
if args.search_method:
    p.search_method = args.search_method
    
# Update the generation method if specified
if args.gen_method:
    p.gen_method = args.gen_method

# Update the model name if specified
if args.model_name:
    p.model_name = args.model_name

# Update the split if specified
if args.split:
    p.split = args.split

# -------------------------------------------------------------------
# Set the loggers
# Record the time
time_now = datetime.now().strftime('%m-%d-%H-%M')
log_folder = Path(p.log_root) / p.search_method
os.makedirs(log_folder, exist_ok=True)

logger = misc.set_logger(log_folder / f'{time_now}.log', level=logging.DEBUG)
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
repo, data = misc.load_data_dojo(dataset_name=p.dataset_name,
                                 dataset_path=p.dataset_path,
                                 split=p.split,
                                 url=p.url,
                                 commit=p.commit)

if args.resume_from:
    re_ind = [item['full_name'] for item in data].index(args.resume_from)
    data = data[re_ind:]

if args.slice_size:
    data = data[:args.slice_size]

# Split data in different shards and use CUDA_VISIBLE_DEVICES to control
# shard_size = len(data) // args.num_shards
# data = data[args.shard * shard_size:(args.shard + 1) * shard_size]
# print("Shard size: %d" % (len(data)))

# Load the model
if p.gen_method == 'vllm':
    model, tokenizer = llms.load_model_vllm(
        model_name=p.model_name,
        tp_degree=p.tp_degree,
        dtype=p.dtype,
        max_num_batched_tokens=p.max_num_batched_tokens)
    # stopping_criteria = None

if p.gen_method == 'hf':
    model, tokenizer = llms.load_model_hf(
        model_name=p.model_name,
        accelerator=accelerator)

if p.gen_method == 'openai':
    model, tokenizer = p.model_name, None
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Run the search
def main():
    results = []

    # Low-level Search
    if p.search_method == 'bfs_low':
        search_fn = search.search_low
        prompt_fn_low = prompts._prompt_low
        prompt_fn_high = None
        search_algorithm = 'bfs'

    if p.search_method == 'mcts_low':
        search_fn = search.search_low
        prompt_fn_low = prompts._prompt_low
        prompt_fn_high = None
        search_algorithm = 'mcts'

    # Low-level Search with old code
    # if p.search_method == 'bfs_low_old':
    #     search_fn = search.bfs_low_old
    #     prompt_fn_low = prompts._prompt_low
    #     prompt_fn_high = None

    # Low-level Search with Raw High-Level Proof Plan in Context
    if p.search_method == 'bfs_low_with_raw_high':
        search_fn = search.search_low
        prompt_fn_low = prompts._prompt_low_with_high
        prompt_fn_high = None
        search_algorithm = 'bfs'

    # Bi-level Search
    if p.search_method == 'bfs_bilevel':
        search_fn = search.bfs_bilevel
        prompt_fn_low = prompts._prompt_low_with_high
        prompt_fn_high = prompts._prompt_high
        search_algorithm = 'bfs'

    for example in tqdm(data, total=len(data)):

        # Set up the data
        file_path = example['file_path']
        theorem_name = example.get('full_name', example.get('id'))
        theorem = Theorem(repo, file_path, theorem_name)

        # To load the data from the huggingface dataset
        formal_statement = example.get('formal_statement', example.get('statement'))
        informal_statement = example['informal_stmt']
        informal_proof = example['informal_proof']
        
        logger.info(file_path)
        logger.info(theorem_name)

        # Start search
        attempt_results = search.proof_search(theorem=theorem,
                                                model=model,
                                                tokenizer=tokenizer,
                                                max_iters_low=p.max_iters_low,
                                                max_iters_high=p.max_iters_high,
                                                temperatures=p.temperatures,
                                                num_samples_low=p.num_samples_low, 
                                                num_samples_high=p.num_samples_high,
                                                search_fn=search_fn,
                                                prompt_fn_low=prompt_fn_low,
                                                prompt_fn_high=prompt_fn_high,
                                                timeout=p.timeout,
                                                early_stop=p.early_stop,
                                                max_tokens=p.max_tokens,  # Maybe two params for low and high
                                                stop=p.stop,
                                                gen_method=p.gen_method,
                                                formal_statement=formal_statement,
                                                informal_statement=informal_statement,
                                                plan_high=informal_proof,  # This will be overwritten by machine generated plan in bfs_bilevel
                                                search_algorithm=search_algorithm,
                                                )


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
        search.print_stats(results)
        try:
            search.log_stats(results)
        except Exception:
            pass


if __name__ == '__main__':
    main()
# -------------------------------------------------------------------
