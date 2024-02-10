from pylean import LeanServer
from utils.search_pylean import (
    load_model,
    simple_search,
    best_first_search,
    sequence_scores,
    run_code,
    is_done,
    get_goal,
    generate,
    parse_step,
    DotDict
)
from pprint import pprint, pformat
from tqdm import tqdm
from itertools import islice
from datasets import load_dataset
from pathlib import Path

from utils import misc

import tiktoken
import os
import time
from datetime import datetime
import yaml
import argparse

os.environ['TOKENIZERS_PARALLELISM'] = 'true'  


# -------------------------------------------------------------------
# Set the parser and config
parser = argparse.ArgumentParser()

# Arguments for dataset
parser.add_argument('-cfgr', '--config_root', type=str, 
                    default='./configs/')
parser.add_argument('-cfg', '--config_name', type=str, 
                    default='default.yaml')
parser.add_argument('-sm', '--search_method', type=str, 
                    default='', help='Rewrite the search method in config.',
                    choices=['simple_search', 'best_first_search', 'mcts', 'rir'])

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
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Set the dataset and the model
# Set up the minif2f dataset
dataset = load_dataset(p.dataset_name, split=p.split)

# Create the theorem_list
theorem_list = []
for x in tqdm(islice(dataset, p.slice_size if p.slice_size else len(dataset))): 
    # Extract them
    header = x['header'] + '\n'
    statement = x['formal_statement'].replace('sorry', 'by {}')
    informal_statement = x['informal_stmt']
    informal_proof = x['informal_proof']
    
    theorem_list.append((header, statement, informal_statement, informal_proof))

# Load the model
model, tokenizer = load_model(p.model_name)
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Run the search
def main():
    results = []

    for data in theorem_list:
        header, statement, informal_statement, informal_proof = data
        
        if p.search_method == 'simple_search':
            result = simple_search(model=model,
                                   tokenizer=tokenizer,
                                   header=header,
                                   statement=statement,
                                   search_budget=p.search_budget)
        
        if p.search_method == 'best_first_search':
            result = best_first_search(model=model,
                                       tokenizer=tokenizer,
                                       header=header,
                                       statement=statement,
                                       max_iters=p.max_iters,
                                       temperatures=p.temperatures,
                                       num_samples=p.num_samples,
                                       verbose=p.verbose)
        
        if p.search_method == 'mcts':
            raise NotImplementedError('MCTS is not implemented yet.')
        
        if p.search_method == 'rir':
            raise NotImplementedError('Bilevel MCTS is not implemented yet.')
        
        logger.info(pformat(result, indent=4))
        logger.info('\n-----\n')
        results.append(result)

    logger.info(f"Success Rate: {sum(x['success'] for x in results) / len(results)}") 


if __name__ == '__main__':
    main()
# -------------------------------------------------------------------
