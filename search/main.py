from pylean import LeanServer
from utils import (
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

from pprint import pprint
from tqdm import tqdm
from itertools import islice
from datasets import load_dataset

import tiktoken
import os
import yaml
import argparse

os.environ['TOKENIZERS_PARALLELISM'] = 'true'  


# -------------------------------------------------------------------
# Set the parser and config
parser = argparse.ArgumentParser()

# Arguments for dataset
parser.add_argument('-cp', '--config_path', type=str, 
                    default='./configs/default.yaml')

# Parse the arguments
args = parser.parse_args('')

# Get the config
with open(args.config_path, 'r') as y_file:
    p = yaml.load(y_file, Loader=yaml.FullLoader)
    p = DotDict(p)
    pprint(p)
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Set the dataset and the model
# Set up the minif2f dataset
dataset = load_dataset(p.dataset_name, split=p.split)

# Create the theorem_list
theorem_list = []
for x in tqdm(islice(dataset, p.slice_size)): 
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
        
        if p.do_simple_search:
            result = simple_search(model=model,
                                   tokenizer=tokenizer,
                                   header=header,
                                   statement=statement,
                                  search_budget=p.search_budget)
        
        if p.do_best_first_search:
            result = best_first_search(model=model,
                                       tokenizer=tokenizer,
                                       header=header,
                                       statement=statement,
                                       max_iters=p.max_iters,
                                       temperatures=p.temperatures,
                                       num_samples=p.num_samples,
                                       verbose=p.verbose)
        
        if p.do_mcts:
            raise NotImplementedError('MCTS is not implemented yet.')
        
        if p.do_bilevel_mcts:
            raise NotImplementedError('Bilevel MCTS is not implemented yet.')
        
        print(result)
        print('\n-----\n')
        results.append(result)

    print('Success Rate: ', sum(x['success'] for x in results) / len(results)) 

if __name__ == '__main__':
    main()
# -------------------------------------------------------------------
