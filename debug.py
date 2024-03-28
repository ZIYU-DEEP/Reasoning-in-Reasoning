import json
from lean_dojo import *


# Helper Function
def load_data_dojo(dataset_name: str='minif2f',
                   dataset_path: str='./data/minif2f_lean4_dojo.jsonl',
                   split: str='valid'):
    """
    Load the data and the repo to LeanGitRepo.
    """
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                data.append(data_)

        if 'valid' in split.lower():
            data = [x for x in data if x['split'] != 'test']
        else:
            data = [x for x in data if x['split'] == 'test']
        repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
    else:
        raise NotImplementedError(split)

    return repo, data

# Set the repo
repo, data = load_data_dojo(dataset_path='./data/minif2f_lean4_dojo.jsonl',
                            split='valid')

# Check the error message
for example in data:

    # Set up the data
    file_path = example['file_path']
    theorem_name = example.get('full_name', example.get('id'))
    theorem = Theorem(repo, file_path, theorem_name)

    # Set the environment
    with Dojo(theorem, hard_timeout=600) as (dojo, init_state):
        breakpoint()
