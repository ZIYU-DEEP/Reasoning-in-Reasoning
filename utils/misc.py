import json
import logging
from lean_dojo import LeanGitRepo


class DotDict(dict):
    """
    Dot notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_logger(log_path, level=logging.INFO):
    """
    Set up a logger for use.
    """
    # Config the logging
    logging.basicConfig(level=level)
    logger = logging.getLogger()

    # Set level and formats
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Record logging
    logger.info(log_path)

    return logger


# def load_data_dojo(dataset_name: str='minif2f-test',
#                    dataset_path: str='./data/minif2f_lean4.jsonl'):
#     """
#     Load the data and the repo to LeanGitRepo.
#     """
#     if 'minif2f' in dataset_name:
#         data = []
#         with open(dataset_path) as f:
#             for line in f.readlines():
#                 data_ = json.loads(line)
#                 # assert data_['commit'] == 'd00c776260c77de7e70125ef0cd119de6c0ff1de'
#                 data.append(data_)

#         if 'valid' in dataset_name:
#             data = [x for x in data if x['split'] == 'valid']
#         else:
#             data = [x for x in data if x['split'] == 'test']
#         repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
#     else:
#         raise NotImplementedError(dataset_name)

#     return repo, data

def load_data_dojo(dataset_name: str='minif2f',
                   dataset_path: str='./data/minif2f_lean4.jsonl',
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