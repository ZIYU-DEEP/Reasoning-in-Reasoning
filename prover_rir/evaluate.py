"""Script for evaluating the prover on theorems extracted by LeanDojo.
"""

import os
import uuid
import json
import pickle
import hashlib
import argparse
from loguru import logger
from lean_dojo import Theorem
from typing import List, Tuple, Optional
from lean_dojo import LeanGitRepo, Theorem, Pos, is_available_in_cache

# from common import set_logger
from common import set_logger
from prover_rir.proof_search import Status, DistributedProver


def _get_theorems(
    data_path: str,
    split: str,
    file_path: str,
    full_name: str,
    name_filter: str,
    num_theorems: int,
    start_ind: int=0,
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    """
    Retrieves theorems from the specified data source.

    Args:
        data_path (str): Path to the data extracted by LeanDojo.
        split (str): The data split to use (e.g., 'val', 'test').
        file_path (str): Filter theorems by the specified file path.
        full_name (str): Filter theorems by the specified full name.
        name_filter (str): Filter theorems by the specified name filter.
        num_theorems (int): The maximum number of theorems to retrieve.

    Returns:
        Tuple[LeanGitRepo, List[Theorem], List[Pos]]: 
            A tuple containing the repository, 
                               list of theorems, 
                               and list of positions.
    """
    repo, theorems, positions = _get_theorems_from_files(
        data_path=data_path,
        split=split,
        file_path=file_path,
        full_name=full_name,
        name_filter=name_filter,
        num_theorems=num_theorems,
        start_ind=start_ind,
    )

    # Check if all theorem's repos are available in the cache
    all_repos = {thm.repo for thm in theorems}
    for r in all_repos:
        assert is_available_in_cache(
            r
        ), f"{r} has not been traced yet. Please use LeanDojo to trace it so that it's available in the cache."

    return repo, theorems, positions


def _get_theorems_from_files(
    data_path: str,
    split: str,
    file_path: Optional[str],
    full_name: Optional[str],
    name_filter: Optional[str],
    num_theorems: Optional[int],
    start_ind: int=0,
) -> Tuple[LeanGitRepo, List[Theorem], List[Pos]]:
    """
    Helper function for _get_theorems.
    
    Returns:
        Tuple[LeanGitRepo, List[Theorem], List[Pos]]: 
            A tuple containing the repository, 
                               list of theorems, 
                               and list of positions.
                               
        The positions is used in model.PremiseRetriever.retrive 
        to provide the context for retrieving the premises.
    """
    # Load theorems from JSON file
    data = json.load(open(os.path.join(data_path, f"{split}.json")))
    theorems = []
    positions = []

    # Filter theorems
    for t in data:
        if file_path is not None and t["file_path"] != file_path:
            continue
        if full_name is not None and t["full_name"] != full_name:
            continue
        if name_filter is not None and not hashlib.md5(
            t["full_name"].encode()
        ).hexdigest().startswith(name_filter):
            continue
        
        repo = LeanGitRepo(t["url"], t["commit"])
        theorems.append(Theorem(repo, t["file_path"], t["full_name"]))
        positions.append(Pos(*t["start"]))

    # Jointly sort theorems and positions
    theorems_and_positions = list(zip(theorems, positions))
    theorems_and_positions.sort(
        key=lambda x: hashlib.md5(
            f"{x[0].file_path}:{x[0].full_name}".encode()
        ).hexdigest()
    )
    theorems, positions = zip(*theorems_and_positions)
    theorems, positions = list(theorems), list(positions)
    
    # Slice theorems and positions based on start index
    if start_ind:
        theorems = theorems[start_ind:]
        positions = positions[start_ind:]
    start_theorem_name = theorems[0].full_name
    logger.info(f"Starting from {start_ind}th theorem named {start_theorem_name}!")

    # Limit the number of theorems if specified
    if num_theorems is not None:
        theorems = theorems[:num_theorems]
        positions = positions[:num_theorems]
    logger.info(f"{len(theorems)} theorems loaded from {data_path}")

    # Load repository metadata
    metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
    repo = LeanGitRepo(metadata["from_repo"]["url"], metadata["from_repo"]["commit"])

    return repo, theorems, positions


def evaluate(
    data_path: str,
    exp_id: Optional[str] = None,
    split: str = "val",
    file_path: Optional[str] = None,
    full_name: Optional[str] = None,
    name_filter: Optional[str] = None,
    num_theorems: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    indexed_corpus_path: Optional[str] = None,
    tactic: Optional[str] = None,
    module: Optional[str] = None,
    num_sampled_tactics: int = 64,
    timeout: int = 600,
    num_workers: int = 1,
    num_gpus: int = 0,
    verbose: bool = False,
    start_ind: int = 0,
    gen_type: str="goal_driven_tactic",
    goal_ckpt_path: Optional[str] = None,  # added for rir
    num_sampled_goals: int = 10,  # added for rir
) -> float:
    """
    Evaluates the prover on the specified theorems.

    Returns:
        float: The pass@1 metric (proportion of proved theorems).
    """
    # set_logger(verbose)

    repo, theorems, positions = _get_theorems(
        data_path=data_path, 
        split=split, 
        file_path=file_path, 
        full_name=full_name, 
        name_filter=name_filter, 
        num_theorems=num_theorems,
        start_ind=start_ind,
    )

    # Search for proofs using multiple concurrent provers.
    prover = DistributedProver(
        ckpt_path=ckpt_path,
        indexed_corpus_path=indexed_corpus_path,
        tactic=tactic,
        module=module,
        num_workers=num_workers,
        num_gpus=num_gpus,
        timeout=timeout,
        num_sampled_tactics=num_sampled_tactics,
        debug=verbose,
        gen_type=gen_type,
        goal_ckpt_path=goal_ckpt_path,
        num_sampled_goals=num_sampled_goals,
    )
    results = prover.search_unordered(repo, theorems, positions)

    # Calculate the result statistics.
    num_proved = num_failed = num_discarded = 0
    for r in results:
        if r is None:
            num_discarded += 1
        elif r.status == Status.PROVED:
            num_proved += 1
        else:
            num_failed += 1

    logger.info(
        f"Evaluation done! {num_proved} theorems proved, {num_failed} theorems failed, {num_discarded} non-theorems discarded"
    )

    if num_proved + num_failed == 0:
        pass_1 = float("nan")
    else:
        pass_1 = num_proved / (num_proved + num_failed)

    # Save the results.
    if exp_id is None:
        exp_id = str(uuid.uuid4())
    os.makedirs(f"./results/{gen_type}", exist_ok=True)
    pickle_path = f"./results/{gen_type}/{exp_id}_results.pickle"
    pickle.dump(results, open(pickle_path, "wb"))
    logger.info(f"Results saved to {pickle_path}")

    return pass_1


def main() -> None:
    """
    The main function that parses command-line arguments and runs the evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Script for evaluating the prover on theorems extracted by LeanDojo."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the data extracted by LeanDojo (e.g., data/leandojo_benchmark/random).",
    )
    parser.add_argument("--exp-id", type=str, help="Experiment ID used for logging.")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
    )
    # `file_path`, `full_name`, `name_filter`, and `num_theorems` can be used to filter theorems.
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--full-name", type=str)
    parser.add_argument("--name-filter", type=str)
    parser.add_argument("--num-theorems", type=int)
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Checkpoint of the tactic generator.",
    )
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        help="Path to a pickled indexed corpus. Not required for models w/o retrieval.",
    )
    parser.add_argument("--tactic", type=str, help="The tactic to evaluate.")
    parser.add_argument("--module", type=str, help="The module to import the tactic.")
    parser.add_argument(
        "--num-sampled-tactics",
        type=int,
        default=64,
        help="Number of tactics to sample at each node during proof search.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum number of seconds the proof search can take.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="The number of concurrent provers."
    )
    parser.add_argument(
        "--num-gpus", type=int, default=0, help="The number of GPUs for proof search."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Set the logging level to DEBUG."
    )
    
    # New arguments added by neurips ddl
    parser.add_argument("--start-ind", type=int, default=0, help="The starting index of theorems to evaluate.")
    parser.add_argument("--gen-type", type=str, default='default',
                        choices=['default', 'goal_driven_tactic', 'goal', 'joint'],
                        help="The type for the generator.")
    
    parser.add_argument(
        "--goal-ckpt-path",
        type=str,
        help="Checkpoint of the goal generator.",
    )
    parser.add_argument(
        "--num-sampled-goals",
        type=int,
        default=10,
        help="Number of goals to sample at each node during proof search.",
    )
    

    args = parser.parse_args()
    log_filename_suffix = f'{args.gen_type}-{args.start_ind}-{args.num_theorems}-{args.num_sampled_goals}'
    set_logger(verbose=args.verbose, filename_suffix=f'{log_filename_suffix}')

    assert args.ckpt_path or args.tactic
    assert args.num_gpus <= args.num_workers

    logger.info(f"PID: {os.getpid()}")
    logger.info(args)

    pass_1 = evaluate(
        args.data_path,
        args.exp_id,
        args.split,
        args.file_path,
        args.full_name,
        args.name_filter,
        args.num_theorems,
        args.ckpt_path,
        args.indexed_corpus_path,
        args.tactic,
        args.module,
        args.num_sampled_tactics,
        args.timeout,
        args.num_workers,
        args.num_gpus,
        args.verbose,
        args.start_ind,
        args.gen_type,
        args.goal_ckpt_path,
        args.num_sampled_goals,
    )

    logger.info(f"Pass@1: {pass_1}")


if __name__ == "__main__":
    main()
