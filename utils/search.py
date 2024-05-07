"""
Using Lean-Dojo as the environment.
Currently support VLLM and Accelerate.
Tested on Llemma.

Test
"""

from .llms import *
import torch
import heapq
import concurrent
import transformers
import os
import json
import time
from datetime import datetime
import logging
import numpy as np
from .searchlight_models import *
from searchlight.datastructures.graphs import ValueGraph2
from searchlight.algorithms.best_first_search import BestFirstSearch
from searchlight.algorithms.mcts_search import SMMonteCarlo

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

from tqdm import tqdm, trange
from pprint import pprint

from .mcts import (
    MonteCarlo, 
    Node,
    limit_depth,
    stats,
    select_with_scores,
    create_score_predicate,
)

try:
    import vllm
except Exception:
    pass


# ===============================================
# Preparations
# ===============================================
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
DOJO_ERROR = (DojoInitError, 
              DojoHardTimeoutError, 
              DojoCrashError, 
              subprocess.CalledProcessError)
logger = logging.getLogger()


def _tactic_state(state):
    """
    Return the string state from the state.
    """
    if isinstance(state, TacticState):
        ts = state.pp
    elif isinstance(state, ProofFinished):
        ts = ''
    else:
        ts = state.unsolved_tactic_state
    return ts


# ===============================================
# DOJO WRAPPER FOR PROOF SEARCH
# ===============================================
def proof_search(theorem, model, tokenizer, 
                 max_iters_low, max_iters_high, 
                 temperatures, 
                 num_samples_low, num_samples_high, 
                 search_fn,
                 prompt_fn_low=None, prompt_fn_high=None,
                 timeout=600, early_stop=False, max_tokens=256, 
                 stop='----', gen_method='vllm', 
                 formal_statement='', informal_statement='', plan_high='', search_algorithm='bfs'):
    """
    Wrapper function to set up search environment and delegate to a search function.
    """
    attempt_results = []

    try:
        with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):
            
            attempt_results = search_fn(
                dojo=dojo,
                init_state=init_state,
                theorem=theorem,
                model=model,
                tokenizer=tokenizer,
                max_iters_low=max_iters_low,
                max_iters_high=max_iters_high,
                temperatures=temperatures,
                num_samples_low=num_samples_low,
                num_samples_high=num_samples_high,
                prompt_fn_low=prompt_fn_low,
                prompt_fn_high=prompt_fn_high,
                early_stop=early_stop,
                max_tokens=max_tokens,
                stop=stop,
                gen_method=gen_method,
                formal_statement=formal_statement,
                informal_statement=informal_statement,
                plan_high=plan_high,
                search_method=search_algorithm
            )
            
    except DOJO_ERROR as e:
        if len(attempt_results) == 0:
            attempt_results.append({
                'theorem': theorem.full_name,
                'success': False,
                'failure_reason': type(e).__name__
            })
        logger.info('Crashed.')

    if len(attempt_results) == 0:
        attempt_results.append({
            'theorem': theorem.full_name,
            'success': False,
            'failure_reason': 'UnknownError'
        })
        logger.info('Search ended with no success.')

    return attempt_results

# ===============================================
# SEARCH FUNCTION 1: bfs_low
# ===============================================
def search_low(dojo, init_state, theorem, 
            model, tokenizer, 
            max_iters_low, max_iters_high, 
            temperatures, 
            num_samples_low=32, num_samples_high=4, 
            prompt_fn_low=None, prompt_fn_high=None,
            timeout=600, early_stop=False, max_tokens=256, 
            stop='----', gen_method='vllm', 
            formal_statement='', informal_statement='', 
            plan_high='', search_method='bfs',):
    """
    Implements Low-Level Best First Search (BFS) algorithm for theorem proving using the searchlight framework
    """

    # ------------------------------------------------
    # PREPARATION
    start = time.time()
    proof_finished = False
    attempt_results = []
    # Generate results
    assert gen_method in ['vllm', 'hf', 'openai']
    if gen_method == 'vllm': generate_fn = generate_vllm
    if gen_method == 'hf': generate_fn = generate_hf
    if gen_method == 'openai': generate_fn = generate_openai
    init_state = LeanDojoState(init_state)
    # ------------------------------------------------
    
    # create the inferencer
    initial_inferencer = LowLevelInferencer(dojo=dojo, gen_method=generate_fn, 
                                             prompt_fn_low=prompt_fn_low, model=model, 
                                             tokenizer=tokenizer, temperatures=temperatures, 
                                             num_samples_low=num_samples_low, stop=stop, 
                                             max_tokens=max_tokens, formal_statement=formal_statement, 
                                             informal_statement=informal_statement, plan_high=plan_high)
    
    # create the ValueGraph
    value_graph = ValueGraph2(players={0})

    if early_stop:
        early_stopping_threshold = {0:float('inf')}
    else:
        early_stopping_threshold = None

    # create search algorithm
    assert search_method in ['bfs', 'mcts']
    if search_method == 'bfs':
        search_algorithm = BestFirstSearch(initial_inferencer, node_budget=max_iters_low,cut_cycles=True)
    elif search_method == 'mcts':
        search_algorithm = SMMonteCarlo(initial_inferencer, num_rollout=max_iters_low, node_budget=max_iters_low, cut_cycles=True, early_stopping_threshold = early_stopping_threshold)

    # run the search
    search_algorithm.expand(datastructure=value_graph, state=init_state)

    optimal_trajectory = []
    action_sequence = []

    # find the state in the graph that is ProofFinished, if any
    proof_finished_state = None
    for state in value_graph.id_to_node.keys():
        if isinstance(state.get_tactic_state(), ProofFinished):
            proof_finished_state = state
            break
    
    if proof_finished_state is not None:
        # get the optimal trajectory by backtracking from the ProofFinished state
        optimal_trajectory = value_graph.get_backtrack_path(proof_finished_state)

        # log the trajectory
        logger.info(f"Optimal Trajectory: {optimal_trajectory}")

        # get action sequence
        action_sequence = [dict(action)[0] for state, action in optimal_trajectory[:-1]]

        # log the action sequence
        logger.info(f"Action Sequence: {action_sequence}")

        proof_finished = True

        # convert the states in optimal trajectory to strings
        optimal_trajectory = [(str(state.get_string()), str(action)) for state, action in optimal_trajectory]

    attempt_results.append({
                    'theorem': theorem.full_name,
                    'proof': action_sequence,
                    'score': proof_finished,
                    'success': proof_finished,
                    'failure_reason': '',
                    'trace': optimal_trajectory,
                    'temperature': temperatures,
                    'elapsed': start - time.time(),
                    'iteration': search_algorithm.get_nodes_expanded(),})

    # ------------------------------------------------
    

    # The exception handling and attempt result management will be in `proof_search`
    return attempt_results



# ===============================================
# SEARCH FUNCTION 1: bfs_low
# ===============================================
def bfs_low_old(dojo, init_state, theorem, 
            model, tokenizer, 
            max_iters_low, max_iters_high, 
            temperatures, 
            num_samples_low=32, num_samples_high=4, 
            prompt_fn_low=None, prompt_fn_high=None,
            timeout=600, early_stop=False, max_tokens=256, 
            stop='----', gen_method='vllm', 
            formal_statement='', informal_statement='', plan_high='', 
            search_method='bfs',):
    """
    Implements Low-Level Best First Search (BFS) algorithm for theorem proving.
    """
    
    attempt_results = []

    # ------------------------------------------------
    # PREPARATION
    start = time.time()
    proof_finished = False
    queue = [(0.0, [], init_state, [])]
    visited = set()
    # ------------------------------------------------
    

    # ------------------------------------------------
    # STEP BY STEP INFERENCE
    for iteration in trange(max_iters_low):
        
        # ---------------------------------------------
        # Preparation
        # Termination criteria
        if len(queue) == 0 or proof_finished: break

        # Get the information from the heapq
        total_score, steps, state, trace = heapq.heappop(queue)
        ts = _tactic_state(state)
        logger.info(f'\nCurrent State:\n{state}\n')
        visited.add(ts)

        # ---------------------------------------------
        # Generate results
        assert gen_method in ['vllm', 'hf', 'openai']
        if gen_method == 'vllm': generate_fn = generate_vllm
        if gen_method == 'hf': generate_fn = generate_hf
        if gen_method == 'openai': generate_fn = generate_openai
        
        # NOTE: this is the action and heuristics generator
        step_cands, step_scores = generate_fn(
            prompt_fn_low(tactic_state=ts,
                          formal_statement=formal_statement,
                          informal_statement=informal_statement,
                          plan_high=plan_high),
            model=model,
            tokenizer=tokenizer,
            temperatures=temperatures,
            num_samples=num_samples_low,
            stop=stop,
            max_tokens=max_tokens
        )

        step_cands = [s.strip() for s in step_cands] 
        
        # DEBUG: Test smt
        step_cands = ['smt!'] + step_cands
        step_scores = [0.0] + step_scores  
        # This shouldn't cause change as smt gives either correct or error
        
        for step in step_cands:
            logger.info(step)
        # ---------------------------------------------

        # ---------------------------------------------
        # Update the queue
        for step, score in zip(step_cands, step_scores):
            # NOTE: this is the forward transitor
            result = dojo.run_tac(state, step)
            step_trace = {
                'tactic': step,
                'state_before': _tactic_state(state)
            }

            # When the proof is finished
            if isinstance(result, ProofFinished):
                attempt_results.append({
                    'theorem': theorem.full_name,
                    'proof': steps + [step],
                    'score': total_score - score,
                    'success': True,
                    'failure_reason': '',
                    'trace': trace + [step_trace],
                    'temperature': temperatures,
                    'elapsed': start - time.time(),
                    'iteration': iteration
                })
                if early_stop:
                    return attempt_results
                proof_finished = True

                logger.info(f'\nstep: {step}; score: {round(score, 3)}')
                logger.info('Congrats. Proof is finished for this theorem.')
                logger.info(attempt_results[-1]['proof'])

                break

            # When there is still unsolved goals
            elif isinstance(result, TacticState):
                if _tactic_state(result) not in visited:
                    # Score is negative log probability summed across steps
                    new_score = (total_score - score)
                    heapq.heappush(
                        queue,
                        (new_score, steps + [step], result, trace + [step_trace]) # 
                    )
                    logger.info(f'\nstep: {step}; score: {round(score, 3)}')
        # ---------------------------------------------

    # The exception handling and attempt result management will be in `proof_search`
    return attempt_results


# ===============================================
# SEARCH FUNCTION 2: bfs_bilevel
# ===============================================
def update_scores_based_on_success(initial_scores,
                                   plan_index,
                                   success,
                                   total_attempts):
    """
    Update scores based on whether the low-level proof was successful.
    This method now takes into account the initial likelihood scores,
    adjusting them by the outcome of attempting to generate a low-level proof.
    """
    adjustment = 1 if success else -1

    initial_scores[plan_index] += adjustment * np.sqrt(
        np.log(total_attempts + 1) / (1 + total_attempts))

    return initial_scores


def bfs_bilevel(dojo, init_state, theorem, 
                model, tokenizer, 
                max_iters_low, max_iters_high, 
                temperatures, 
                num_samples_low=32, num_samples_high=4, 
                prompt_fn_low=None, prompt_fn_high=None,
                timeout=600, early_stop=False, max_tokens=256, 
                stop='----', gen_method='vllm', 
                formal_statement='', informal_statement='', plan_high=''):
    """
    Implements Low-Level Best First Search (BFS) algorithm for theorem proving.
    """
    
    attempt_results = []

    # ------------------------------------------------
    # High-Level Search
    total_attempts = 0
    assert gen_method in ['vllm', 'hf', 'openai']
    if gen_method == 'vllm': generate_fn = generate_vllm
    if gen_method == 'hf': generate_fn = generate_hf
    if gen_method == 'openai': generate_fn = generate_openai
    
    high_level_plans, high_level_scores = generate_fn(
        prompt_fn_high(formal_statement=formal_statement,
                    informal_statement=informal_statement),
        model=model,
        tokenizer=tokenizer,
        temperatures=temperatures,
        num_samples=num_samples_high,
        stop=stop,
        max_tokens=max_tokens
    )
    # ------------------------------------------------
    
    # ------------------------------------------------
    # STEP BY STEP INFERENCE
    for iteration in trange(max_iters_high):
        
        # Calculate UCB values for each plan
        ucb_values = high_level_scores + np.sqrt(
            2 * np.log(total_attempts + 1) / (total_attempts + 1))
        
        # Print all high-level plans and their UCB values
        logger.info(f'\n\n======= High-Level Plans and UCB Values =======')
        for i, (plan, value) in enumerate(zip(high_level_plans, ucb_values)):
            logger.info(f'Plan {i} UCB Value: {np.round(np.exp(value), 2)}\n{plan}\n')
        
        # Select the high-level plan with the highest UCB value
        plan_to_try = np.argmax(ucb_values)
        selected_plan = high_level_plans[plan_to_try]
        logger.info(f'Selected High-Level Plan Index: {plan_to_try}')
        
        # Attempt low-level proof generation with the selected high-level plan
        attempt_results = bfs_low(
            dojo=dojo,
            init_state=init_state,
            theorem=theorem,
            model=model,
            tokenizer=tokenizer,
            max_iters_low=max_iters_low,
            max_iters_high=max_iters_high,  # This is redundant
            temperatures=temperatures,
            num_samples_low=num_samples_low,
            num_samples_high=num_samples_high, # This is redundant
            prompt_fn_low=prompt_fn_low,
            prompt_fn_high=prompt_fn_high,  # This is redundant
            timeout=timeout,
            early_stop=early_stop,
            max_tokens=max_tokens,
            stop=stop,
            gen_method=gen_method,
            formal_statement=formal_statement,
            informal_statement=informal_statement,
            plan_high=selected_plan  # Using the selected plan
        )
        
        # Update success status based on the low-level proof result
        try: success = attempt_results[-1]['success']
        except: success = False
        
        # Update scores based on success
        high_level_scores = update_scores_based_on_success(
            high_level_scores,
            plan_to_try,
            success,
            total_attempts)

        total_attempts += 1
        
        if success:
            break  # Exit loop if successful

    # The exception handling and attempt result management will be in `proof_search`
    return attempt_results


# ===============================================
# SEARCH FUNCTION 3: mcts_low (to edit)
# ===============================================
# Extraction and return of results as previously outlined
def format_solution(node):
    """
    Recursively trace back from the solution node to the root to construct the proof steps.
    """
    solution_steps = []
    current_node = node
    
    while current_node.parent is not None:  # Trace back to root
        solution_steps.append(current_node.state)
        current_node = current_node.parent
        
    solution_steps.reverse()  # Reverse to get the correct order
    return solution_steps

def find_solution_node(root_node):
    """
    Traverse the tree to find a node where proof_finished is True.
    """
    if root_node.proof_finished:
        return root_node
    
    for child in root_node.children:
        solution_node = find_solution_node(child)
        if solution_node:
            return solution_node
        
    return None  # No solution found in this branch


def mct_search(theorem,
               model,
               tokenizer,
               max_iters,
               temperatures,
               num_samples,
               prompt_fn,
               timeout=600,
               early_stop=False,
               max_tokens=256,
               stop='----',
               gen_method='vllm',
               expansion_count=1) -> dict:

    # =========================================================
    # DEFINE NODE EVALUATOR AND CHILD FINDER FOR MCTS
    def node_evaluator(child, montecarlo):
        if child.proof_finished:
            print('DEBUG: PROOF IS FINISHED!')
            return 1
        # else:
        #     return - 1 
        # DEBUG
        elif limit_depth(child, max_depth=max_iters):
           return -1
    
    def child_finder(node, montecarlo):
        """
        Add type checked child_node to the current node.
        Notice that we are using beam search to find the children.
        """

        # ---------------------------------------------
        # Get the next step candidates
        prompt = prompt_fn(node.state)

        assert gen_method in ['vllm', 'hf', 'openai']
        if gen_method == 'vllm': generate_fn = generate_vllm
        if gen_method == 'hf': generate_fn = generate_hf
        if gen_method == 'openai': generate_fn = generate_openai

        step_cands, step_scores = generate_fn(
            prompt,
            model=model,
            tokenizer=tokenizer,
            temperatures=temperatures,
            num_samples=num_samples,
            stop=stop,
            max_tokens=max_tokens
        )
        
        step_cands = [s.strip() for s in step_cands]
        # ---------------------------------------------

        # ---------------------------------------------
        # Find the valid child
        for step, score in zip(step_cands, step_scores):
            # Apply the step to the current state
            new_state = dojo.run_tac(node.dojo_state, step)
        
            # Only type checked node is added
            if isinstance(new_state, (ProofFinished, TacticState)):
                # Init the node
                child_node = Node(new_state)
                child_node.score = score
                
                # DEBUG
                logger.info(child_node.state)
                
                # Special handling for ProofFinished state
                if isinstance(new_state, ProofFinished):
                    child_node.proof_finished = True
                    montecarlo.solution = new_state  #TODO: Looks like this is not used
                    node.add_child(child_node)
                    node.proof_finished = True  #TODO: DEBUG
                    logger.info('Success!')
                    return
                           
                # Add the child node to the current node
                node.add_child(child_node)
    # =========================================================

    # Initialize the results
    attempt_results = []

    try:
        with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):

            # ------------------------------------------------
            # Get the state for the root node
            # Notice we use the string rep as the state
            # Instead of the dojo state itself

            # Initialize the root node
            root_node = Node(init_state)

            # Initialize the monte carlo class
            montecarlo = MonteCarlo(root_node, mins_timeout=timeout / 60)

            # Set the child finder and the node evaluator
            montecarlo.child_finder = child_finder
            montecarlo.node_evaluator = node_evaluator

            # Run the simulation
            # TODO: Use a different expansion count parameter
            # TODO: Return the result immediately when a solution is found
            montecarlo.simulate(expansion_count=expansion_count)

            # At the end of mct_search function, after montecarlo.simulate(expansion_count=max_iters)
            solution_node = find_solution_node(montecarlo.root_node)
            if solution_node:
                proof_steps = format_solution(solution_node)
                attempt_results.append({
                    'theorem': theorem.full_name,
                    'success': True,
                    'proof_steps': proof_steps,
                    'details': {
                        'iterations': max_iters,
                        'timeout': timeout
                    }
                })
            else:
                attempt_results.append({
                    'theorem': theorem.full_name,
                    'success': False,
                    'failure_reason': 'No solution found within the given constraints',
                    'details': {
                        'iterations': max_iters,
                        'timeout': timeout
                    }
                })

            return attempt_results

    except DOJO_ERROR as e:
        attempt_results = _record_results(attempt_results, theorem, logger, type(e).__name__)

    attempt_results = _record_results(attempt_results, theorem, logger, 'SearchEnded')
    
    return attempt_results



# ===============================================
# Additional Helpers
# ===============================================
def print_stats(results):
    print(len([x for x in results if x['success']]) / len(results))
    print("# successes: ", len([x for x in results if x['success']]), sep="\t")

def log_stats(results):
    logger.info(len([x for x in results if x['success']]) / len(results))
    logger.info("# successes: ", len([x for x in results if x['success']]), sep="\t")

def _record_results(attempt_results, theorem, logger, failure_reason):
    """
    Log the failure.
    """
    if len(attempt_results) == 0:
        attempt_results.append({
            'theorem': theorem.full_name,
            'success': False,
            'failure_reason': failure_reason
        }) 

    if failure_reason == 'SearchEnded':
        logger.info('Search ended with no success.')
    else:
        logger.info('Crashed.')

    return attempt_results
