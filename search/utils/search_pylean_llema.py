"""
Reference: https://github.com/wellecks/ntptutorial/blob/main/partI_nextstep/ntp_python/proofsearch_pylean.py
"""

# Utilities for interacting with Lean and proof search

from accelerate import Accelerator
from pylean import LeanServer
import torch
import heapq
import concurrent
import transformers
import os
import logging
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# ------------------------------------------------
# Some prep
accelerator = Accelerator()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'  
logger = logging.getLogger()
# ------------------------------------------------


class DotDict(dict):
    """
    Dot notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

def is_done(state):
    return not state.get('sorries') and not state.get('messages')


def get_goal(state):
    goal = None
    msgs = state.get('messages')

    if msgs:
        for msg in msgs:
            
            if msg['data'].startswith('unsolved goals\n'):
                goal = '\n'.join(msg['data'].split('\n')[1:])
                
            elif msg['severity'] == 'error':
                return None
    return goal


def get_errors(state):
    return state['messages']


def parse_step(step):
    step = step.replace('<|endoftext|>', '')
    return step


def format_code(header, statement, steps_so_far, next_step):
    return header + (statement.replace(" {}", "") + '\n' + '\n'.join(steps_so_far + [next_step]))
    

def run_code(code):
    lean = LeanServer()
    out = lean.run_code(code)
    lean.proc.close()
    del lean
    return out


def sequence_scores(out, prompt_length, model, tokenizer):
    # Returns each output sequence's log probability normalized by the number of tokens.
    # An output sequence is defined as the tokens after the prompt up to and including eos.
    text = tokenizer.batch_decode(out.sequences)
    input_ids = tokenizer(
        text, return_tensors="pt", padding='longest', truncation=True
    ).to(model.device)
    with torch.no_grad():
        out = model(**input_ids)
        probs = torch.log_softmax(out.logits, dim=-1).detach()
        probs = probs[:, :-1, :]
        input_ids_shifted = input_ids.input_ids[:, 1:]
        log_probs = torch.gather(probs, 2, input_ids_shifted[:, :, None]).squeeze(-1)
        log_probs = log_probs[:, prompt_length:]
        up_to_eos_mask = (input_ids_shifted[:,prompt_length:].eq(
            tokenizer.eos_token_id).cumsum(1).cumsum(1) <= 1).type(log_probs.dtype)
        normalized_sequence_scores = (log_probs * up_to_eos_mask).sum(1) / up_to_eos_mask.sum(1)
    return normalized_sequence_scores


def generate(prompt, model, tokenizer, temperatures, num_samples) -> Tuple[List[str], List[float]]:
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    texts = []
    scores = []
    with torch.no_grad():
        # Does beam search at temp 0.0, otherwise temperature sampling.
        for temp in temperatures:
            decoding_params = dict(
                max_new_tokens=256,
                do_sample=temp > 0,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_samples,
                return_dict_in_generate=True,
                output_scores=True,
            )
            if temp == 0.0:
                decoding_params['num_beams'] = num_samples
            out = model.generate(
                input_ids, **decoding_params
            )
            
            texts.extend(tokenizer.batch_decode(
                out.sequences[:,input_ids.shape[1]:],
                skip_special_tokens=True
            ))
            scores_ = sequence_scores(
                out=out, 
                prompt_length=input_ids.shape[1], 
                model=model, 
                tokenizer=tokenizer
            )
            scores.extend(scores_.view(-1).tolist())

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


def _unique_sorted(texts, scores):
    texts_, scores_ = [], []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_


def _print_type_checked_candidates(results):
    logger.info('--- type-checked candidates:\n\t' + '\n\t'.join(
        '(%.3f) %s' % (step_score, step) 
        for state, step, step_score in results if (
        get_goal(state) is not None or is_done(state))
    ))


def _print_current(theorem_statement, steps):
    logger.info('--- current:\n\t%s\n\t%s' % (
        theorem_statement.replace('{}', ''),
        '\n\t'.join(steps)) 
    )


def simple_search(model, tokenizer, header, statement, search_budget):
    logger.info(header)
    logger.info(statement)
    success = False

    code = header + statement
    steps = []
    proof = ''

    for i in range(search_budget):
        # ---------------------------------------------------------------- #
        # Get the goal information
        # Print the current result
        logger.info(f"== Current ({i}): \n{statement[:-3]}\n{proof}")

        # Get the current state
        state = run_code(code)
        
        # Stop if the proof is complete.
        if is_done(state):
            success = True
            break

        # Get the current goal
        goal_candidate = get_goal(state)
        if goal_candidate is None:
            logger.info("-- Error: backtracking")
            steps = steps[:-1]
        else:
            goal = goal_candidate

        logger.info(f"-- Goal: \n{goal}")
        # ----------------------------------------------------------------

        # ---------------------------------------------------------------- #
        # Generate a next-step
        prompt = f"[GOAL]{goal}[PROOFSTEP]"
        texts, _= generate(prompt, model, tokenizer, temperatures=[0.5], num_samples=1)
        step = parse_step(texts[0])
        # ----------------------------------------------------------------

        # ---------------------------------------------------------------- #
        # Add the next-step to the proof-so-far
        steps.append(step)
        proof = '\n'.join(steps)
        code = header + statement.replace(" {}", "") + '\n' + proof
        # ----------------------------------------------------------------

    if success: logger.info("\nSUCCESS!")
    else: logger.info("\nFAILED")
    
    logger.info(statement.replace(' {}', ''))
    logger.info('  ' + proof.replace('\n', '\n  '))
    
    return {'theorem_statement': statement, 'proof': proof, 'success': success}


def best_first_search(model, tokenizer, header, statement, max_iters, 
                      temperatures, num_samples, verbose=False) -> dict:
    
    # ---------------------------------------------------------------- #
    # GET THE GOAL INFORMATION
    goal = get_goal(run_code(header + statement))
    if goal is None:
        return {
            'theorem_statement': statement, 
            'success': False, 
            'msg': run_code(header + statement)
        }
    # ----------------------------------------------------------------

    # ---------------------------------------------------------------- #
    # GENERATE CANDIDATES
    # Score, steps-so-far, goal state
    queue = [(0.0, [], goal)]
    visited = set()
    while len(queue) > 0 and max_iters > 0:
        
        # Dequeue the tuple with the minimum score
        score, steps, goal = heapq.heappop(queue)
        visited.add(goal)
        if verbose:
            _print_current(statement, steps)

        # Generate next-step candidates
        prompt = f'[GOAL]{goal}[PROOFSTEP]'
        step_cands, step_scores = generate(
            prompt, 
            model, 
            tokenizer, 
            temperatures=temperatures, 
            num_samples=num_samples
        )

        # Run type checking in parallel via futures. 
        with ThreadPoolExecutor(max_workers=16) as executor:
            # We need to save the step and score associated to each future.
            future2step = {}
            for step, step_score in zip(step_cands, step_scores):
                code = format_code(header, statement, steps, step)
                future = executor.submit(run_code, **dict(code=code))
                future2step[future] = (step, step_score)

            # Collect the type checking results as they complete.
            results = []
            for future in tqdm(concurrent.futures.as_completed(future2step.keys()), 
                               total=len(future2step)):
                result = future.result()
                results.append((result, *future2step[future]))

        if verbose:
            _print_type_checked_candidates(results)
            
        for state, step, step_score in results:
            # Stop if we have found a complete proof.
            if is_done(state):
                return {
                    'theorem_statement': statement, 
                    'proof': steps + [step], 
                    'state': state,
                    'score': score - step_score,
                    'success': True
                }
            goal_cand = get_goal(state)
            # Add new candidates to the queue.
            if goal_cand is not None and goal_cand not in visited:
                # Score is normalized negative log probability summed across steps
                new_score = (score - step_score)
                heapq.heappush(
                    queue, (new_score, steps+[step], goal_cand)
                )
        
        max_iters -= 1

    return {'theorem_statement': statement, 'success': False}


def _save(results):
    from datetime import datetime
    import json
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")
    output_file = 'results__%s.json' % (dt_string)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
        logger.info(output_file)


def load_model(model_name):
    if 'pythia' in model_name:
        model = transformers.GPTNeoXForCausalLM.from_pretrained(
            model_name,
            device_map='auto')
        tokenizer = transformers.GPTNeoXTokenizerFast.from_pretrained(model_name)
    else:
        # Set the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
        tokenizer.pad_token = tokenizer.eos_token

        # Set the model
        model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map='auto',
                    low_cpu_mem_usage=True)

        # Set the eos
        model.config.pad_token_id = model.config.eos_token_id
    
    # Prepare the model
    model = accelerator.prepare(model)
    
    # Set the model in eval mode (to edit later)
    model.eval()
    return model, tokenizer
