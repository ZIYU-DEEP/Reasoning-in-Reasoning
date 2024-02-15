"""
Using Lean-Dojo as the environment.
Currently support VLLM and Accelerate.
Tested on Llemma.
"""

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


# -----------------------------------------------
# PREP

# accelerator = Accelerator()
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
DOJO_ERROR = (DojoInitError, DojoHardTimeoutError, DojoCrashError, subprocess.CalledProcessError)
logger = logging.getLogger()


class DotDict(dict):
    """
    Dot notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print_stats(results):
    print(len([x for x in results if x['success']]) / len(results))
    print("# successes: ", len([x for x in results if x['success']]), sep="\t")


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
# -----------------------------------------------


# -----------------------------------------------
# ENV
def _tactic_state(state):
    if isinstance(state, TacticState):
        ts = state.pp
    else:
        ts = state.unsolved_tactic_state
    return ts


def _load_data(dataset_name: str='minif2f-test',
               dataset_path: str='./data/minif2f.jsonl'):
    if 'minif2f' in dataset_name:
        data = []
        with open(dataset_path) as f:
            for line in f.readlines():
                data_ = json.loads(line)
                assert data_['commit'] == 'd00c776260c77de7e70125ef0cd119de6c0ff1de'
                data.append(data_)

        if 'valid' in dataset_name:
            data = [x for x in data if x['split'] == 'valid']
        else:
            data = [x for x in data if x['split'] == 'test']
        repo = LeanGitRepo(data[0]['url'], data[0]['commit'])
    else:
        raise NotImplementedError(dataset_name)

    return repo, data
# -----------------------------------------------


# -----------------------------------------------
# PROMPT

def _prompt_fewshot(tactic_state):
    prompt = """Given the Lean 4 tactic state, suggest a next tactic.
Here are some examples:

Tactic state:
----
α : Type u_1
r : α → α → Prop
inst✝¹ : DecidableEq α
inst✝ : IsIrrefl α r
⊢ CutExpand r ≤ InvImage (Finsupp.Lex (rᶜ ⊓ fun x x_1 => x ≠ x_1) fun x x_1 => x < x_1) ↑toFinsupp
----
Next tactic:
----
rintro s t ⟨u, a, hr, he⟩
----

Tactic state:
----
ι : Type u_1
I✝ J✝ : Box ι
x y : ι → ℝ
I J : WithBot (Box ι)
⊢ ↑I = ↑J ↔ I = J
----
Next tactic:
----
simp only [Subset.antisymm_iff, ← le_antisymm_iff, withBotCoe_subset_iff]
----

Tactic state:
----
m n : ℕ
h : Nat.coprime m n
⊢ Nat.gcd m n = 1
----
Next tactic:
----
rw [← h.gcd_eq_one]
----

In your response, include only the lean code for only the next tactic and nothing else.
Tactic state:
----
%s
----
Next tactic:
----""" % (tactic_state)
    return prompt
# -----------------------------------------------


# -----------------------------------------------
# LOADING MODELS

def load_model_vllm(model_name: str='open-web-math/llemma_7b',
                    tp_degree: int=1,
                    dtype: str='float16',
                    max_num_batched_tokens: int=4096):

    model = vllm.LLM(
        model=model_name,
        tensor_parallel_size=tp_degree,
        dtype=dtype,
        max_num_batched_tokens=max_num_batched_tokens,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def load_model_hf(model_name, accelerator=None):
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
# -----------------------------------------------


# -----------------------------------------------
# GENERATION

## Helper
def _unique_sorted(texts, scores):
    """
    Sort the texts according to log prob.
    """
    texts_, scores_ = [], []
    for t, s in sorted(zip(texts, scores), key=lambda x: -x[1]):
        if t not in texts_:
            texts_.append(t)
            scores_.append(s)
    return texts_, scores_

## For VLLM
def generate_vllm(prompt, model, tokenizer, temperatures, num_samples, stop, max_tokens=256):

    # Init textx and scores
    texts, scores = [], []

    for temperature in temperatures:
        # Set the params
        params = vllm.SamplingParams(
            n=num_samples,
            temperature=temperature,
            use_beam_search=temperature==0.0,
            max_tokens=max_tokens,
            stop=stop,
        )

        # Get the outputs
        outputs = model.generate([prompt], params, use_tqdm=False)
        if len(outputs) == 0:
            return [], []

        # Get texts and scores
        for output in outputs[0].outputs:
            text = output.text.replace(tokenizer.eos_token, '')
            score = output.cumulative_logprob/max(len(output.token_ids), 1)
            texts.append(text)
            scores.append(score)

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores


## For HF
def trim_output(output_text, stop_div='----', trim_at_last=False):
    """
    Trims the generated output text to remove the stop sequence and everything after it. Can trim at the first
    or last occurrence of the stop sequence based on the trim_at_last flag.

    Parameters:
    - output_text (str): The generated text.
    - stop_div (str): The sequence after which the text should be trimmed.
    - trim_at_last (bool): Flag to trim at the last occurrence of stop_div. Default is False.

    Returns:
    - str: The trimmed text.
    """
    if trim_at_last:
        stop_index = output_text.rfind(stop_div)  # Finds the last occurrence
    else:
        stop_index = output_text.find(stop_div)  # Finds the first occurrence

    if stop_index != -1:
        return output_text[:stop_index]
    else:
        return output_text


def sequence_scores(out, prompt_length, model, tokenizer, stop_div='----'):
    """
    Returns each output sequence's log probability normalized by the number of tokens.
    An output sequence is defined as the tokens after the prompt up to and including eos.
    """

    # Get the text
    # TODO: The text should be trimmed
    text = tokenizer.batch_decode(out.sequences)

    # Trim the text
    # Unlike VLLM, the stop words will be retained, so we have to manually remove
    # This part may still need some improvement
    for i, text_i in enumerate(text):
        text[i] = trim_output(text_i,
                              stop_div='----',
                              trim_at_last=True).strip()
        text[i] += '</s>'

    input_ids = tokenizer(
        text, return_tensors="pt", padding='longest', truncation=True
    ).to(model.device)

    with torch.no_grad():
        # Get the probs
        out = model(**input_ids)
        probs = torch.log_softmax(out.logits, dim=-1).detach()
        probs = probs[:, :-1, :]

        # Get the probs after the prompt
        input_ids_shifted = input_ids.input_ids[:, 1:]
        log_probs = torch.gather(probs, 2, input_ids_shifted[:, :, None]).squeeze(-1)
        log_probs = log_probs[:, prompt_length:]
        up_to_eos_mask = (input_ids_shifted[:,prompt_length:].eq(
            tokenizer.eos_token_id).cumsum(1).cumsum(1) <= 1).type(log_probs.dtype)

        # Normalize the scores
        normalized_sequence_scores = (log_probs * up_to_eos_mask).sum(1) / up_to_eos_mask.sum(1)

    return normalized_sequence_scores


# class StoppingCriteriaSub(StoppingCriteria):
# # In this case:
# stop_words = ['----', '----\n', '\n----']
# stop_words_ids = [tokenizer(stop_word, return_tensors='pt', 
#                             add_special_tokens=False)['input_ids'].squeeze() 
#                   for stop_word in stop_words]

# stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, tokenizer=tokenizer)])
# 
#     def __init__(self, stops = [], encounters=1, tokenizer=None):
#         super().__init__()
#         self.stops = [stop.to('cuda') for stop in stops]
#         self.tokenizer = tokenizer

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         last_token = input_ids[0][-1]
#         # print(tokenizer.decode(last_token), last_token, len(input_ids[0]))
#         for stop in self.stops:
#             if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
#                 return True
#         return False

class StoppingCriteriaSub(StoppingCriteria):
    """
    Notice this stopping criteria is specifically configured.
    
    We count the occurence of '----' in the few-shot prompt, and stop at the first occurence
    in the generated text, which shold be 16.
    
    It is slightly troublesome that '----' is not always encoded as the same thing.
    
    Usage:
    stop_words_ids = [torch.tensor([807])]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    """

    def __init__(self, stops=[], encounters=16):
        super().__init__()
        self.stops = [stop.to('cuda') for stop in stops]
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        
        for stop in self.stops:
            stop_count = (input_ids[0] == stop).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
            
        return False


def generate_hf(prompt, model, tokenizer, temperatures, num_samples, stopping_criteria):

    # Init texts and scores
    texts, scores = [], []

    # Get the input ids
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    # Generate
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
                stopping_criteria=stopping_criteria,
            )
            if temp == 0.0:
                decoding_params['num_beams'] = num_samples

            # Get the output
            out = model.generate(
                input_ids, **decoding_params
            )

            # Get the texts
            # TODO: Apply the text trimmining
            decoded_seqs = tokenizer.batch_decode(
                out.sequences[:,input_ids.shape[1]:],
                skip_special_tokens=True
            )

            # Remove that '----'
            decoded_seqs = [trim_output(text,
                                 stop_div='----').strip()
                            for text in decoded_seqs]

            # Extend to the texts
            texts.extend(decoded_seqs)
            # pprint(texts)

            # Get the scores
            scores_ = sequence_scores(
                out=out,
                prompt_length=input_ids.shape[1],
                model=model,
                tokenizer=tokenizer
            )
            scores.extend(scores_.view(-1).tolist())

    texts, scores = _unique_sorted(texts, scores)
    return texts, scores
# -----------------------------------------------


# -----------------------------------------------
# SEARCH
def best_first_search(
        theorem,
        model,
        tokenizer,
        max_iters,
        temperatures,
        num_samples,
        prompt_fn,
        timeout=600,
        early_stop=False,
        max_tokens=256,
        stopping_criteria=None,
        gen_method='vllm',
) -> dict:
    """
    Best first search.
    """
    # Initialize the results
    attempt_results = []

    try:
        with Dojo(theorem, hard_timeout=timeout) as (dojo, init_state):

            # ------------------------------------------------
            # PREPARATION
            start = time.time()
            proof_finished = False
            queue = [(0.0, [], init_state, [])]
            visited = set()
            # ------------------------------------------------

            # ------------------------------------------------
            # STEP BY STEP INFERENCE
            for iteration in trange(max_iters):

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
                assert gen_method in ['vllm', 'hf']
                if gen_method == 'vllm':
                    step_cands, step_scores = generate_vllm(
                        prompt_fn(ts),
                        model,
                        tokenizer,
                        temperatures,
                        num_samples,
                        stop='----',
                        max_tokens=max_tokens
                    )

                elif gen_method == 'hf':
                    step_cands, step_scores = generate_hf(
                        prompt=prompt_fn(ts),
                        model=model,
                        tokenizer=tokenizer,
                        temperatures=temperatures,
                        num_samples=num_samples,
                        stopping_criteria=stopping_criteria)

                step_cands = [s.strip() for s in step_cands]
                # ---------------------------------------------

                # ---------------------------------------------
                # Update the queue
                for step, score in zip(step_cands, step_scores):
                    result = dojo.run_tac(state, step)
                    step_trace = {
                        'tactic': step,
                        'state_before': _tactic_state(state)
                    }
                    # logger.info(step)

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
                                (new_score, steps + [step], result, trace + [step_trace])
                            )
                            logger.info(f'\nstep: {step}; score: {round(score, 3)}')
                # ---------------------------------------------

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
            'failure_reason': 'SearchEnded'
        })
        logger.info('Search ended with no success.')

    return attempt_results
# -----------------------------------------------

# -----------------------------------------------
# MCTS
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
               stopping_criteria=None,
               gen_method='vllm') -> dict:

    # =========================================================
    # DEFINE NODE EVALUATOR AND CHILD FINDER FOR MCTS
    def node_evaluator(child, montecarlo):
        if child.proof_finished:
            return 1
        else:
            return - 1 
    
    def child_finder(node, montecarlo):
        """
        Add type checked child_node to the current node.
        """

        # ---------------------------------------------
        # Get the next step candidates
        string_state = _tactic_state(node.state)
        prompt = prompt_fn(node.state)
        
        assert gen_method in ['vllm', 'hf']
        if gen_method == 'vllm':
            step_cands, step_scores = generate_vllm(
                prompt,
                model=model,
                tokenizer=tokenizer,
                temperatures=temperatures,
                num_samples=num_samples,
                stop='----',
                max_tokens=max_tokens
            )

        elif gen_method == 'hf':
            step_cands, step_scores = generate_hf(
                prompt,
                model=model,
                tokenizer=tokenizer,
                temperatures=temperatures,
                num_samples=num_samples,
                stopping_criteria=stopping_criteria)

        step_cands = [s.strip() for s in step_cands]
        # ---------------------------------------------

        # ---------------------------------------------
        # Find the valid child
        for step, score in zip(step_cands, step_scores):
            # Apply the step to the current state
            new_state = dojo.run_tac(state, step)
        
            # Only type checked node is added
            if isinstance(new_state, (ProofFinished, TacticState)):
                # Init the node
                new_state_str = _tactic_state(new_state)
                child_node = Node(new_state_str)

                # Update the score
                child_node.score = score
                
                # Special handling for ProofFinished state
                if isinstance(new_state, ProofFinished):
                    child_node.proof_finished = True
                    montecarlo.solution = True
                
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
            init_state_str = _tactic_state(init_state)

            # Initialize the root node
            root_node = Node(init_state_str)

            # Initialize the monte carlo class
            montecarlo = MonteCarlo(root_node, mins_timeout=timeout / 60)

            # Set the child finder and the node evaluator
            montecarlo.child_finder = child_finder
            montecarlo.node_evaluator = node_evaluator

            # Run the simulation
            montecarlo.simulate(expansion_count=max_iters)

            # At the end of mct_search function, after montecarlo.simulate(expansion_count=max_iters)
            solution_node = find_solution_node(montecarlo.root_node)
            if solution_node:
                proof_steps = format_solution(solution_node)
                attempt_results.append({
                    'theorem': theorem,  # Adjust according to how you identify theorems
                    'success': True,
                    'proof_steps': proof_steps,
                    'details': {
                        'iterations': max_iters,
                        'timeout': timeout
                    }
                })
            else:
                attempt_results.append({
                    'theorem': theorem,
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