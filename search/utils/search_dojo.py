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

    except (DojoInitError, DojoHardTimeoutError, DojoCrashError, subprocess.CalledProcessError) as e:
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

def apply_step_to_state(current_state, step, dojo):
    """
    Applies a generated step (tactic) to the current state and evaluates the outcome.
    
    Parameters:
    - current_state: The current state of the theorem proving process.
    - step: The tactic to apply to the current state.
    - dojo: The theorem proving environment or logic handler.
    
    Returns:
    - new_state: The state resulting from applying the step, or None if the step is invalid.
    - result: An instance indicating the outcome of the step application, which can be ProofFinished, TacticState, or None.
    """
    result = dojo.run_tac(current_state, step)
    if isinstance(result, ProofFinished):
        # Proof is successfully finished with this step
        return current_state, result  # Or however you represent the "finished" state
    elif isinstance(result, TacticState):
        # Step leads to a valid new state
        new_state = result  # Assuming result contains the new state information
        return new_state, result
    else:
        # Step is invalid or not applicable
        return None, None


def mct_search(
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
    def child_finder(node, montecarlo):
        prompt = prompt_fn(node.state)
        step_cands, step_scores = generate_vllm(
            prompt, model, tokenizer, temperatures, num_samples, max_tokens=max_tokens
        )
        for step, score in zip(step_cands, step_scores):
            child_state, result = apply_step_to_state(node.state, step)  
            
            if isinstance(result, ProofFinished):
                child_node = Node(child_state)
                child_node.proof_finished = True
                child_node.score = score
                node.add_child(child_node)
                
            elif isinstance(result, TacticState):
                child_node = Node(child_state)
                child_node.valid_tactic = True
                child_node.score = score
                node.add_child(child_node)

    def node_evaluator(child, montecarlo):
        # Increment visit count for exploration tracking
        child.visits += 1
        
        if child.proof_finished:
            # Only update the reward for finished proofs
            child.total_reward += 1

    root_state = prompt_fn(theorem)
    root_node = Node(root_state)
    montecarlo = MonteCarlo(root_node, mins_timeout=timeout/60)
    montecarlo.child_finder = child_finder
    montecarlo.node_evaluator = node_evaluator

    montecarlo.simulate(expansion_count=max_iters)

    # Extraction and return of results as previously outlined
    def format_solution(node):
        """
        Recursively trace back from the solution node to the root to construct the proof steps.
        """
        solution_steps = []
        current_node = node
        while current_node.parent is not None:  # Trace back to root
            if hasattr(current_node, 'tactic'):  # Assuming nodes store their tactic
                solution_steps.append(current_node.tactic)
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

    # At the end of mct_search function, after montecarlo.simulate(expansion_count=max_iters)
    solution_node = find_solution_node(montecarlo.root_node)
    if solution_node:
        proof_steps = format_solution(solution_node)
        result = {
            'theorem': theorem,  # Adjust according to how you identify theorems
            'success': True,
            'proof_steps': proof_steps,
            'details': {
                # Additional details like total iterations, time taken, etc.
                'iterations': max_iters,
                'timeout': timeout
            }
        }
    else:
        result = {
            'theorem': theorem,
            'success': False,
            'failure_reason': 'No solution found within the given constraints',
            'details': {
                'iterations': max_iters,
                'timeout': timeout
            }
        }

    return result



# def generate_complete(text, 
#                       montecarlo, 
#                       current_completion_depth: int=1,
#                       max_completion_depth: int=30):

#     if current_completion_depth >= max_completion_depth:
#         return None
    
#     prev = text
#     texts = llm.generate(text, 1)  # TO EDIT
#     text = texts[0]
#     score = score_func(text)  # TO EDIT WITH DOJO
#     print(diffprompt(prev, texts))  # TO EDIT WITH DOJO

#     if score is not None:
#         if score < 0:
#             return None
#         else:
#             if can_be_solution(text, min_lines, check_func):  # TO EDIT WITH DOJO
#                 montecarlo.solution = text
#             return text
#     else:
#         return generate_complete(text, montecarlo, current_completion_depth + 1)


# def child_finder(node, montecarlo):
#     if limit_depth(node):
#         return

#     text = generate_complete(node.state, montecarlo)
#     if text is None:
#         node.update_win_value(-1)
#     else:
#         child = Node(text)
#         node.add_child(child)
#         child.update_win_value(1)
#         child.update_policy_value(1)

#         child = Node(node.state)
#         node.add_child(child)
#         child.update_policy_value(0.2)

# def mct_search(prompt: str='',
#                mins_timeout: int=None,
#                expansion_count: int=None):
    
#     # Initialize the montecarlo
#     montecarlo = MonteCarlo(Node(prompt), mins_timeout)  # TO EDIT
#     montecarlo.child_finder = child_finder  # TO WRITE THIS FUNCTION
    
#     # Simulate
#     montecarlo.simulate(expansion_count)
    
#     # Record the results
#     return montecarlo.solution  # TO EDIT
    
    