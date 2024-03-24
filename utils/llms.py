"""
The LLM generation functions.
"""

from .prompts import *
from accelerate import Accelerator
import torch
import concurrent
import transformers
import os
import json
import time
from datetime import datetime
import logging
import openai
import tiktoken
from openai import OpenAI

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

try:
    import vllm
except Exception:
    pass


# ===============================================
# 1. Load the model
# ===============================================
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                  low_cpu_mem_usage=True)
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


# ===============================================
# 2. Generation with VLLM
# ===============================================
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
def generate_vllm(prompt, 
                  model, 
                  tokenizer, 
                  temperatures, 
                  num_samples, 
                  stop: str='----', 
                  max_tokens: int=256):

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


# ===============================================
# 3. Generation with HuggingFace
# ===============================================
# THIS PART IS A BIT PROBLEMATIC DUE TO SOME INCONSISTENCY OF STOP WORDS
# TO MODIFY LATER

def trim_output(output_text, 
                stop_div='----', 
                trim_at_last=False):
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


def sequence_scores(out, 
                    prompt_length, 
                    model, 
                    tokenizer, 
                    stop_div='----'):
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
        try:
            self.stops = [stop.to('cuda') for stop in stops]
        except Exception:
            self.stops = [stop for stop in stops]
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        
        for stop in self.stops:
            stop_count = (input_ids[0] == stop).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
            
        return False

# This part is hard-coded for the stopping criteria for test purpose.
stop_words_ids = [torch.tensor([807])]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

def generate_hf(prompt, 
                model, 
                tokenizer, 
                temperatures, 
                num_samples, 
                stop: str='----',
                max_tokens: int=256):

    # Init texts and scores
    texts, scores = [], []

    # Get the input ids
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    # Generate
    with torch.no_grad():
        # Does beam search at temp 0.0, otherwise temperature sampling.
        for temp in temperatures:
            decoding_params = dict(
                max_new_tokens=max_tokens,
                do_sample=temp > 0,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=num_samples,
                return_dict_in_generate=True,
                output_scores=True,
                stopping_criteria=stopping_criteria,  # This part is to be editted
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


# ===============================================
# 4. Generation with OpenAI
# ===============================================
def generate_openai(prompt: str='',
                    model: str='gpt-4-0125-preview',
                    tokenizer: str=None,
                    temperatures: tuple=(0.4,),
                    num_samples: int=4,
                    stop: str='----',
                    max_tokens: int=256):

    # Format the messages
    messages = [{'role': 'system',
                 'content': SYSTEM_MESSAGE},
                {'role': 'user',
                 'content': prompt}]

    # Init texts and scores
    texts, scores = [], []

    for temperature in temperatures:
        params = dict(model=model,
                      messages=messages,
                      n=num_samples,
                      max_tokens=max_tokens,
                      temperature=temperature,
                      stop=stop,
                      logprobs=True)
        
        # Set the encoding
        encoding = tiktoken.encoding_for_model(model)

        # Get the completion
        client = OpenAI()
        completion = client.chat.completions.create(**params)

        # Get the texts and scores (cumulative logprobs normalized by length)
        for choice in completion.choices:
            # Get the text
            text = choice.message.content

            # Get the score
            cumulative_logprob = sum([i.logprob for i in choice.logprobs.content])
            score = cumulative_logprob / len(encoding.encode(text))

            texts.append(text)
            scores.append(score)

        texts, scores = _unique_sorted(texts, scores)

        return texts, scores