import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from models import llm

from utils.cmdline import args

GREEDY = args.greedy
N_SAMPLES = args.n_samples
MAX_N_SAMPLES = args.max_n_samples

from langs.lang import can_be_solution
from langs.lang import score_func as uncached_score_func
from utils.common import (
    limit_depth,
    max_completion_depth,
    stats,
    create_cached_func,
    diffprompt
)

score_func, cache_stats, reset_cache = create_cached_func(uncached_score_func)

from utils.prompts import prompt, min_lines, check_func



score_stats = {'positive': 0, 'negative': 0, 'unknown': 0}
solution_stats = {'yes': 0, 'no': 0}
solutions = []

def attempt():

    # ---------------------------------------------------------------- #
    # Set up the training mode
    if GREEDY:
        text = llm.generate_full(prompt)
    else:
        text = llm.generate_full(prompt,
            do_sample=True, top_p=0.9, top_k=7, temperature=0.8)
    # -----------------------------------------------------------------

    # ----------------------------------------------------------------- #
    # Set up the scoring function
    score = score_func(text)

    if score is None:
        score_key = 'unknown'
    elif score > 0:
        score_key = 'positive'
    else:
        score_key = 'negative'

    score_stats[score_key] += 1
    # ------------------------------------------------------------------

    # ----------------------------------------------------------------- #
    # Get the solution key
    if (score is not None
        and score > 0
        and can_be_solution(text, min_lines, check_func)):
        solution_key = 'yes'
    else:
        solution_key = 'no'

    # Update the global solution stats
    solution_stats[solution_key] += 1

    if solution_key == 'yes':
        solutions.append(text)
        return text
    # -----------------------------------------------------------------

    return None

def main(mins_timeout=None):
    """
    """

    if MAX_N_SAMPLES is not None:
        assert not GREEDY
        n_calls, solution = 0, None

        # Keep attempting until a solution is found or reach max samples
        while solution is None and n_calls < MAX_N_SAMPLES:
            n_calls += 1
            solution = attempt()

        if solution:  print(f'SOLUTION FOUND:\n{solution}')
        else:         print('SOLUTION is None.')

        return {'n_calls': n_calls}

    else:
        for i in range(0, 1 if GREEDY else N_SAMPLES):
            attempt()

        for solution in solutions:
            print(f'One solution:\n{solution}')

        print(f'{score_stats}\n{solution_stats}')

        return solution_stats

if __name__ == '__main__':
    main()
