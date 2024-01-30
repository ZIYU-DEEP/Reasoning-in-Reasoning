import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

from models import llm

from montecarlo.node import Node
from montecarlo.montecarlo import MonteCarlo

from langs.lang import can_be_solution
from langs.lang import score_func as uncached_score_func

from utils.prompts import prompt, expansion_count, min_lines, check_func
from utils.common import (
    limit_depth,
    max_completion_depth,
    stats,
    create_cached_func,
    diffprompt
)


score_func, cache_stats, reset_cache = create_cached_func(uncached_score_func)


def generate_complete(text, montecarlo, current_completion_depth=1):

    if current_completion_depth >= max_completion_depth:
        return None
    
    prev = text
    texts = llm.generate(text, 1)
    text = texts[0]
    score = score_func(text)
    print(diffprompt(prev, texts))

    if score is not None:
        if score < 0:
            return None
        else:
            if can_be_solution(text, min_lines, check_func):
                montecarlo.solution = text
            return text
    else:
        return generate_complete(text, montecarlo, current_completion_depth + 1)


def child_finder(node, montecarlo):
    if limit_depth(node):
        return

    text = generate_complete(node.state, montecarlo)
    if text is None:
        node.update_win_value(-1)
    else:
        child = Node(text)
        node.add_child(child)
        child.update_win_value(1)
        child.update_policy_value(1)

        child = Node(node.state)
        node.add_child(child)
        child.update_policy_value(0.2)

def main(mins_timeout = None):
    # Initialize the montecarlo
    montecarlo = MonteCarlo(Node(prompt), mins_timeout)
    montecarlo.child_finder = child_finder

    # Simulate
    montecarlo.simulate(expansion_count)

    # Get the results
    print(f'CHOSEN SOLUTION: {montecarlo.solution}')

    stats(montecarlo)
    print('cache stats', cache_stats)
    # with open("graph.dot", "w") as f:
    #     montecarlo.print_tree(f)

    return cache_stats

if __name__ == "__main__":
    main()
