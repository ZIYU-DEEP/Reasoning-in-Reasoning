import sys
sys.path.append('../')
from .prompts import max_depth
from cmdline import args

max_completion_depth = args.max_completion_depth

# max_completion_depth = 30

# ###############################################
# General
# ###############################################
def count_depth(node, f=lambda x: x):
    depth = 1
    curr = node

    while (curr.parent is not None):
        if f(curr.state) != f(curr.parent.state):
            depth += 1
        curr = curr.parent

    return depth

def limit_depth(node, f=lambda x: x):
    if max_depth is not None:
        depth = count_depth(node, f)
        if depth >= max_depth:
            node.update_win_value(-100)
            return True
    return False


# ###############################################
# Stats
# ###############################################
def stats(montecarlo, f=lambda x: x):
    n_nodes = 0
    n_gen_nodes = 0
    n_back_nodes = 0
    n_gen_leaves = 0
    n_back_leaves = 0
    queue = [montecarlo.root_node]
    while queue != []:
        node = queue.pop()
        n_nodes += 1
        is_back = node.parent is not None and f(node.state) == f(node.parent.state)
        is_leaf = node.children == []
        if is_back:
            n_back_nodes += 1
        else:
            n_gen_nodes += 1
        if is_leaf:
            if is_back:
                n_back_leaves += 1
            else:
                n_gen_leaves += 1
        queue += node.children

    print((
        f'STATS\n'
        f'--------------------------------\n'
        f'Number of Nodes: {n_nodes}\n'
        f'Number of Generative Nodes: {n_gen_nodes}\n'
        f'    - Including Leaves: {n_gen_leaves}\n'
        f'Number of Backward Nodes: {n_back_nodes}\n'
        f'    - Including Leaves: {n_back_leaves}\n\n'
        f'Expansion Count: {montecarlo.stats_expansion_count}\n'
        f'    - Including Failed Expansions: {montecarlo.stats_failed_expansion_count}\n'
        f'--------------------------------'
    ))

    return (n_nodes, n_gen_nodes, n_back_nodes, n_gen_leaves, n_back_leaves)


# ###############################################
# Cache
# ###############################################
def select_with_scores(texts, scores, score_predicate, select):
    indices = [i for i in range(len(texts)) if score_predicate(scores[i])]
    if indices == []:
        return texts[0], scores[0]
    text = select([texts[i] for i in indices], indices)
    return text, [scores[i] for i in indices if text == texts[i]][0]

def create_score_predicate(f=lambda x: x):
    def fetch(x):
        score = f(x)
        return score is None or score > 0
    return fetch

def score_first(x):
    return x[0]

def create_cached_func(f):
    cache = {}
    stats = {'hit': 0, 'miss': 0}
    def fetch(x):
        INITIAL = object()
        y = cache.get(x, INITIAL)
        if y == INITIAL:
            stats['miss'] += 1
            y = f(x)
            cache[x] = y
        else:
            stats['hit'] += 1
        return y

    def reset_cache():
        cache.clear()
        stats['hit'] = 0
        stats['miss'] = 0

    return fetch, stats, reset_cache


# ###############################################
# Interactive
# ###############################################
def strip_instructions(prompt):
    try:
        return prompt[prompt.index('[/INST]'):]
    except ValueError:
        return prompt

def diffprompt(prompt, results):
    n = len(strip_instructions(prompt))
    return [strip_instructions(r)[n:] for r in results]

def ask_keep(prompt, texts):
    i = 0
    for t in diffprompt(prompt, texts):
        print(i, t)
        i += 1
    inp = input("Keep which? [0...] or comment: ").strip()
    try:
        return int(inp)
    except ValueError:
        return inp
