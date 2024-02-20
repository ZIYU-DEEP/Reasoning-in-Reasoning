"""
Basic implementation of MCTS.
(We would move on to the more advanced version with the lightzero framework later.)
"""

import random
import time
import json
import math

from math import log, sqrt
from lean_dojo import *


# ###############################################
# MCTS
# ###############################################
class MonteCarlo:
    """
    Monte Carlo Tree Search.
    """
    def __init__(self, root_node, mins_timeout=None):
        
        self.root_node = root_node
        self.solution = None
        
        self.child_finder = None
        self.node_evaluator = lambda child, montecarlo: None
        
        self.stats_expansion_count = 0
        self.stats_failed_expansion_count = 0
        self.mins_timeout = mins_timeout

    def make_choice(self):
        """
        Greedy selection for nodes.
        """
        best_children = []
        most_visits = float('-inf')

        for child in self.root_node.children:
            if child.visits > most_visits:
                most_visits = child.visits
                best_children = [child]
                
            elif child.visits == most_visits:
                best_children.append(child)

        return random.choice(best_children)

    def make_exploratory_choice(self):
        """
        Selection of the node according to their probabilities.
        """
        children_visits = map(lambda child: child.visits, self.root_node.children)
        children_visit_probabilities = [
            visit / self.root_node.visits for visit in children_visits
        ]
        random_probability = random.uniform(0, 1)
        probabilities_already_counted = 0.0

        for i, probability in enumerate(children_visit_probabilities):
            if probabilities_already_counted + probability >= random_probability:
                return self.root_node.children[i]

            probabilities_already_counted += probability

    def make_ucb_choice(self, C=1.41):
        """
        Selection of the node according to the most basic UCB.
        """
        best_child = None
        highest_ucb = float('-inf')

        for child in self.root_node.children:
            if child.visits > 0:
                avg_reward = child.total_reward / child.visits  # Assuming total_reward exists
                ucb = avg_reward + C * math.sqrt(2 * math.log(self.root_node.visits) / child.visits)
                if ucb > highest_ucb:
                    highest_ucb = ucb
                    best_child = child
            else:
                # Handle the case where a child has not been visited
                return child  # This ensures that every child is visited at least once

        return best_child

    def make_scored_ucb_choice(self, C=1.41):
        best_child = None
        highest_score = float('-inf')  # Adjust for scores being negative log probabilities

        for child in self.root_node.children:
            if child.visits > 0:
                # Integrate the score in the selection criteria
                # For simplicity, you might start with a direct comparison of scores,
                # or adjust the UCB formula to incorporate scores.
                ucb_value = child.score + C * math.sqrt(math.log(self.root_node.visits) / child.visits)
                if ucb_value > highest_score:
                    highest_score = ucb_value
                    best_child = child
            else:
                # Ensure all children are explored at least once
                return child

        return best_child

    def simulate(self, expansion_count=1):
        """
        Run monte carlo tree search from the root node.
        """
        # Init
        i, start_time = 0, time.time()

        while expansion_count is None or i < expansion_count:
            i += 1
            print('Simulating from the beginning!')

            # ----------------------------------------------------------------- #
            # STOP CONDITIONS
            # Stop if found a solution
            if self.solution:
                print('Solution found. No more simulations.')
                return

            # Stop if reached the time limit
            if self.mins_timeout is not None:
                duration = time.time() - start_time

                if duration > (self.mins_timeout * 60):
                    print('I am tired. Stopping expansion on current node.')
                    return
            # -----------------------------------------------------------------

            # Select a node
            current_node = self.root_node
            while current_node.expanded:
                current_node = current_node.get_preferred_child(self.root_node)
            
            if current_node.proof_finished:
                self.solution = True
                return

            # Rollout the selected node
            self.expand(current_node)

    def expand(self, node):
        """
        Expansion for a given node.
        """
        # DEBUG
        if node.proof_finished:
            self.solution = True
            return
        
        self.stats_expansion_count += 1
        self.child_finder(node, self)

        for child in node.children:
            child_win_value = self.node_evaluator(child, self)

            if child_win_value != None:
                child.update_win_value(child_win_value)
            
            if child.proof_finished:
                self.solution = True
                return # TODO: During inference, we return immediately when success

            if (not child.is_scorable()) and (not child.proof_finished):
                self.random_rollout(child)
                child.children = []

        if len(node.children):
            node.expanded = True
        else:
            self.stats_failed_expansion_count += 1

    def random_rollout(self, node):
        """
        Simulation.
        
        Notice that for inference, we will return immediately when success.
        """
        # TODO: DEBUG
        if node.proof_finished:
            self.solution = True
            return
        
        # Add child_node to the current node
        self.child_finder(node, self)
        
        # Randomly select a child to proceed
        # TODO: MAKE THIS WITH UCB
        child = random.choice(node.children)
        node.children = []
        node.add_child(child)
        
        # Calculdate the win value
        child_win_value = self.node_evaluator(child, self)

        if child_win_value != None:
            node.update_win_value(child_win_value)
            # TODO: DEBUG # During inference, we return immediately when success
            if child.proof_finished:
                self.solution = True
                return
        else:
            self.random_rollout(child)

    def print_tree(self, f):
        f.write("graph\n{\n")
        self.root_node.print_node(f, 0, self.root_node, "a")
        f.write("}\n")
        

# ###############################################
# Node
# ###############################################
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


class Node:
    def __init__(self, state):
        # Set the state
        self.dojo_state = state # Be it TacticState, ProofFinished or Error
        self.state = _tactic_state(state)  
        
        self.win_value = 0
        self.policy_value = None
        self.visits = 0
        self.parent = None
        self.children = []
        self.expanded = False
        self.player_number = None
        self.discovery_factor = 0.35
        
        # Additional attributes for theorem proving
        if isinstance(state, ProofFinished):
            self.proof_finished = True
        else:
            self.proof_finished = False
        # self.valid_tactic = False  # Type check the format
        self.score = 0

    def update_win_value(self, value):
        self.win_value += value
        self.visits += 1

        if self.parent:
            self.parent.update_win_value(value)

    def update_policy_value(self, value):
        self.policy_value = value

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    # def get_preferred_child(self, root_node, C=1.41):
    #     best_child = None
    #     highest_ucb_value = float('-inf')

    #     for child in self.children:
    #         if child.visits > 0:
    #             # Incorporate score into the decision-making process
    #             # Assuming 'score' is stored in child nodes and represents negative log probability
    #             ucb_value = child.score + C * math.sqrt(math.log(self.visits) / child.visits)
    #             if ucb_value > highest_ucb_value:
    #                 highest_ucb_value = ucb_value
    #                 best_child = child
    #         else:
    #             # Select unvisited child
    #             return child

    #     return best_child
    
    def get_preferred_child(self, root_node):
        best_children = []
        best_score = float("-inf")

        for child in self.children:
            score = child.get_score(root_node)

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return random.choice(best_children)

    def get_score(self, root_node):
        discovery_operand = (
            self.discovery_factor
            * (self.policy_value or 1)
            * sqrt(log(self.parent.visits) / (self.visits or 1))
        )

        win_multiplier = (
            1 if self.parent.player_number == root_node.player_number else -1
        )
        win_operand = win_multiplier * self.win_value / (self.visits or 1)

        self.score = win_operand + discovery_operand

        return self.score

    def is_scorable(self):
        return self.visits or self.policy_value != None
    
    def print_node(self, f, i, root, st):
        escape = lambda x : json.dumps(x).strip('"')
        if self.parent is None:
            f.write((' ' * i) + st + " [label=\"" + escape(self.state) + "\",shape=box]\n")
        else:
            diff = '\n'.join([x for x in self.state.split("\n") if x not in self.parent.state.split("\n")])
            f.write((' ' * i) + st + " [label=\"" + escape(diff) + "\",shape=box]\n")

        num = 0
        for child in self.children:
            new_st = st + "_" + str(num)
            child.print_node(f, i + 2, root, new_st)
            f.write(' ' * i + st + " -- " + new_st + "\n")
            num = num + 1


# ###############################################
# Helper
# ###############################################
def count_depth(node, f=lambda x: x):
    depth = 1
    curr = node

    while (curr.parent is not None):
        if f(curr.state) != f(curr.parent.state):
            depth += 1
        curr = curr.parent

    return depth


def limit_depth(node, f=lambda x: x, max_depth: int=25):
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