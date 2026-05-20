from game import *
from node import *
import numpy as np

# Separate node map for CFR+
node_map_plus = {}

def cfr_plus(cards, history, p0, p1):
    """
    CFR+ recursion.
    Differs from standard CFR by using positive regret updates.
    """
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    if is_terminal(history):
        payoff = get_payoff(history, cards)
        return payoff if player == 0 else -payoff

    infoset = cards[player] + history
    if infoset not in node_map_plus:
        node_map_plus[infoset] = Node()

    node = node_map_plus[infoset]
    reach_prob = p0 if player == 0 else p1
    strategy = node.get_strategy(reach_prob)

    util = np.zeros(NUM_ACTIONS)
    node_util = 0.0

    for a in range(NUM_ACTIONS):
        next_history = history + ACTIONS[a]
        if player == 0:
            util[a] = -cfr_plus(cards, next_history, p0 * strategy[a], p1)
        else:
            util[a] = -cfr_plus(cards, next_history, p0, p1 * strategy[a])
        node_util += strategy[a] * util[a]

    # CFR+ update: keep only positive regrets
    for a in range(NUM_ACTIONS):
        regret = util[a] - node_util
        if player == 0:
            node.regret_sum[a] = max(0.0, node.regret_sum[a] + p1 * regret)
        else:
            node.regret_sum[a] = max(0.0, node.regret_sum[a] + p0 * regret)

    return node_util