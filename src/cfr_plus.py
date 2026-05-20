import numpy as np
from node import Node

# Separate node map for CFR+
node_map_plus = {}

def cfr_plus(game, cards, history, p0, p1):
    """
    CFR+ recursion.
    Differs from standard CFR by using positive regret updates.

    Args:
        game: Game instance
    """
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    if game.is_terminal(history):
        payoff = game.get_payoff(history, cards)
        return payoff if player == 0 else -payoff

    infoset = cards[player] + history
    if infoset not in node_map_plus:
        node_map_plus[infoset] = Node(num_actions=game.num_actions)

    node = node_map_plus[infoset]
    reach_prob = p0 if player == 0 else p1
    strategy = node.get_strategy(reach_prob)

    na = game.num_actions
    util = np.zeros(na)
    node_util = 0.0

    legal_actions = game.get_legal_actions(history)
    legal_set = set(legal_actions)

    for a in range(na):
        if game.ACTIONS[a] not in legal_set:
            continue
        next_history = game.build_next_history(history, game.ACTIONS[a])
        if player == 0:
            util[a] = -cfr_plus(game, cards, next_history, p0 * strategy[a], p1)
        else:
            util[a] = -cfr_plus(game, cards, next_history, p0, p1 * strategy[a])
        node_util += strategy[a] * util[a]

    # CFR+ update: keep only positive regrets
    for a in range(na):
        if game.ACTIONS[a] not in legal_set:
            continue
        regret = util[a] - node_util
        if player == 0:
            node.regret_sum[a] = max(0.0, node.regret_sum[a] + p1 * regret)
        else:
            node.regret_sum[a] = max(0.0, node.regret_sum[a] + p0 * regret)

    return node_util
