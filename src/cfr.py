from game import *
from node import *
import numpy as np


# infoset map
node_map = {}


def cfr(cards, history, p0, p1):
    """
    CFR recursion.

    Args:
        cards:
            player private cards

        history:
            betting history

        p0:
            reach probability for player0

        p1:
            reach probability for player1

    Returns:
        utility value
    """

    plays = len(history)

    # current player
    player = plays % 2

    opponent = 1 - player

    # terminal node
    if is_terminal(history):

        payoff = get_payoff(history, cards)

        return payoff if player == 0 else -payoff

    # infoset key
    infoset = cards[player] + history

    # get/create node
    if infoset not in node_map:
        node_map[infoset] = Node()

    node = node_map[infoset]

    # current strategy
    strategy = node.get_strategy(
        p0 if player == 0 else p1
    )

    util = np.zeros(NUM_ACTIONS)

    node_util = 0

    # for each action
    for a in range(NUM_ACTIONS):

        next_history = history + ACTIONS[a]

        # recursive traversal
        if player == 0:

            util[a] = -cfr(
                cards,
                next_history,
                p0 * strategy[a],
                p1
            )

        else:

            util[a] = -cfr(
                cards,
                next_history,
                p0,
                p1 * strategy[a]
            )

        node_util += strategy[a] * util[a]

    # regret update
    for a in range(NUM_ACTIONS):

        regret = util[a] - node_util

        if player == 0:
            node.regret_sum[a] += p1 * regret
        else:
            node.regret_sum[a] += p0 * regret

    return node_util