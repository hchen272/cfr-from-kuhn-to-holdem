import numpy as np
from tabular.node import Node


# infoset map
node_map = {}


def cfr(game, cards, history, p0, p1):
    """
    CFR recursion.

    Args:
        game:
            Game instance (provides is_terminal, get_payoff, ACTIONS, etc.)

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
    if game.is_terminal(history):

        payoff = game.get_payoff(history, cards)

        return payoff if player == 0 else -payoff

    # infoset key
    infoset = cards[player] + history

    # get/create node
    if infoset not in node_map:
        node_map[infoset] = Node(num_actions=game.num_actions)

    node = node_map[infoset]

    # current strategy
    strategy = node.get_strategy(
        p0 if player == 0 else p1
    )

    na = game.num_actions
    util = np.zeros(na)

    node_util = 0

    # Determine legal actions for this state
    legal_actions = game.get_legal_actions(history)
    legal_set = set(legal_actions)

    # for each action (only legal ones)
    for a in range(na):

        if game.ACTIONS[a] not in legal_set:
            continue  # skip illegal actions

        next_history = game.build_next_history(history, game.ACTIONS[a])

        # recursive traversal
        if player == 0:

            util[a] = -cfr(
                game,
                cards,
                next_history,
                p0 * strategy[a],
                p1
            )

        else:

            util[a] = -cfr(
                game,
                cards,
                next_history,
                p0,
                p1 * strategy[a]
            )

        node_util += strategy[a] * util[a]

    # regret update
    for a in range(na):

        if game.ACTIONS[a] not in legal_set:
            continue

        regret = util[a] - node_util

        if player == 0:
            node.regret_sum[a] += p1 * regret
        else:
            node.regret_sum[a] += p0 * regret

    return node_util
