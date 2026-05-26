import numpy as np
from algo.tabular.node import Node

# Separate node map for DCFR
node_map_dcfr = {}

# DCFR hyper-parameters (Brown & Sandholm, 2019)
ALPHA = 1.5   # discount exponent for positive regrets
BETA  = 0.0   # discount exponent for negative regrets
GAMMA = 2.0   # discount exponent for strategy accumulation

# Internal iteration counter (auto-incremented at root calls)
_iter_cnt = 0


def _discount(t, exponent):
    """Compute (t / (t + 1)) ** exponent."""
    return (t / (t + 1)) ** exponent


def dcfr(game, cards, history, p0, p1):
    """
    Discounted CFR (DCFR) recursion.

    Applies separate discounts to positive regrets, negative regrets,
    and strategy accumulation, as described in:

        Brown & Sandholm (2019) — "Solving Imperfect-Information Games
        via Discounted Regret Minimization"

    Default parameters: α=1.5, β=0, γ=2.

    Args:
        game: Game instance
    """
    global _iter_cnt

    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    # Terminal node
    if game.is_terminal(history):
        payoff = game.get_payoff(history, cards)
        return payoff if player == 0 else -payoff

    # Auto-increment iteration counter at the root of each traversal
    if history == "":
        _iter_cnt += 1
    t = _iter_cnt  # current iteration (1-indexed)

    # Infoset key
    infoset = cards[player] + history
    if infoset not in node_map_dcfr:
        node_map_dcfr[infoset] = Node(num_actions=game.num_actions)

    node = node_map_dcfr[infoset]
    reach_prob = p0 if player == 0 else p1

    # Get strategy — pass DCFR discount for strategy accumulation
    strategy = node.get_strategy(reach_prob, strategy_discount=_discount(t, GAMMA))

    # Evaluate each action
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
            util[a] = -dcfr(game, cards, next_history, p0 * strategy[a], p1)
        else:
            util[a] = -dcfr(game, cards, next_history, p0, p1 * strategy[a])

        node_util += strategy[a] * util[a]

    # DCFR regret update:
    #   R^t[a] = d_pos * (R^{t-1}[a])_+  +  d_neg * (R^{t-1}[a])_-  +  r^t[a]
    pos_discount = _discount(t, ALPHA)
    neg_discount = _discount(t, BETA)

    for a in range(na):
        if game.ACTIONS[a] not in legal_set:
            continue
        regret = util[a] - node_util
        weighted_regret = (p1 if player == 0 else p0) * regret

        prev = node.regret_sum[a]
        discounted = (
            pos_discount * max(prev, 0.0)
            + neg_discount * min(prev, 0.0)
        )
        node.regret_sum[a] = discounted + weighted_regret

    return node_util
