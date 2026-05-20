from game import *
from node import *
import numpy as np

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


def dcfr(cards, history, p0, p1):
    """
    Discounted CFR (DCFR) recursion.

    Applies separate discounts to positive regrets, negative regrets,
    and strategy accumulation, as described in:

        Brown & Sandholm (2019) — "Solving Imperfect-Information Games
        via Discounted Regret Minimization"

    Default parameters: α=1.5, β=0, γ=2.
    """
    global _iter_cnt

    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    # Terminal node
    if is_terminal(history):
        payoff = get_payoff(history, cards)
        return payoff if player == 0 else -payoff

    # Auto-increment iteration counter at the root of each traversal
    if history == "":
        _iter_cnt += 1
    t = _iter_cnt  # current iteration (1-indexed)

    # Infoset key
    infoset = cards[player] + history
    if infoset not in node_map_dcfr:
        node_map_dcfr[infoset] = Node()

    node = node_map_dcfr[infoset]
    reach_prob = p0 if player == 0 else p1

    # Get strategy — pass DCFR discount for strategy accumulation
    strategy = node.get_strategy(reach_prob, strategy_discount=_discount(t, GAMMA))

    # Evaluate each action
    util = np.zeros(NUM_ACTIONS)
    node_util = 0.0

    for a in range(NUM_ACTIONS):
        next_history = history + ACTIONS[a]

        if player == 0:
            util[a] = -dcfr(cards, next_history, p0 * strategy[a], p1)
        else:
            util[a] = -dcfr(cards, next_history, p0, p1 * strategy[a])

        node_util += strategy[a] * util[a]

    # DCFR regret update:
    #   R^t[a] = d_pos * (R^{t-1}[a])_+  +  d_neg * (R^{t-1}[a])_-  +  r^t[a]
    pos_discount = _discount(t, ALPHA)
    neg_discount = _discount(t, BETA)

    for a in range(NUM_ACTIONS):
        regret = util[a] - node_util
        weighted_regret = (p1 if player == 0 else p0) * regret

        prev = node.regret_sum[a]
        discounted = (
            pos_discount * max(prev, 0.0)
            + neg_discount * min(prev, 0.0)
        )
        node.regret_sum[a] = discounted + weighted_regret

    return node_util
