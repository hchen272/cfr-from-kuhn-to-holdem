import numpy as np
from node import Node

# Separate node map for PDCFR+
node_map_pdcfr = {}

# PDCFR+ hyper-parameters (Brown & Sandholm, 2019)
ALPHA = 1.5   # discount exponent for positive regrets
BETA  = 0.0   # discount exponent for negative regrets
GAMMA = 2.0   # discount exponent for strategy accumulation

# Internal iteration counter (auto-incremented at root calls)
_iter_cnt = 0


def _discount(t, exponent):
    """Compute (t / (t + 1)) ** exponent."""
    return (t / (t + 1)) ** exponent


def _predictive_strategy(node, num_actions):
    """
    Compute strategy from PREDICTED cumulative regret.

    PDCFR+ uses R + last_inst_regret as the prediction of what the
    cumulative regret will be after the next update, instead of using
    R alone (as in standard regret matching).

    Returns:
        np.array: predicted strategy (also stored in node.strategy)
    """
    pred = node.regret_sum + node.last_inst_regret

    normalizing_sum = 0.0
    for a in range(num_actions):
        node.strategy[a] = max(pred[a], 0.0)
        normalizing_sum += node.strategy[a]

    if normalizing_sum > 0:
        node.strategy /= normalizing_sum
    else:
        node.strategy[:] = 1.0 / num_actions

    return node.strategy


def pdcfr_plus(game, cards, history, p0, p1):
    """
    Predictive Discounted CFR+ (PDCFR+) recursion.

    Combines three ideas:
      - **Predictive** regret matching: strategy is computed from
        R(a) + last_inst_regret(a), anticipating the next update.
      - **Discounted** regret update (α, β, γ) from DCFR.
      - **CFR+** positive regret clamping (max with 0).

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
    if infoset not in node_map_pdcfr:
        node_map_pdcfr[infoset] = Node(num_actions=game.num_actions)

    node = node_map_pdcfr[infoset]
    reach_prob = p0 if player == 0 else p1
    na = game.num_actions

    # -------- Predictive strategy --------
    strategy = _predictive_strategy(node, na)

    # -------- Traverse children --------
    util = np.zeros(na)
    node_util = 0.0

    legal_actions = game.get_legal_actions(history)
    legal_set = set(legal_actions)

    for a in range(na):
        if game.ACTIONS[a] not in legal_set:
            continue
        next_history = game.build_next_history(history, game.ACTIONS[a])

        if player == 0:
            util[a] = -pdcfr_plus(game, cards, next_history, p0 * strategy[a], p1)
        else:
            util[a] = -pdcfr_plus(game, cards, next_history, p0, p1 * strategy[a])

        node_util += strategy[a] * util[a]

    # -------- Regret update (DCFR + CFR+) --------
    pos_discount = _discount(t, ALPHA)
    neg_discount = _discount(t, BETA)
    strat_discount = _discount(t, GAMMA)

    for a in range(na):
        if game.ACTIONS[a] not in legal_set:
            continue
        # Instant counterfactual regret
        inst_regret = util[a] - node_util
        weighted_regret = (p1 if player == 0 else p0) * inst_regret

        # DCFR: discount positive and negative parts separately
        prev = node.regret_sum[a]
        discounted = (
            pos_discount * max(prev, 0.0)
            + neg_discount * min(prev, 0.0)
        )

        # CFR+: clamp cumulative regret to >= 0
        node.regret_sum[a] = max(0.0, discounted + weighted_regret)

        # Save instant regret for next iteration's prediction
        node.last_inst_regret[a] = weighted_regret

        # -------- Strategy accumulation (with DCFR discount γ) --------
        node.strategy_sum[a] = (
            strat_discount * node.strategy_sum[a]
            + reach_prob * strategy[a]
        )

    return node_util
