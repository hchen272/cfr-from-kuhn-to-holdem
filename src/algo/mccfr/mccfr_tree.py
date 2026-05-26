"""External Sampling Monte Carlo CFR (MCCFR).

Samples a single deal per iteration.  The traverser explores ALL actions;
the opponent samples ONE action from their current strategy.  This avoids
full-game-tree traversal while maintaining unbiased regret estimates.

Uses the existing GameTree + Node infrastructure (tabular storage).
"""
import random
import numpy as np

from algo.tabular.node import Node


def _get_strategy(node, num_actions):
    """Regret-matching strategy from node.regret_sum."""
    r = np.maximum(node.regret_sum, 0.0)
    s = r.sum()
    if s > 1e-12:
        return r / s
    return np.ones(num_actions) / num_actions


def mccfr_tree(tree, cards, comm_rank, hid, p0, p1, node_map,
               traverser, update_player=-1):
    """External-sampling MCCFR traversal (tabular).

    Parameters
    ----------
    tree : GameTree
    cards : (p0_rank, p1_rank)
    comm_rank : str
    hid : int — current node ID
    p0, p1 : float — reach probabilities
    node_map : dict {iid → Node}
    traverser : int (0 or 1) — the player who explores all actions
    update_player : int — -1=both, else only that player

    Returns
    -------
    float — node utility (from traverser's perspective at root; P0 elsewhere)
    """
    ni = tree.nodes[hid]
    if ni.is_terminal:
        pay = tree.get_payoff(hid, cards)
        return pay if ni.player == 0 else -pay

    player = ni.player
    iid = tree.infoset_id(cards[player], hid)
    if iid not in node_map:
        node_map[iid] = Node(tree.num_actions)
    node = node_map[iid]
    strategy = _get_strategy(node, tree.num_actions)
    legal = ni.legal_actions

    if player == 0 and p0 == 0.0:
        return 0.0
    if player == 1 and p1 == 0.0:
        return 0.0

    na = tree.num_actions
    child_values = [0.0] * na
    reach = p0 if player == 0 else p1
    opp_reach = p1 if player == 0 else p0

    # Accumulate strategies weighted by reach (for strategy_sum update later)
    for a in legal:
        node.strategy_sum[a] += reach * strategy[a]

    if traverser == player:
        # ── traverser: explore all actions ──
        expl = {0: 1.0, 1: 1.0, 2: 1.0}  # dummy — we just need indices
        for a in legal:
            ch = ni.child_for(a, comm_rank)
            if ch is None:
                continue
            np0, np1 = (reach * strategy[a], p1) if player == 0 else (p0, reach * strategy[a])
            v = mccfr_tree(tree, cards, comm_rank, ch, np0, np1, node_map,
                           traverser, update_player)
            child_values[a] = -v   # negate: child value → parent perspective
            # accumulate regret: counterfactual value = opp_reach * payoff_contribution
            # instant regret = opp_reach * (child_value - node_value)
            # We compute at the end once we have node_value

        node_value = sum(strategy[a] * child_values[a] for a in legal)
        node_value /= sum(strategy[a] for a in legal)

        # Update regrets
        if update_player == -1 or update_player == player:
            for a in legal:
                instant = opp_reach * (child_values[a] - node_value)
                node.regret_sum[a] += instant

        return node_value

    else:
        # ── opponent: sample one action ──
        probs = np.array([strategy[a] for a in legal])
        probs /= probs.sum()
        a = np.random.choice(legal, p=probs)
        ch = ni.child_for(a, comm_rank)
        if ch is None:
            return 0.0
        np0, np1 = (reach * strategy[a], p1) if player == 0 else (p0, reach * strategy[a])
        v = mccfr_tree(tree, cards, comm_rank, ch, np0, np1, node_map,
                       traverser, update_player)
        return -v    # negate: child → parent perspective
