"""
Tree-accelerated tabular CFR variants.

These functions replace string-based game-logic calls with pre-computed
integer lookups from ``GameTree``.  The interface mirrors the original
module-level functions but accepts a ``tree`` parameter and uses integer
history / infoset IDs internally.

Usage
-----
>>> tree = GameTree(game)
>>> node_map = {}
>>> cfr_tree(tree, cards, comm_rank, tree.nodes[0].hid, 1.0, 1.0, node_map)
"""

import numpy as np
from node import Node


# ════════════════════════════════════════════════════════════════════
#  Standard CFR (tree version)
# ════════════════════════════════════════════════════════════════════

def cfr_tree(tree, cards, comm_rank, hid, p0, p1, node_map):
    """Standard CFR backed by a pre-computed game tree."""
    node_info = tree.nodes[hid]
    player = node_info.player
    opponent = 1 - player

    if node_info.is_terminal:
        payoff = tree.get_payoff(hid, cards)
        return payoff if player == 0 else -payoff

    iid = tree.infoset_id(cards[player], hid)
    if iid not in node_map:
        node_map[iid] = Node(num_actions=tree.num_actions)
    node = node_map[iid]

    reach_prob = p0 if player == 0 else p1
    strategy = node.get_strategy(reach_prob)

    na = tree.num_actions
    util = np.zeros(na)
    node_util = 0.0

    for a in node_info.legal_actions:
        child_hid = node_info.child_for(a, comm_rank)
        if child_hid is None:
            continue
        if player == 0:
            util[a] = -cfr_tree(tree, cards, comm_rank, child_hid,
                                p0 * strategy[a], p1, node_map)
        else:
            util[a] = -cfr_tree(tree, cards, comm_rank, child_hid,
                                p0, p1 * strategy[a], node_map)
        node_util += strategy[a] * util[a]

    for a in node_info.legal_actions:
        regret = util[a] - node_util
        if player == 0:
            node.regret_sum[a] += p1 * regret
        else:
            node.regret_sum[a] += p0 * regret

    return node_util


# ════════════════════════════════════════════════════════════════════
#  CFR+ (tree version)
# ════════════════════════════════════════════════════════════════════

def cfr_plus_tree(tree, cards, comm_rank, hid, p0, p1, node_map):
    """CFR+ backed by a pre-computed game tree."""
    node_info = tree.nodes[hid]
    player = node_info.player
    opponent = 1 - player

    if node_info.is_terminal:
        payoff = tree.get_payoff(hid, cards)
        return payoff if player == 0 else -payoff

    iid = tree.infoset_id(cards[player], hid)
    if iid not in node_map:
        node_map[iid] = Node(num_actions=tree.num_actions)
    node = node_map[iid]

    reach_prob = p0 if player == 0 else p1
    strategy = node.get_strategy(reach_prob)

    na = tree.num_actions
    util = np.zeros(na)
    node_util = 0.0

    for a in node_info.legal_actions:
        child_hid = node_info.child_for(a, comm_rank)
        if child_hid is None:
            continue
        if player == 0:
            util[a] = -cfr_plus_tree(tree, cards, comm_rank, child_hid,
                                     p0 * strategy[a], p1, node_map)
        else:
            util[a] = -cfr_plus_tree(tree, cards, comm_rank, child_hid,
                                     p0, p1 * strategy[a], node_map)
        node_util += strategy[a] * util[a]

    for a in node_info.legal_actions:
        regret = util[a] - node_util
        wt = p1 if player == 0 else p0
        node.regret_sum[a] = max(0.0, node.regret_sum[a] + wt * regret)

    return node_util


# ════════════════════════════════════════════════════════════════════
#  DCFR (tree version)
# ════════════════════════════════════════════════════════════════════

def _discount(t, exponent):
    return (t / (t + 1)) ** exponent


def dcfr_tree(tree, cards, comm_rank, hid, p0, p1, node_map,
              iter_cnt_ref, alpha=1.5, beta=0.0, gamma=2.0):
    """Discounted CFR backed by a pre-computed game tree.

    ``iter_cnt_ref`` should be a single-element list [N] that is
    incremented by the caller once per root call before invoking.
    """
    node_info = tree.nodes[hid]
    player = node_info.player
    opponent = 1 - player

    if node_info.is_terminal:
        payoff = tree.get_payoff(hid, cards)
        return payoff if player == 0 else -payoff

    t = iter_cnt_ref[0]

    iid = tree.infoset_id(cards[player], hid)
    if iid not in node_map:
        node_map[iid] = Node(num_actions=tree.num_actions)
    node = node_map[iid]

    reach_prob = p0 if player == 0 else p1
    strategy = node.get_strategy(reach_prob, strategy_discount=_discount(t, gamma))

    na = tree.num_actions
    util = np.zeros(na)
    node_util = 0.0

    for a in node_info.legal_actions:
        child_hid = node_info.child_for(a, comm_rank)
        if child_hid is None:
            continue
        if player == 0:
            util[a] = -dcfr_tree(tree, cards, comm_rank, child_hid,
                                 p0 * strategy[a], p1, node_map, iter_cnt_ref,
                                 alpha, beta, gamma)
        else:
            util[a] = -dcfr_tree(tree, cards, comm_rank, child_hid,
                                 p0, p1 * strategy[a], node_map, iter_cnt_ref,
                                 alpha, beta, gamma)
        node_util += strategy[a] * util[a]

    pos_d = _discount(t, alpha)
    neg_d = _discount(t, beta)

    for a in node_info.legal_actions:
        regret = util[a] - node_util
        wt = p1 if player == 0 else p0
        weighted = wt * regret
        prev = node.regret_sum[a]
        discounted = pos_d * max(prev, 0.0) + neg_d * min(prev, 0.0)
        node.regret_sum[a] = discounted + weighted

    return node_util


# ════════════════════════════════════════════════════════════════════
#  PDCFR+ (tree version)
# ════════════════════════════════════════════════════════════════════

def _predictive_strategy(node, num_actions):
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


def pdcfr_plus_tree(tree, cards, comm_rank, hid, p0, p1, node_map,
                    iter_cnt_ref, alpha=1.5, beta=0.0, gamma=2.0):
    """Predictive Discounted CFR+ backed by a pre-computed game tree."""
    node_info = tree.nodes[hid]
    player = node_info.player
    opponent = 1 - player

    if node_info.is_terminal:
        payoff = tree.get_payoff(hid, cards)
        return payoff if player == 0 else -payoff

    t = iter_cnt_ref[0]

    iid = tree.infoset_id(cards[player], hid)
    if iid not in node_map:
        node_map[iid] = Node(num_actions=tree.num_actions)
    node = node_map[iid]

    reach_prob = p0 if player == 0 else p1
    na = tree.num_actions
    strategy = _predictive_strategy(node, na)

    util = np.zeros(na)
    node_util = 0.0

    for a in node_info.legal_actions:
        child_hid = node_info.child_for(a, comm_rank)
        if child_hid is None:
            continue
        if player == 0:
            util[a] = -pdcfr_plus_tree(tree, cards, comm_rank, child_hid,
                                       p0 * strategy[a], p1, node_map,
                                       iter_cnt_ref, alpha, beta, gamma)
        else:
            util[a] = -pdcfr_plus_tree(tree, cards, comm_rank, child_hid,
                                       p0, p1 * strategy[a], node_map,
                                       iter_cnt_ref, alpha, beta, gamma)
        node_util += strategy[a] * util[a]

    pos_d = _discount(t, alpha)
    neg_d = _discount(t, beta)
    strat_d = _discount(t, gamma)

    for a in node_info.legal_actions:
        inst = util[a] - node_util
        wt = p1 if player == 0 else p0
        weighted = wt * inst
        prev = node.regret_sum[a]
        discounted = pos_d * max(prev, 0.0) + neg_d * min(prev, 0.0)
        node.regret_sum[a] = max(0.0, discounted + weighted)
        node.last_inst_regret[a] = weighted
        node.strategy_sum[a] = (strat_d * node.strategy_sum[a]
                                + reach_prob * strategy[a])

    return node_util
