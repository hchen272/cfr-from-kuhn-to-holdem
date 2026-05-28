"""
MCCFR traversal for Texas Hold'em (on-the-fly tree).

Adapted from ``src/algo/mccfr/mccfr_tree.py``.  Uses External Sampling:
the traverser explores all actions; the opponent samples one action from
their current strategy.  This avoids full game-tree traversal while
maintaining unbiased regret estimates.

Reuses ``src.algo.tabular.node.Node`` for regret / strategy storage.

Usage
-----
>>> from cfr import mccfr
>>> node_map = {}
>>> util = mccfr(tree, cards, "", node_map, traverser=0)
"""
import sys
import os
import random
import numpy as np

# Allow running as script
if __name__ == '__main__' and __package__ is None:
    # Insert src/ first, then holdem/ on top so local modules take priority
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algo.tabular.node import Node

# ── Regret-matching strategy ─────────────────────────────────────────


def _get_strategy(node, num_actions):
    """Compute regret-matching strategy from ``node.regret_sum``."""
    r = np.maximum(node.regret_sum, 0.0)
    s = r.sum()
    if s > 1e-12:
        return r / s
    return np.ones(num_actions) / num_actions


# ── External Sampling MCCFR ──────────────────────────────────────────


def mccfr(tree, cards, history, node_map, traverser,
          p0_reach=1.0, p1_reach=1.0, update_player=-1,
          bucket0=None, bucket1=None):
    """External-sampling MCCFR on the on-the-fly tree.

    Parameters
    ----------
    tree : OnTheFlyTree
        The on-the-fly game tree (with abstraction).
    cards : (p0_hole, p1_hole)
        Each hole is a 2-tuple of (rank, suit).
    history : str
        Current history string.
    node_map : dict
        ``{infoset_key: Node}`` — tabular regret/strategy storage.
    traverser : int
        0 or 1 — the player who explores all actions.
    p0_reach, p1_reach : float
        Reach probabilities for the current path.
    update_player : int
        -1 = update both, else update only that player's regrets.
    bucket0, bucket1 : int or None
        Pre-computed abstraction bucket ids.  Computed on first call.

    Returns
    -------
    float
        Node utility from P0's perspective at root; traverser's
        perspective elsewhere.
    """
    if tree.is_terminal(history):
        pay = tree.get_payoff(history, cards)
        # get_payoff returns P0's payoff.  Convert to *this node's*
        # player perspective so that the recursion chain's `-v`
        # negation correctly propagates back to the root.
        player = tree.player(history)
        return pay if player == 0 else -pay

    player = tree.player(history)

    # Compute buckets on first call
    if bucket0 is None:
        bucket0 = tree.abstraction.bucket_id(cards[0])
    if bucket1 is None:
        bucket1 = tree.abstraction.bucket_id(cards[1])

    bucket = bucket0 if player == 0 else bucket1
    infoset = tree.infoset_key(bucket, history)

    if infoset not in node_map:
        node_map[infoset] = Node(tree.num_actions)
    node = node_map[infoset]

    strategy = _get_strategy(node, tree.num_actions)
    legal = tree.legal_actions(history)
    na = tree.num_actions

    reach = p0_reach if player == 0 else p1_reach
    opp_reach = p1_reach if player == 0 else p0_reach

    # Early termination if reach is zero
    if reach == 0.0:
        return 0.0

    # Accumulate strategy weighted by reach
    for a in legal:
        node.strategy_sum[a] += reach * strategy[a]

    if traverser == player:
        # ── traverser: explore ALL actions ──
        child_values = [0.0] * na
        for a in legal:
            child_hist = tree.child_history(history, a)
            if player == 0:
                v = mccfr(tree, cards, child_hist, node_map, traverser,
                          p0_reach * strategy[a], p1_reach,
                          update_player, bucket0, bucket1)
            else:
                v = mccfr(tree, cards, child_hist, node_map, traverser,
                          p0_reach, p1_reach * strategy[a],
                          update_player, bucket0, bucket1)
            child_values[a] = -v   # negate: child → parent perspective

        # Expected value under current strategy
        denom = sum(strategy[a] for a in legal)
        node_value = (sum(strategy[a] * child_values[a] for a in legal)
                      / denom if denom > 0 else 0.0)

        # Update regrets
        if update_player == -1 or update_player == player:
            for a in legal:
                instant = opp_reach * (child_values[a] - node_value)
                node.regret_sum[a] += instant

        return node_value

    else:
        # ── opponent: sample ONE action ──
        probs = np.array([strategy[a] for a in legal])
        probs /= probs.sum()
        a = int(np.random.choice(legal, p=probs))
        child_hist = tree.child_history(history, a)

        if player == 0:
            v = mccfr(tree, cards, child_hist, node_map, traverser,
                      p0_reach * strategy[a], p1_reach,
                      update_player, bucket0, bucket1)
        else:
            v = mccfr(tree, cards, child_hist, node_map, traverser,
                      p0_reach, p1_reach * strategy[a],
                      update_player, bucket0, bucket1)

        return -v   # negate: child → parent perspective


# ── CFR+ (for on-the-fly tree) ────────────────────────────────────────


def cfr_plus(tree, cards, history, node_map,
             p0_reach=1.0, p1_reach=1.0, iter_cnt=1,
             update_player=-1, bucket0=None, bucket1=None):
    """CFR+ on the on-the-fly tree (full traversal — small games only).

    This is suitable for small subgames or for testing.  For full
    Texas Hold'em, use ``mccfr`` instead.
    """
    if tree.is_terminal(history):
        pay = tree.get_payoff(history, cards)
        player = tree.player(history)
        return pay if player == 0 else -pay

    player = tree.player(history)

    if bucket0 is None:
        bucket0 = tree.abstraction.bucket_id(cards[0])
    if bucket1 is None:
        bucket1 = tree.abstraction.bucket_id(cards[1])

    bucket = bucket0 if player == 0 else bucket1
    infoset = tree.infoset_key(bucket, history)

    if infoset not in node_map:
        node_map[infoset] = Node(tree.num_actions)
    node = node_map[infoset]

    reach = p0_reach if player == 0 else p1_reach
    strategy = node.get_strategy(reach, linear_weight=float(iter_cnt))

    legal = tree.legal_actions(history)
    na = tree.num_actions
    util = np.zeros(na)
    node_util = 0.0

    for a in legal:
        child_hist = tree.child_history(history, a)
        if player == 0:
            v = cfr_plus(tree, cards, child_hist, node_map,
                         p0_reach * strategy[a], p1_reach,
                         iter_cnt, update_player, bucket0, bucket1)
        else:
            v = cfr_plus(tree, cards, child_hist, node_map,
                         p0_reach, p1_reach * strategy[a],
                         iter_cnt, update_player, bucket0, bucket1)
        util[a] = -v
        node_util += strategy[a] * util[a]

    if update_player == -1 or update_player == player:
        for a in legal:
            regret = util[a] - node_util
            wt = p1_reach if player == 0 else p0_reach
            node.regret_sum[a] = max(0.0, node.regret_sum[a] + wt * regret)

    return node_util


# ── Smoke test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from game import TexasHoldemGame, _card_str
    from tree import OnTheFlyTree
    from abstraction import CardAbstraction

    print("Building abstraction...")
    absn = CardAbstraction(n_buckets=50, n_mc_samples=2000)

    game = TexasHoldemGame()
    tree = OnTheFlyTree(game, absn)

    cards = game.deal_cards()
    p0h, p1h = cards
    b0 = absn.bucket_id(p0h)
    b1 = absn.bucket_id(p1h)
    print(f"P0: {_card_str(p0h[0])} {_card_str(p0h[1])} → B{b0}")
    print(f"P1: {_card_str(p1h[0])} {_card_str(p1h[1])} → B{b1}")
    print(f"Comm: {[_card_str(c) for c in game._community_cards]}")

    # Run a few MCCFR traversals
    node_map = {}
    total = 0.0
    for t in range(5):
        traverser = t % 2
        util = mccfr(tree, cards, "", node_map, traverser=traverser,
                     bucket0=b0, bucket1=b1)
        total += util
        print(f"  iter {t}: traverser=P{traverser}  util={util:+.4f}")

    print(f"\nAvg util (5 iters): {total/5:+.4f}")
    print(f"Node map size: {len(node_map)} infosets")

    # Show a few strategies
    for i, (key, node) in enumerate(sorted(node_map.items())[:5]):
        strat = node.get_average_strategy()
        print(f"  '{key}': {strat}")
    if len(node_map) > 5:
        print(f"  ... and {len(node_map) - 5} more")
