"""
External-sampling MCCFR tree traversal for Deep CFR.

Paper specification (Brown et al. 2019, Section 3 & 8):

  - Traverser explores ALL actions at their own decision nodes
    → counterfactual regrets are exact for the traverser.

  - Opponent samples ONE action from σ(a) and continues
    → unbiased regret estimates with lower variance.

  - Chance nodes are sampled once before each traversal
    (``game.deal_cards()``) — not part of recursion.

Strategy accumulation happens for BOTH players at every visited
node (Section 6.2: Strategy Memory).  Regret samples are stored
only for the *traverser* (Section 6.1: Advantage Memory).
"""

import numpy as np
import torch

from algo.deep_cfr.model import get_strategy_from_regrets


class Traverser:
    """Stateful traverser holding references to network, buffers, and
    the strategy-accumulation dictionary."""

    def __init__(self, regret_net, regret_buffer, game, device=None):
        self.regret_net = regret_net
        self.regret_buffer = regret_buffer      # ReservoirBuffer for regrets
        self.game = game
        self.device = device or torch.device("cpu")

        # {infoset_key: np.array}  —  cumulative strategy for averaging
        self.strategy_sum = {}

    def reset_strategy_sum(self):
        self.strategy_sum.clear()

    def get_average_strategy(self, infoset):
        total = self.strategy_sum.get(infoset)
        if total is None or total.sum() == 0:
            return np.ones(self.game.num_actions) / self.game.num_actions
        return total / total.sum()

    # ── traversal ──────────────────────────────────────────────────

    def traverse(self, cards, history, p0, p1, traverser):
        """One external-sampling MCCFR traversal.

        Parameters
        ----------
        cards : (str, str)
            (p0_rank, p1_rank).
        history : str
            Current action-history string (empty at root).
        p0, p1 : float
            Counterfactual reach probabilities.
        traverser : int
            0 or 1 — the player whose regrets are collected this traversal.

        Returns
        -------
        float
            Node utility from P0's perspective.
        """
        game = self.game
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        if game.is_terminal(history):
            payoff = game.get_payoff(history, cards)
            return payoff if player == 0 else -payoff

        # ── feature encoding & network prediction ─────────────────
        infoset = cards[player] + history
        features = game.infoset_to_features(infoset)

        with torch.no_grad():
            feats_t = torch.from_numpy(
                np.array(features, dtype=np.float32)
            ).to(self.device)
            regrets = self.regret_net(
                feats_t.unsqueeze(0)
            )[0].cpu().numpy()

        strategy = get_strategy_from_regrets(regrets, game.num_actions)

        # ── mask illegal actions ──────────────────────────────────
        legal_chars = game.get_legal_actions(history)     # list[str]   e.g. ['c','r']
        legal_set = set(legal_chars)
        na = game.num_actions
        # build int-index list for operations that need it
        legal_inds = [i for i in range(na) if game.ACTIONS[i] in legal_set]

        for a in range(na):
            if game.ACTIONS[a] not in legal_set:
                strategy[a] = 0.0
        ssum = strategy.sum()
        if ssum > 0:
            strategy /= ssum
        else:
            strategy[:] = 0.0
            strategy[legal_inds[0]] = 1.0

        # ── accumulate average strategy (both players, every node) ─
        reach_prob = p0 if player == 0 else p1
        if infoset not in self.strategy_sum:
            self.strategy_sum[infoset] = np.zeros(na, dtype=np.float64)
        self.strategy_sum[infoset] += reach_prob * strategy

        # ── recurse ───────────────────────────────────────────────
        util = np.zeros(na, dtype=np.float64)
        node_util = 0.0

        if player == traverser:
            # ── traverser: explore ALL legal actions ──────────────
            for a in range(na):
                if game.ACTIONS[a] not in legal_set:
                    continue
                next_hist = game.build_next_history(history, game.ACTIONS[a])
                if player == 0:
                    util[a] = -self.traverse(cards, next_hist,
                                             p0 * strategy[a], p1,
                                             traverser)
                else:
                    util[a] = -self.traverse(cards, next_hist,
                                             p0, p1 * strategy[a],
                                             traverser)
                node_util += strategy[a] * util[a]

            # ── store instantaneous regrets (traverser only) ──────
            regret_vec = np.zeros(na, dtype=np.float64)
            for a in range(na):
                if game.ACTIONS[a] not in legal_set:
                    continue
                inst = util[a] - node_util
                wt = p1 if player == 0 else p0
                regret_vec[a] = wt * inst
            self.regret_buffer.add(features, regret_vec)

        else:
            # ── opponent: sample ONE action ───────────────────────
            probs = np.array([strategy[i] for i in legal_inds])
            probs /= probs.sum()
            chosen = np.random.choice(legal_inds, p=probs)

            next_hist = game.build_next_history(history, game.ACTIONS[chosen])
            if player == 0:
                util[chosen] = -self.traverse(cards, next_hist,
                                              p0 * strategy[chosen], p1,
                                              traverser)
            else:
                util[chosen] = -self.traverse(cards, next_hist,
                                              p0, p1 * strategy[chosen],
                                              traverser)
            node_util = util[chosen]

        return node_util
