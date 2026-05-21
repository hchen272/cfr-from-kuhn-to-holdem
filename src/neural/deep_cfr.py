"""
Deep CFR — game-tree traversal with neural-network regret prediction.

During each traversal:
  1. Encode the current infoset as a feature vector.
  2. Query RegretNet for action regrets, then apply regret matching.
  3. Accumulate the strategy for the final average.
  4. Recurse into child states.
  5. Compute actual instantaneous regrets and store in the buffer.

This module is stateful: it holds a reference to the regret network,
the reservoir buffer, and the strategy-accumulation dict.
"""

import numpy as np

from tabular.node import NUM_ACTIONS
from neural.model import get_strategy_from_regrets


class DeepCFR:
    """Manages one Deep CFR training session.

    Parameters
    ----------
    regret_net : RegretNet
        The PyTorch network whose forward pass returns action regrets.
    buffer : ReservoirBuffer
        Experience replay buffer.
    game : Game
        Game instance (provides is_terminal, get_payoff, ACTIONS,
        infoset_to_features, etc.)
    device : torch.device, optional
    """

    def __init__(self, regret_net, buffer, game, device=None):
        self.regret_net = regret_net
        self.buffer = buffer
        self.game = game
        self.device = device or torch.device("cpu")

        # Strategy accumulation  {infoset: np.array([sum_pass, sum_bet])}
        self.strategy_sum = {}

    def reset_strategy_sum(self):
        """Clear accumulated strategies (call before a new training run)."""
        self.strategy_sum.clear()

    def get_average_strategy(self, infoset):
        """Return the normalised average strategy for *infoset*."""
        total = self.strategy_sum.get(infoset)
        if total is None or total.sum() == 0:
            return np.ones(self.game.num_actions) / self.game.num_actions
        return total / total.sum()

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------
    def traverse(self, cards, history, p0, p1):
        """One recursive CFR traversal backed by the neural network.

        Returns the node utility (from player-0's perspective).
        """
        game = self.game
        plays = len(history)
        player = plays % 2
        opponent = 1 - player

        if game.is_terminal(history):
            payoff = game.get_payoff(history, cards)
            return payoff if player == 0 else -payoff

        # ---- feature encoding & network prediction -------------------
        infoset = cards[player] + history
        features = game.infoset_to_features(infoset)

        with torch.no_grad():
            features_t = torch.from_numpy(
                np.array(features, dtype=np.float32)
            ).to(self.device)
            regrets = self.regret_net(features_t.unsqueeze(0))[0].cpu().numpy()

        # ---- regret matching ----------------------------------------
        strategy = get_strategy_from_regrets(regrets, game.num_actions)

        # ---- zero out illegal actions & re-normalise ---------------
        legal_actions = game.get_legal_actions(history)
        legal_set = set(legal_actions)
        for a in range(game.num_actions):
            if game.ACTIONS[a] not in legal_set:
                strategy[a] = 0.0
        ssum = strategy.sum()
        if ssum > 0:
            strategy /= ssum
        else:
            n_legal = len(legal_actions)
            strategy[:] = 0.0
            for a in range(game.num_actions):
                if game.ACTIONS[a] in legal_set:
                    strategy[a] = 1.0 / n_legal

        # ---- accumulate average strategy ----------------------------
        reach_prob = p0 if player == 0 else p1
        if infoset not in self.strategy_sum:
            self.strategy_sum[infoset] = np.zeros(game.num_actions, dtype=np.float64)
        self.strategy_sum[infoset] += reach_prob * strategy

        # ---- traverse children --------------------------------------
        na = game.num_actions
        util = np.zeros(na, dtype=np.float64)
        node_util = 0.0

        for a in range(na):
            if game.ACTIONS[a] not in legal_set:
                continue
            next_hist = game.build_next_history(history, game.ACTIONS[a])
            if player == 0:
                util[a] = -self.traverse(cards, next_hist,
                                         p0 * strategy[a], p1)
            else:
                util[a] = -self.traverse(cards, next_hist,
                                         p0, p1 * strategy[a])
            node_util += strategy[a] * util[a]

        # ---- store instant regrets in buffer ------------------------
        regret_vec = np.zeros(na, dtype=np.float64)
        for a in range(na):
            if game.ACTIONS[a] not in legal_set:
                continue
            inst = util[a] - node_util
            regret_vec[a] = (p1 if player == 0 else p0) * inst
        self.buffer.add(features, regret_vec)

        return node_util


# ------------------------------------------------------------------
# Avoid circular import; torch is imported here rather than at the top
# so that model.py can import game without triggering a full chain.
# ------------------------------------------------------------------
import torch
