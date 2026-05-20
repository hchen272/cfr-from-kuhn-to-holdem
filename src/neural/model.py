"""
Neural network model for Deep CFR.

RegretNet maps an information-set feature vector to
counterfactual regret values for each legal action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from node import NUM_ACTIONS


def infoset_to_features(infoset: str):
    """
    Convert an infoset string (e.g. ``"J"``, ``"Kp"``, ``"Qpb"``)
    into a fixed-size numeric feature vector.

    Features (dim = 5):
        [is_J, is_Q, is_K, first_action, second_action]

    ``first_action`` / ``second_action``:
        -1 = no action yet
         0 = pass/check/fold
         1 = bet/call
    """
    card = infoset[0]                       # 'J', 'Q', 'K'
    history = infoset[1:]                    # remaining betting history

    # -- card one-hot ------------------------------------------------
    card_vec = [1.0 if card == c else 0.0 for c in ("J", "Q", "K")]

    # -- history encoding --------------------------------------------
    slot = [-1.0, -1.0]                     # -1 = slot unused
    for i, action in enumerate(history):
        slot[i] = 0.0 if action == "p" else 1.0

    return card_vec + slot                  # 3 + 2 = 5 floats


class RegretNet(nn.Module):
    """Small fully-connected network that predicts action regrets.

    Architecture:
        input (5) → FC(64) → ReLU → FC(64) → ReLU → FC(2)
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)                  # raw regret logits


def get_strategy_from_regrets(regrets):
    """Regret matching: σ(a) ∝ max(regret[a], 0)."""
    strategy = regrets.copy()
    strategy = np.maximum(strategy, 0.0)
    total = strategy.sum()
    if total > 0:
        strategy /= total
    else:
        strategy[:] = 1.0 / NUM_ACTIONS
    return strategy


# Lazy import for type-hint convenience
import numpy as np
