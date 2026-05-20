"""
Neural network model for Deep CFR.

RegretNet maps an information-set feature vector to
counterfactual regret values for each legal action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from node import NUM_ACTIONS


class RegretNet(nn.Module):
    """Small fully-connected network that predicts action regrets.

    Architecture:
        input (feature_dim) → FC(64) → ReLU → FC(64) → ReLU → FC(num_actions)
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64,
                 output_dim: int = NUM_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)                  # raw regret logits


def get_strategy_from_regrets(regrets, num_actions=NUM_ACTIONS):
    """Regret matching: σ(a) ∝ max(regret[a], 0)."""
    strategy = regrets.copy()
    strategy = np.maximum(strategy, 0.0)
    total = strategy.sum()
    if total > 0:
        strategy /= total
    else:
        strategy[:] = 1.0 / num_actions
    return strategy


# Lazy import for type-hint convenience
import numpy as np
