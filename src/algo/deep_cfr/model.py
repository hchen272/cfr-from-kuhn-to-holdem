"""
Deep CFR neural network (paper specification).

Brown et al. (ICML 2019):
  - 7 fully connected layers with skip connections
  - ReLU activation
  - Optional skip connection when dimensions match:  x_{i+1} = ReLU(W x_i [+ x_i])
  - Feature normalisation (zero mean, unit variance) on final hidden layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hidden_dim(game_name: str) -> int:
    """Game-adaptive hidden layer size matching paper's ~100K param scale."""
    return {'kuhn': 64, 'leduc': 128}.get(game_name, 128)


class RegretNet(nn.Module):
    """7-layer MLP with skip connections, per the Deep CFR paper.

    Layers:  fc1 → fc2 → fc3 → fc4 → fc5 → fc6 → fc7
    Skips:   [skip1]   [skip2]   [skip3]   [skip4]   [skip5]   [skip6]
             (skip when in_dim == hidden_dim, i.e. layers 2-6)

    Final hidden features are normalised to zero mean / unit variance
    before the output projection (Section 15.2 of the paper).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 output_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # ── layer stack ─────────────────────────────────────────
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)

        # layer norm for final feature normalisation (applied before fc7)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # fc1 (no skip — input_dim ≠ hidden_dim in general)
        out = F.relu(self.fc1(x))

        # fc2–fc6  with skip connections (hidden_dim == hidden_dim)
        out = F.relu(self.fc2(out) + out)
        out = F.relu(self.fc3(out) + out)
        out = F.relu(self.fc4(out) + out)
        out = F.relu(self.fc5(out) + out)
        out = F.relu(self.fc6(out) + out)

        # normalise final hidden features → zero mean, unit variance
        out = self.norm(out)

        # output projection (no activation — raw regret logits)
        return self.fc7(out)


# ──────────────────────────────────────────────────────────────────
#  Regret matching (same as neural/model.py)
# ──────────────────────────────────────────────────────────────────

def get_strategy_from_regrets(regrets, num_actions):
    """σ(a) ∝ max(regret[a], 0).

    When all regrets ≤ 0: uniform random (standard CFR fallback).
    The paper's §2.2 "highest-regret" fallback is safe only when the
    network rarely outputs all-negative — at our scale, it triggers
    constantly and locks strategies into deterministic corners,
    killing all exploration.
    """
    import numpy as np
    strategy = np.maximum(regrets, 0.0)
    total = strategy.sum()
    if total > 0:
        strategy /= total
    else:
        strategy[:] = 1.0 / num_actions
    return strategy
