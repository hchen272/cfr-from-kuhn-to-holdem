"""Q-Network for DQN: infoset features → Q-values per action."""
import torch
import torch.nn as nn


def _hidden_dim(game_name: str) -> int:
    return {'kuhn': 64, 'leduc': 128}.get(game_name, 128)


class QNetwork(nn.Module):
    """MLP: feature_dim → hidden → hidden → num_actions."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
