"""
Training orchestration for Deep CFR.

Usage (from project root)::

    python src/trainer.py -a deep_cfr -g kuhn
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_selector import get_game
from tabular.node import NUM_ACTIONS
from neural.model import RegretNet, get_strategy_from_regrets
from neural.buffer import ReservoirBuffer
from neural.deep_cfr import DeepCFR
from utils import save_strategy_txt, save_model

# ──────────────────────────────────────────────────────────────────────
#  Hyper-parameters
# ──────────────────────────────────────────────────────────────────────
BATCH_SIZE = 512
TRAIN_FREQ = 20                   # train the network every K episodes
TRAIN_STEPS = 10                  # gradient steps per training event
LEARNING_RATE = 0.001
WARM_START_RATIO = 10             # 1/WARM_START_RATIO of buffer filled with random play
PROGRESS_EVERY = 5000             # print a one-line progress every N episodes


def _hidden_dim(game_name: str) -> int:
    """Game-adaptive hidden-layer size."""
    return {'kuhn': 32, 'leduc': 256}.get(game_name, 64)


def _buffer_capacity(iterations):
    """Dynamically-sized buffer: bigger for more iterations so that each
    buffer entry is not re-sampled too many times.

    Formula:  floor(iterations × 0.2),  clamped to [50 000, 500 000].
    """
    return max(50_000, min(int(iterations * 0.2), 500_000))


def train_deep_cfr(iterations, log_prefix="deep_cfr", game_name="kuhn",
                   alternate=False):
    """Run the full Deep CFR training loop.

    Parameters
    ----------
    iterations : int
    log_prefix : str
        Algorithm name used in log/model filenames.
    game_name : str
        Game name used in file naming (e.g. 'kuhn', 'leduc').
    alternate : bool
        If True, swap the updating player every episode (recommended).

    Returns
    -------
    agent : DeepCFR
        Trained agent whose ``strategy_sum`` dict holds the average
        strategies.
    """
    game = get_game(game_name)
    print(f"Game: {game.name}  |  hidden_dim: {_hidden_dim(game_name)}  "
          f"|  alternate: {alternate}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # --- initialise components ---------------------------------------
    capacity = _buffer_capacity(iterations)
    regret_net = RegretNet(input_dim=game.feature_dim,
                           hidden_dim=_hidden_dim(game_name),
                           output_dim=game.num_actions).to(device)
    optimizer = optim.Adam(regret_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    buffer = ReservoirBuffer(capacity=capacity)
    agent = DeepCFR(regret_net, buffer, game, device=device)
    node_map_for_logging = {}      # see note below

    # --- warm-start: fill buffer with a small amount of random play --
    warm_start_n = max(1, capacity // WARM_START_RATIO)
    _warm_start_buffer(buffer, n=warm_start_n, agent=agent, game=game)

    # --- main training loop ------------------------------------------
    from collections import deque
    window = max(20, iterations // 50)
    recent_vals = deque(maxlen=window)
    snapshot_every = max(1, iterations // 100)

    # Checkpoint: track strategy closest to Nash equilibrium value
    nash = getattr(game, 'nash_value', None)
    best_dist = float("inf")
    best_strategy_sum = None
    best_cur_val = None

    for episode in range(1, iterations + 1):
        cards = game.deal_cards()

        up = episode % 2 if alternate else -1   # -1 = both
        episode_util = agent.traverse(cards, "", 1.0, 1.0,
                                      update_player=up)
        recent_vals.append(episode_util)

        # --- train network every TRAIN_FREQ episodes -----------------
        if episode % TRAIN_FREQ == 0 and len(buffer) >= BATCH_SIZE:
            _train_step(regret_net, optimizer, loss_fn, buffer,
                        BATCH_SIZE, device, steps=TRAIN_STEPS)

        # --- lightweight one-line progress ---------------------------
        if episode % PROGRESS_EVERY == 0:
            roll_avg = sum(recent_vals) / len(recent_vals)
            print(f"  Episode {episode:>8,}  |  roll avg: {roll_avg:+.4f}  "
                  f"|  buffer: {len(buffer):,}")

            # Checkpoint: save if closer to Nash value (using current value)
            cur_d = abs(roll_avg - nash) if nash is not None else float("inf")
            if cur_d < best_dist:
                best_dist = cur_d
                best_cur_val = roll_avg
                best_strategy_sum = {
                    k: v.copy() for k, v in agent.strategy_sum.items()
                }
                print(f"           [new best] roll={roll_avg:+.4f} (dist={cur_d:.4f})")

        # --- full snapshot (log + strategies) ------------------------
        if episode % snapshot_every == 0:
            roll_avg = sum(recent_vals) / len(recent_vals)
            _log_snapshot(agent, episode, roll_avg, iterations,
                          log_prefix, node_map_for_logging,
                          game_name=game_name)

    # --- restore best checkpoint for final output --------------------
    if best_strategy_sum is not None:
        agent.strategy_sum = best_strategy_sum
        print(f"\n[CHECKPOINT] Restored best "
              f"roll={best_cur_val:+.4f} (dist={best_dist:.4f})")

    # --- final output ------------------------------------------------
    roll_avg = sum(recent_vals) / max(len(recent_vals), 1)
    print(f"\n=== FINAL STRATEGIES ===\n")
    for infoset in sorted(agent.strategy_sum):
        avg = agent.get_average_strategy(infoset)
        print(f"{infoset}: {avg}")
    print(f"Rolling avg game value: {roll_avg:+.4f}\n")

    # save (_FakeNode-wrapped → exploitability-compatible)
    exp_map = {k: _FakeNode(agent.get_average_strategy(k))
               for k in agent.strategy_sum}
    save_model(exp_map, iterations, log_prefix, game_name=game_name)

    if best_strategy_sum is not None:
        saved = dict(agent.strategy_sum)
        agent.strategy_sum = best_strategy_sum
        exp_best = {k: _FakeNode(agent.get_average_strategy(k))
                    for k in best_strategy_sum}
        agent.strategy_sum = saved
        save_model(exp_best, iterations, log_prefix + "_best", game_name=game_name)
        print(f"Best checkpoint saved (dist={best_dist:.4f})")
    return agent


# ══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════

def _warm_start_buffer(buffer, n, agent, game):
    """Fill the buffer with random-strategy traversals."""
    saved_net = agent.regret_net
    # temporarily use a uniform-random strategy by zeroing regrets
    class _RandomNet(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), game.num_actions)
    agent.regret_net = _RandomNet()
    for _ in range(n):
        cards = game.deal_cards()
        agent.traverse(cards, "", 1.0, 1.0)
    agent.regret_net = saved_net


def _train_step(net, optimizer, loss_fn, buffer, batch_size, device, steps=1):
    """Perform *steps* gradient steps on mini-batches from the buffer."""
    for _ in range(steps):
        features, targets = buffer.sample(batch_size)
        feats_t = torch.from_numpy(features).to(device)
        tgts_t  = torch.from_numpy(targets).to(device)

        optimizer.zero_grad()
        preds = net(feats_t)
        loss = loss_fn(preds, tgts_t)
        loss.backward()
        optimizer.step()


def _log_snapshot(agent, episode, avg_val, total_iters, algo_prefix,
                  node_map_for_logging, game_name="kuhn"):
    """Print progress and write strategy snapshot in the same format
    as tabular CFR, so that ``visualize.py`` can parse it.

    We build a lightweight node_map_for_logging that stores
    ``get_average_strategy()`` as a ``Node``-like object so the
    existing ``save_strategy_txt`` works.
    """
    print(f"Iteration {episode}")
    print(f"Average game value: {avg_val:.4f}")
    print("-" * 40)

    # Build a fake node-map for save_strategy_txt
    for infoset in sorted(agent.strategy_sum):
        avg = agent.get_average_strategy(infoset)
        node_map_for_logging[infoset] = _FakeNode(avg)

    save_strategy_txt(node_map_for_logging, episode, avg_val,
                      total_iters, algo_prefix, game_name=game_name)


def _print_final_strategies(agent, total_util, iterations):
    """Print the final average strategies to stdout."""
    print("\n=== FINAL STRATEGIES ===\n")
    for infoset in sorted(agent.strategy_sum):
        avg = agent.get_average_strategy(infoset)
        print(f"{infoset}: {avg}")
    print(f"Average game value: {total_util / iterations:.4f}\n")


class _FakeNode:
    """Minimal duck-typed object that mimics ``Node`` for logging."""
    def __init__(self, avg_strategy):
        self._avg = avg_strategy
    def get_average_strategy(self):
        return self._avg


# ══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep CFR — Kuhn Poker")
    parser.add_argument("--iterations", "-i", type=int, default=1000000,
                        help="Number of training episodes")
    args = parser.parse_args()

    train_deep_cfr(iterations=args.iterations)
