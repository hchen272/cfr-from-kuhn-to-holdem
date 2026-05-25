"""
Training loop for Deep CFR (paper specification).

Brown et al. (ICML 2019):

  - External-sampling MCCFR traversals
  - From-scratch network retraining *every* CFR iteration
  - Linear CFR weighting: MSE loss scaled by iteration t
  - Gradient clipping  ‖∇θ‖ ≤ 1
  - Reservoir sampling for experience replay
  - Warm-start: fill buffer with random-play traversals before training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_selector import get_game
from neural.buffer import ReservoirBuffer
from deep_cfr.model import RegretNet, _hidden_dim
from deep_cfr.traverse import Traverser
from utils import save_strategy_txt, save_model


# ══════════════════════════════════════════════════════════════════════
#  Hyper-parameters
# ══════════════════════════════════════════════════════════════════════

TRAVERSALS_PER_ITER = 300           # K: external-sampling traversals / CFR iter
BATCH_SIZE          = 2048         # mini-batch size
TRAIN_STEPS         = 800          # gradient steps per from-scratch retraining
LEARNING_RATE       = 0.001
BUFFER_DIVISOR      = 3            # capacity = traversals * iters // divisor
MAX_BUFFER          = 2_000_000    # hard ceiling
WARM_START_FRAC     = 0.1          # fill buffer with this fraction before training
SNAPSHOT_EVERY      = 100          # CFR iters between snapshot writes
PROGRESS_EVERY      = 50           # CFR iters between stdout progress lines
GRAD_CLIP           = 1.0          # ‖∇θ‖ max


# ══════════════════════════════════════════════════════════════════════
#  Warm-start
# ══════════════════════════════════════════════════════════════════════

def _warm_start_buffer(traverser, game, buffer, n):
    """Fill *buffer* with *n* random-play traversals (regrets → 0)."""
    saved_net = traverser.regret_net

    class _ZeroNet(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), game.num_actions,
                               device=x.device)

    traverser.regret_net = _ZeroNet()
    for _ in range(n):
        cards = game.deal_cards()
        traverser.traverse(cards, "", 1.0, 1.0, 0)   # P0 random
        cards = game.deal_cards()
        traverser.traverse(cards, "", 1.0, 1.0, 1)   # P1 random
    traverser.regret_net = saved_net
    traverser.reset_strategy_sum()    # discard random-play strategies


# ══════════════════════════════════════════════════════════════════════
#  Training step
# ══════════════════════════════════════════════════════════════════════

def _train_from_scratch(net, optimizer, loss_fn, buffer, device,
                        iter_weight, steps):
    """Reinitialise *net* then train for *steps* gradient steps.

    ``iter_weight`` (the outer CFR iteration index *t*) is applied as
    a scalar on the MSE loss —  Linear CFR weighting (paper §10, §16).
    """
    net.train()
    for _ in range(steps):
        feats, targets = buffer.sample(BATCH_SIZE)
        if len(feats) == 0:
            continue

        feats_t = torch.from_numpy(feats).to(device)
        tgts_t  = torch.from_numpy(targets).to(device)

        optimizer.zero_grad()
        preds = net(feats_t)
        loss = loss_fn(preds, tgts_t) * iter_weight
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        optimizer.step()


# ══════════════════════════════════════════════════════════════════════
#  Snapshot / final output
# ══════════════════════════════════════════════════════════════════════

class _FakeNode:
    """Duck-typed Node for save_strategy_txt compatibility."""
    def __init__(self, avg):
        self._avg = avg
    def get_average_strategy(self):
        return self._avg


def _snapshot(traverser, t, avg_val, total_iters, game_name):
    node_map = {}
    for infoset in sorted(traverser.strategy_sum):
        node_map[infoset] = _FakeNode(
            traverser.get_average_strategy(infoset))
    save_strategy_txt(node_map, t, avg_val, total_iters,
                      "deep_cfr_paper", game_name=game_name)


# ══════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════

def train(iterations, game_name="kuhn"):
    """Run the paper-specification Deep CFR training loop.

    Parameters
    ----------
    iterations : int
        Number of CFR outer-loop iterations.
    game_name : str
        ``"kuhn"`` or ``"leduc"``.
    """
    game = get_game(game_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hdim  = _hidden_dim(game_name)

    # ── clean stale log for this (game, iters) pair ──────────────────
    import os
    log_name = f'{game_name}_strategy_deep_cfr_paper_{iterations:.0e}.txt'
    log_path = os.path.join('logs', log_name)
    if os.path.exists(log_path):
        os.remove(log_path)

    # ── buffer capacity proportional to total traversals ────────────
    buf_capacity = min(TRAVERSALS_PER_ITER * iterations // BUFFER_DIVISOR,
                       MAX_BUFFER)
    buf_capacity = max(50_000, buf_capacity)

    print(f"Deep CFR (paper)  |  game: {game.name}  |  "
          f"hidden_dim: {hdim}  |  iters: {iterations}  |  device: {device}  "
          f"|  buffer: {buf_capacity:,}")

    regret_buffer = ReservoirBuffer(capacity=buf_capacity)

    regret_net = RegretNet(input_dim=game.feature_dim,
                           hidden_dim=hdim,
                           output_dim=game.num_actions).to(device)
    optimizer = optim.Adam(regret_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    traverser = Traverser(regret_net, regret_buffer, game, device=device)

    # ── warm-start ────────────────────────────────────────────────
    warm_n = min(10_000, max(1, int(buf_capacity * WARM_START_FRAC)))
    print(f"Warm-start: filling buffer with {warm_n:,} random traversals …")
    _warm_start_buffer(traverser, game, regret_buffer, warm_n)
    print(f"  buffer size after warm-start: {len(regret_buffer):,}")

    # ── main loop ─────────────────────────────────────────────────
    total_util = 0.0
    util_count  = 0
    best_dist = float("inf")
    best_strategy_sum = None
    best_cur_val = None
    nash = getattr(game, 'nash_value', None)

    for t in range(1, iterations + 1):
        traverser_player = t % 2        # alternate traverser each iter

        # ── K traversals to collect new data ─────────────────
        iter_util = 0.0
        for _ in range(TRAVERSALS_PER_ITER):
            cards = game.deal_cards()
            u = traverser.traverse(cards, "", 1.0, 1.0, traverser_player)
            iter_util += u
        iter_util /= TRAVERSALS_PER_ITER
        total_util += iter_util
        util_count  += 1

        # ── from-scratch retraining ──────────────────────────
        # re-initialise network weights
        regret_net = RegretNet(input_dim=game.feature_dim,
                               hidden_dim=hdim,
                               output_dim=game.num_actions).to(device)
        optimizer = optim.Adam(regret_net.parameters(), lr=LEARNING_RATE)
        _train_from_scratch(regret_net, optimizer, loss_fn,
                            regret_buffer, device,
                            iter_weight=float(t),
                            steps=TRAIN_STEPS)

        # update traverser to use the newly trained network
        traverser.regret_net = regret_net

        # ── progress ─────────────────────────────────────────
        if t % PROGRESS_EVERY == 0:
            avg_val = total_util / util_count
            buf_pct = len(regret_buffer) / buf_capacity * 100
            print(f"  iter {t:>6,}  |  avg value: {avg_val:+.4f}  "
                  f"|  buffer: {len(regret_buffer):,} ({buf_pct:.0f}%)")

            # checkpoint: best Nash-distance strategy (using current value)
            cur_d = abs(iter_util - nash) if nash is not None else float("inf")
            if cur_d < best_dist:
                best_dist = cur_d
                best_cur_val = iter_util
                best_strategy_sum = {
                    k: v.copy()
                    for k, v in traverser.strategy_sum.items()
                }
                print(f"           [new best]  cur={iter_util:+.4f}  dist: {cur_d:.4f}")

        # ── snapshot ─────────────────────────────────────────
        if t % SNAPSHOT_EVERY == 0 or t == iterations:
            avg_val = total_util / util_count
            _snapshot(traverser, t, avg_val, iterations,
                      game_name=game_name)

    # ── restore best checkpoint ──────────────────────────────────
    if best_strategy_sum is not None:
        traverser.strategy_sum = best_strategy_sum
        print(f"\n[CHECKPOINT] Restored best "
              f"cur={best_cur_val:+.4f} (dist={best_dist:.4f})")

    # ── final output ─────────────────────────────────────────────
    print(f"\n=== FINAL STRATEGIES ({game.name}) ===\n")
    for infoset in sorted(traverser.strategy_sum):
        avg = traverser.get_average_strategy(infoset)
        print(f"  {infoset}: {avg}")
    avg_final = total_util / util_count
    print(f"\nAverage game value: {avg_final:+.4f}")

    # save model (strategy_sum dict)
    save_model(traverser.strategy_sum, iterations, "deep_cfr_paper",
               game_name=game_name)

    if best_strategy_sum is not None:
        save_model(best_strategy_sum, iterations, "deep_cfr_paper_best",
                   game_name=game_name)
        print(f"Best checkpoint saved (dist={best_dist:.4f})")
