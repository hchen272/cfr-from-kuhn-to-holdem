"""
MC-CFR Trainer for Texas Hold'em.

Independent training CLI — does NOT inherit from ``src/trainer.py``.
Uses External Sampling MC-CFR with the on-the-fly tree.

Shared dependencies (import from ``src/``):
    - ``algo.tabular.node.Node``
    - ``utils`` (save_model, load_model)

Usage
-----
    cd myCFR
    python holdem/trainer.py -i 1000000 --buckets 100
"""
import sys
import os
import copy
import json
import argparse
import time
from collections import deque

import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────
_HOLDEM_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HOLDEM_DIR)

sys.path.insert(0, os.path.join(_PROJ_ROOT, 'src'))
sys.path.insert(0, _HOLDEM_DIR)

from game import TexasHoldemGame, _card_str
from tree import OnTheFlyTree
from abstraction import CardAbstraction
from cfr import mccfr
from algo.tabular.node import Node

# ── Save helpers ─────────────────────────────────────────────────────────
import pickle

_MODELS_DIR = os.path.join(_PROJ_ROOT, "models")
_LOGS_DIR = os.path.join(_PROJ_ROOT, "logs")
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_LOGS_DIR, exist_ok=True)

try:
    from utils import save_model, load_model
except ImportError:
    def save_model(node_map, iterations, algorithm, game_name="texas_holdem"):
        filename = f"{game_name}_{algorithm}_{iterations:.0e}.pkl"
        filepath = os.path.join(_MODELS_DIR, filename)
        with open(filepath, "wb") as f:
            pickle.dump(node_map, f)
        print(f"Model saved to {filepath}")


def _save_final_strategy(node_map, iterations, algo_key):
    """Save full strategy txt once, at end of training."""
    filename = f"texas_holdem_strategy_{algo_key}_{iterations:.0e}.txt"
    filepath = os.path.join(_LOGS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== MC-CFR STRATEGY SNAPSHOT ===\n\n")
        f.write(f"Iterations: {iterations}\n")
        for infoset in sorted(node_map):
            avg_strat = node_map[infoset].get_average_strategy()
            f.write(f"{infoset}: {avg_strat}\n")
    print(f"Strategy log saved to {filepath}")


def _save_stats(stats_list, iterations, algo_key):
    """Save scalar stats as JSON for plotting."""
    filename = f"texas_holdem_stats_{algo_key}_{iterations:.0e}.json"
    filepath = os.path.join(_LOGS_DIR, filename)
    payload = {
        "game": "texas_holdem",
        "algorithm": algo_key,
        "total_iterations": iterations,
        "history": stats_list,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Stats saved to {filepath}")


# ── Trainer ────────────────────────────────────────────────────────

_REPORT_EVERY = 100         # how often to print & log stats
_CHECKPOINT_PCT = 10        # save a checkpoint every N% of training
_BEST_FROM_LAST = 3          # pick best from the last N checkpoints


class HoldemTrainer:
    """MC-CFR trainer for Texas Hold'em."""

    def __init__(self, n_buckets=50, n_mc_samples=2000):
        self.n_buckets = n_buckets
        self.algo_key = f"mccfr_b{n_buckets}"

        print(f"Initializing abstraction ({n_buckets} buckets, "
              f"{n_mc_samples} MC samples)...")
        t0 = time.time()
        self.abstraction = CardAbstraction(n_buckets=n_buckets,
                                           n_mc_samples=n_mc_samples)
        print(f"  done in {time.time() - t0:.1f}s")

        self.game = TexasHoldemGame()
        self.tree = OnTheFlyTree(self.game, self.abstraction)
        self.game_name = self.game.name

    def train(self, iterations):
        game = self.game
        tree = self.tree
        absn = self.abstraction

        node_map = {}
        total_util = 0.0

        # Rolling window for smoothing (large enough to dampen MCCFR noise)
        window = max(100, iterations // 20)
        recent_vals = deque(maxlen=window)
        report_interval = max(1, iterations // _REPORT_EVERY)

        # Stats history for plotting   [{iter, avg, cur, roll_avg, roll_std, nodes}, …]
        stats_history = []

        # Checkpoint tracking
        checkpoint_interval = max(1, iterations * _CHECKPOINT_PCT // 100)
        checkpoints = []          # [(iter, node_map, roll_avg, roll_std), …]

        t_start = time.time()
        algo_key = self.algo_key

        print(f"\n{'='*55}")
        print(f"  Texas Hold'em MC-CFR Training")
        print(f"  Buckets: {self.n_buckets}  |  Iterations: {iterations:,}")
        print(f"  Checkpoint: every {_CHECKPOINT_PCT}% ({checkpoint_interval:,} iters)")
        print(f"  Game: Fixed-Limit, Heads-up")
        print(f"{'='*55}\n")

        for t in range(1, iterations + 1):
            cards = game.deal_cards()
            b0 = absn.bucket_id(cards[0])
            b1 = absn.bucket_id(cards[1])

            traverser = 0 if t % 2 == 1 else 1

            iter_util = mccfr(tree, cards, "", node_map,
                              traverser=traverser, bucket0=b0, bucket1=b1)

            total_util += iter_util
            recent_vals.append(iter_util)

            # ── Report + stats collection ──
            if t % report_interval == 0 or t == iterations:
                avg_value = total_util / t
                cur_value = iter_util
                roll_avg = sum(recent_vals) / len(recent_vals) if recent_vals else 0.0
                roll_std = float(np.std(list(recent_vals))) if len(recent_vals) > 1 else 0.0
                pct = t / iterations * 100
                elapsed = time.time() - t_start
                iters_per_sec = t / elapsed if elapsed > 0 else 0

                print(f"[{pct:5.1f}%] Iter {t:>10,}  |  "
                      f"avg: {avg_value:+.6f}  cur: {cur_value:+.6f}  "
                      f"roll: {roll_avg:+.6f}  std: {roll_std:+.4f}  "
                      f"|  {iters_per_sec:.1f} it/s  "
                      f"nodes: {len(node_map)}")

                stats_history.append({
                    "iter": t,
                    "avg": round(avg_value, 6),
                    "cur": round(cur_value, 6),
                    "roll_avg": round(roll_avg, 6),
                    "roll_std": round(roll_std, 6),
                    "nodes": len(node_map),
                    "elapsed_s": round(elapsed, 1),
                })

            # ── Checkpoint every N% ──
            if t % checkpoint_interval == 0:
                roll_avg = (sum(recent_vals) / len(recent_vals)
                            if recent_vals else 0.0)
                roll_std = (float(np.std(list(recent_vals)))
                            if len(recent_vals) > 1 else 0.0)
                cp_copy = {k: copy.deepcopy(v) for k, v in node_map.items()}
                checkpoints.append((t, cp_copy, roll_avg, roll_std))
                print(f"  >>> checkpoint at iter {t:,} "
                      f"(roll_avg={roll_avg:+.4f}, std={roll_std:.4f})")

        # ── End of training ──
        elapsed = time.time() - t_start
        print(f"\n{'='*55}")
        print(f"  Training complete: {iterations:,} iters in {elapsed:.1f}s")
        print(f"  Avg game value: {total_util / iterations:+.6f}")
        print(f"  Total infosets: {len(node_map)}")
        print(f"  Checkpoints saved: {len(checkpoints)}")
        print(f"{'='*55}")

        conv = node_map  # already string-keyed

        # 1. Save final model
        save_model(conv, iterations, algo_key, game_name=self.game_name)

        # 2. Pick best checkpoint from the last few (lowest rolling_std)
        if checkpoints:
            n_consider = min(_BEST_FROM_LAST, len(checkpoints))
            recent_cps = checkpoints[-n_consider:]
            best_cp = min(recent_cps, key=lambda x: x[3])  # sort by roll_std
            best_iter, best_map, best_avg, best_std = best_cp
            save_model(best_map, best_iter, algo_key + "_best",
                       game_name=self.game_name)
            print(f"Best checkpoint: iter {best_iter:,} "
                  f"(roll_avg={best_avg:+.4f}, std={best_std:.4f})")

            # Also save the very last checkpoint as an explicit milestone
            last_cp = checkpoints[-1]
            if last_cp[0] != best_iter:
                save_model(last_cp[1], last_cp[0], algo_key + "_last",
                           game_name=self.game_name)

        # 3. Save final strategy txt (only at end)
        _save_final_strategy(conv, iterations, algo_key)

        # 4. Save stats JSON for plotting
        _save_stats(stats_history, iterations, algo_key)

        return node_map


# ── CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MC-CFR Trainer for Texas Hold'em (Fixed-Limit)")
    parser.add_argument("--iterations", "-i", type=int, default=100000,
                        help="Number of MC-CFR iterations (default: 100000)")
    parser.add_argument("--buckets", "-b", type=int, default=50,
                        help="Card abstraction buckets (default: 50)")
    parser.add_argument("--mc-samples", type=int, default=2000,
                        help="MC samples for equity estimation (default: 2000)")
    args = parser.parse_args()

    trainer = HoldemTrainer(n_buckets=args.buckets,
                            n_mc_samples=args.mc_samples)
    trainer.train(args.iterations)


if __name__ == '__main__':
    main()
