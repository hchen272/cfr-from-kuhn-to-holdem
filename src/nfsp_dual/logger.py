"""Dual-strategy log helpers for nfsp_dual."""
import os
import numpy as np


def format_probs(probs):
    """Return '[p0 p1 p2]' string."""
    return "[" + " ".join(f"{p:.6e}" for p in probs) + "]"


def save_dual_log(p0_strategies, p1_strategies, iteration, avg_reward,
                  total_iters, game_name="leduc", log_dir="logs"):
    """Write a snapshot showing both P0 and P1 strategies side by side.

    Parameters
    ----------
    p0_strategies : dict  {infoset_key: np.array}
    p1_strategies : dict  {infoset_key: np.array}
    iteration : int
    avg_reward : float
    total_iters : int
    """
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{game_name}_strategy_nfsp_dual_pair_{total_iters:.0e}.txt"
    filepath = os.path.join(log_dir, filename)

    mode = "w" if iteration <= 0 else "a"
    with open(filepath, mode, encoding="utf-8") as f:
        f.write("=" * 70 + "\n\n")
        f.write(f"Iterations: {iteration}\n")
        all_keys = sorted(set(p0_strategies.keys()) | set(p1_strategies.keys()))
        for key in all_keys:
            n0 = p0_strategies.get(key)
            n1 = p1_strategies.get(key)
            p0p = n0.get_average_strategy() if hasattr(n0, 'get_average_strategy') else np.ones(3)/3
            p1p = n1.get_average_strategy() if hasattr(n1, 'get_average_strategy') else np.ones(3)/3
            f.write(f"{key}: {format_probs(p0p)}  {format_probs(p1p)}\n")
        f.write(f"Average game value: {avg_reward:.4f}\n\n")
