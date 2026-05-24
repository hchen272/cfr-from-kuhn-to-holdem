"""CFR training on Leduc Hold'em — rlcard_like.

Usage:
    python rlcard_like/train.py
    python rlcard_like/train.py -i 100000
    python rlcard_like/train.py -i 50000 --eval-every 1000 --eval-games 2000
"""
import argparse
import sys
import os
import time
import pickle
import numpy as np

# ensure the project root is on the path so we can import rlcard_like
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcard_like.games.leducholdem.game import LeducholdemGame
from rlcard_like.agents.cfr_agent import CFRAgent

# ── action names for display ─────────────────────────────────────
ACTION_NAMES = ["call", "raise", "fold"]


def format_strategy(probs):
    """Return [p0, p1, p2] string."""
    return "[" + " ".join(f"{p:.4f}" for p in probs) + "]"


def save_snapshot(agent, iteration, payoff, total_iters, log_dir, game_name="leduc"):
    """Write strategy snapshot in the same format as main project's logs."""
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{game_name}_strategy_rlcard_cfr_{total_iters:.0e}.txt"
    filepath = os.path.join(log_dir, filename)

    mode = "w" if iteration <= agent._snap_every else "a"
    with open(filepath, mode) as f:
        f.write("=" * 50 + "\n\n")
        f.write(f"Iterations: {iteration}\n")
        for obs in sorted(agent.average_policy.keys()):
            avg = agent.average_policy[obs]
            total = avg.sum()
            probs = avg / total if total > 0 else np.ones_like(avg) / len(avg)
            f.write(f"{obs}: {format_strategy(probs)}\n")
        f.write(f"Average game value: {payoff:.4f}\n\n")


def evaluate(env, agent, num_games=1000):
    """Self-play evaluation using the *average* policy. Returns mean P0 payoff."""
    payoffs = []
    for _ in range(num_games):
        state, _ = env.init_game()
        while not env.is_over():
            action, _ = agent.eval_step(state)
            state, _ = env.step(action)
        payoffs.append(env.get_payoffs()[0])
    return float(np.mean(payoffs))


def main():
    parser = argparse.ArgumentParser(description="CFR on Leduc Hold'em (rlcard-like)")
    parser.add_argument("--iterations", "-i", type=int, default=50000,
                        help="Number of CFR iterations")
    parser.add_argument("--eval-every", type=int, default=5000,
                        help="Evaluate every N iterations")
    parser.add_argument("--eval-games", type=int, default=2000,
                        help="Number of games per evaluation")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for strategy snapshots")
    parser.add_argument("--log-every", type=int, default=5000,
                        help="Save strategy snapshot every N iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save", type=str, default="rlcard_like/model.pkl",
                        help="Path to save the final average policy")
    args = parser.parse_args()

    np.random.seed(args.seed)

    env = LeducholdemGame(allow_step_back=True)
    agent = CFRAgent(env)
    agent._snap_every = args.log_every  # used by save_snapshot for header mode

    print(f"CFR on Leduc Hold'em")
    print(f"  iterations : {args.iterations:,}")
    print(f"  eval every : {args.eval_every:,}")
    print(f"  eval games : {args.eval_games:,}")
    print(f"  log   every: {args.log_every:,}")
    print(f"  log   dir  : {args.log_dir}")
    print(f"  Nash value : {env.NASH_VALUE}")
    print(f"  seed       : {args.seed}")
    print()

    # ── clean old log for this run ──
    log_name = f"leduc_strategy_rlcard_cfr_{args.iterations:.0e}.txt"
    log_path = os.path.join(args.log_dir, log_name)
    if os.path.exists(log_path):
        os.remove(log_path)

    best_payoff = -float("inf")
    best_iter = 0
    best_policy = None
    t0 = time.time()

    for i in range(1, args.iterations + 1):
        agent.train()

        # ── snapshot ──
        if i % args.log_every == 0 or i == args.iterations:
            avg = evaluate(env, agent, num_games=args.eval_games)
            save_snapshot(agent, i, avg, args.iterations, args.log_dir)

        # ── eval + progress ──
        if i % args.eval_every == 0 or i == args.iterations:
            avg = evaluate(env, agent, num_games=args.eval_games)
            elapsed = time.time() - t0
            dist = abs(avg - env.NASH_VALUE)
            print(f"  iter {i:>7,}  |  payoff: {avg:+.4f}  "
                  f"|  dist to Nash: {dist:.4f}  "
                  f"|  infosets: {len(agent.average_policy)}  "
                  f"|  time: {elapsed:.0f}s")

            if dist < abs(best_payoff - env.NASH_VALUE):
                best_payoff = avg
                best_iter = i
                best_policy = {
                    k: v.copy() for k, v in agent.average_policy.items()
                }
                print(f"           [new best]")

    # ── final output ──
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  Training complete  |  {args.iterations:,} iters  |  {elapsed:.0f}s")
    print(f"  Best payoff : {best_payoff:+.4f}  (iter {best_iter})")
    print(f"  Nash        : {env.NASH_VALUE}")
    print(f"{'='*55}")

    if best_policy is not None:
        save_dir = os.path.dirname(args.save)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save, "wb") as f:
            pickle.dump(best_policy, f)
        print(f"\nBest policy saved to {args.save}")
        print(f"Policy size: {len(best_policy)} infosets")


if __name__ == "__main__":
    main()
