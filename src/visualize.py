"""Visualize strategy evolution and game value from training logs.

Output goes to eval/visualizations/ (or custom --output-dir).
Supports both "Average game value" and "Current game value" lines.
"""
import os, re, glob, argparse
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_NASH = {
    "kuhn":  (-1/18, f"Nash (-1/18 ≈ {-1/18:.4f})"),
    "leduc": (-0.0855, "Nash (≈ -0.0855)"),
}


def parse_log(filepath):
    """Return dicts for per-infoset history and game-value progression."""
    infoset_data = {}
    gv = {"iters": [], "avg": [], "cur": []}
    cur_iter = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Iterations:"):
                cur_iter = int(line.split(":")[1].strip())
            elif line.startswith("Average game value:"):
                gv["iters"].append(cur_iter)
                gv["avg"].append(float(line.split(":")[1].strip()))
            elif line.startswith("Current game value:"):
                gv["cur"].append(float(line.split(":")[1].strip()))
            elif ":" in line and "[" in line and not line.startswith("Average") and not line.startswith("Current"):
                parts = line.split(":")
                key = parts[0].strip()
                nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+", parts[1])
                if len(nums) >= 2:
                    if key not in infoset_data:
                        infoset_data[key] = {"iters": [], "vals": []}
                    infoset_data[key]["iters"].append(cur_iter)
                    infoset_data[key]["vals"].append(float(nums[0]))

    return infoset_data, gv


def plot(filepath, game_name, algo, iters_str, out_dir):
    infoset_data, gv = parse_log(filepath)
    os.makedirs(out_dir, exist_ok=True)

    # ── per-infoset strategy plots (Kuhn only) ──
    if game_name == "kuhn":
        for key, d in infoset_data.items():
            plt.figure(figsize=(8, 5))
            plt.plot(d["iters"], d["vals"])
            plt.xlabel("Iterations"); plt.ylabel("Check/Pass prob")
            plt.title(f"Strategy – {key}"); plt.grid(True)
            plt.savefig(os.path.join(out_dir, f"{key}.png"), dpi=300, bbox_inches="tight")
            plt.close()
    else:
        if infoset_data:
            print(f"  [skip] {len(infoset_data)} infoset plots (too many for {game_name})")

    # ── game value plot (avg + current dual lines) ──
    if gv["iters"]:
        plt.figure(figsize=(10, 5))
        plt.plot(gv["iters"], gv["avg"], label="Average (cumulative)", color="steelblue", linewidth=1)

        # Align current-values to match iterations (possibly shorter)
        if gv["cur"]:
            cur_iters = gv["iters"][-len(gv["cur"]):] if len(gv["cur"]) <= len(gv["iters"]) else gv["iters"]
            cur_vals = gv["cur"]
            plt.plot(cur_iters[-len(cur_vals):], cur_vals,
                     label="Current (per snapshot)", color="darkorange", linewidth=1, alpha=0.8)

        nv, nl = _NASH.get(game_name, (None, None))
        if nv is not None:
            plt.axhline(y=nv, color="r", linestyle="--", linewidth=1, label=nl)
        plt.xlabel("Iterations"); plt.ylabel("Game Value (P0)")
        plt.title(f"Game Value – {game_name}_{algo}_{iters_str}")
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(out_dir, "game_value.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  saved: {out_dir}/game_value.png")
    else:
        print(f"  [warn] no game-value data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=os.path.join(_ROOT, "logs"))
    parser.add_argument("--output-dir", default=os.path.join(_ROOT, "eval", "visualizations"))
    parser.add_argument("log_pattern", nargs="?", default="*_strategy_*.txt",
                        help="glob relative to log-dir")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, args.log_pattern)))
    if not files:
        print(f"No logs matching {args.log_pattern} in {args.log_dir}")
        return

    for fp in files:
        bn = os.path.basename(fp)
        if "nfsp_dual_pair" in bn:
            continue
        game, algo, iters = None, None, None
        if "_strategy_" in bn:
            pref, suff = bn.split("_strategy_", 1)
            core = suff.removesuffix(".txt")
            m = re.match(r"^([a-zA-Z_]+)_(\de[+-]\d+)$", core)
            if m:
                game, algo, iters = pref, m.group(1), m.group(2)
        if algo is None:
            print(f"[skip] {bn}")
            continue
        out_sub = os.path.join(args.output_dir, f"{game}_{algo}_{iters}")
        print(f"[plot] {game}_{algo}_{iters}")
        plot(fp, game, algo, iters, out_sub)


if __name__ == "__main__":
    main()
