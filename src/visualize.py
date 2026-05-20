import os
import re
import glob

import matplotlib.pyplot as plt

# Script lives in src/; project root is one level up
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
VIS_DIR = os.path.join(_PROJECT_ROOT, "visualizations")


def _parse_log_filename(filepath):
    """
    Extract algorithm name and iteration string from a log filename.

    Supports two naming conventions:

        logs/strategy_{algo}_{iters}.txt    (new, e.g. strategy_dcfr_1e+07.txt)
        logs/strategy_{iters}.txt           (old, e.g. strategy_1e+10.txt)

    Returns:
        tuple (algo, iters_str) or (None, None) if unparsable.
    """
    basename = os.path.basename(filepath)                 # strategy_dcfr_1e+07.txt
    core = basename.removeprefix("strategy_").removesuffix(".txt")  # dcfr_1e+07

    # New format:  {algo}_{iters}   e.g. cfr_1e+07, cfr_plus_1e+07
    m = re.match(r"^([a-zA-Z_]+)_(\de[+-]\d+)$", core)
    if m:
        return m.group(1), m.group(2)

    # Old format:  {iters} only    e.g. 1e+10
    m = re.match(r"^(\de[+-]\d+)$", core)
    if m:
        return "legacy", m.group(1)

    return None, None


def _folder_path(algo, iters_str):
    """Return expected visualization folder path under VIS_DIR."""
    return os.path.join(VIS_DIR, f"{algo}_{iters_str}")


def _read_log(filepath):
    """
    Parse a strategy log file into structured data.

    Returns:
        dict: {infoset_name: {"iters": [int, ...], "values": [float, ...]}}
    """
    history = {}
    current_iter = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Iterations:"):
                current_iter = int(line.split(":")[1].strip())

            # Parse strategy lines like "J: [0.73 0.27]", skip "Average ..."
            elif ":" in line and "[" in line and not line.startswith("Average"):
                parts = line.split(":")
                infoset = parts[0].strip()
                numbers = re.findall(
                    r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+",
                    parts[1],
                )
                if len(numbers) >= 2:
                    check_prob = float(numbers[0])  # probability of first action
                    if infoset not in history:
                        history[infoset] = {"iters": [], "values": []}
                    history[infoset]["iters"].append(current_iter)
                    history[infoset]["values"].append(check_prob)

    return history


def _read_game_values(filepath):
    """
    Extract the (iteration, average_game_value) progression from a log file.

    Returns:
        dict: {"iters": [int, ...], "values": [float, ...]}
    """
    iters = []
    values = []
    current_iter = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Iterations:"):
                current_iter = int(line.split(":")[1].strip())
            elif line.startswith("Average game value:"):
                val = float(line.split(":")[1].strip())
                if current_iter is not None:
                    iters.append(current_iter)
                    values.append(val)

    return {"iters": iters, "values": values}


def plot_strategy_evolution(filepath, algo, iters_str):
    """
    Plot strategy evolution for every infoset found in *filepath*,
    plus the overall average game value progression.

    Saves one PNG per infoset plus ``game_value.png`` under::

        visualizations/{algo}_{iters_str}/
    """
    history = _read_log(filepath)
    game_values = _read_game_values(filepath)

    output_dir = _folder_path(algo, iters_str)
    os.makedirs(output_dir, exist_ok=True)

    # ---- per-infoset strategy plots ---------------------------------
    for infoset, data in history.items():
        plt.figure(figsize=(8, 5))
        plt.plot(data["iters"], data["values"])
        plt.xlabel("Iterations")
        plt.ylabel("Check / Pass Probability")
        plt.title(f"Strategy Evolution – {infoset}")
        plt.grid(True)

        out_path = os.path.join(output_dir, f"{infoset}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    # ---- average game value plot ------------------------------------
    if game_values["iters"]:
        plt.figure(figsize=(8, 5))
        plt.plot(game_values["iters"], game_values["values"],
                 label="Average game value")
        plt.axhline(y=-1/18, color="r", linestyle="--", linewidth=1,
                    label=f"Nash equilibrium (-1/18 ≈ -0.0556)")
        plt.xlabel("Iterations")
        plt.ylabel("Average Game Value (player 0)")
        plt.title(f"Average Game Value – {algo}_{iters_str}")
        plt.legend()
        plt.grid(True)

        out_path = os.path.join(output_dir, "game_value.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")
    else:
        print(f"  [WARN] No game-value data found in {filepath}")


if __name__ == "__main__":
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "strategy_*.txt")))

    if not log_files:
        print(f"No log files found in {LOG_DIR}/")
        exit(0)

    for log_file in log_files:
        algo, iters_str = _parse_log_filename(log_file)

        if algo is None:
            print(f"[SKIP]  Unrecognised filename: {os.path.basename(log_file)}")
            continue

        print(f"[WORK]  {algo}_{iters_str}  ← {os.path.basename(log_file)}")
        plot_strategy_evolution(log_file, algo, iters_str)

    print("\nDone.")
