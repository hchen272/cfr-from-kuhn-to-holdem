"""Evaluation pipeline — auto-discovers models, runs visualization + exploitability.

Usage:
    python eval_pipeline.py                    # scan models/, run all
    python eval_pipeline.py --model leduc_cfr_plus_2e+06  # single model
"""
import os, sys, re, glob, subprocess, argparse

_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_ROOT, "src")
MODELS_DIR = os.path.join(_ROOT, "models")
LOGS_DIR = os.path.join(_ROOT, "logs")
EVAL_DIR = os.path.join(_ROOT, "eval")
VIS_DIR = os.path.join(EVAL_DIR, "visualizations")
EXP_DIR = os.path.join(EVAL_DIR, "exploitability")

# Which algorithms produce tabular models (check_exploit-compatible)
TABULAR_ALGOS = {"cfr", "cfr_plus", "dcfr", "pdcfr_plus"}

os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)


def discover_models():
    """Return list of (game, algo, iters_str, pkl_path)."""
    models = []
    for fp in sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl"))):
        bn = os.path.basename(fp).removesuffix(".pkl")
        # Format: {game}_{algo}_{iters}
        m = re.match(r"^([a-z]+)_([a-z_]+)_(\de[+-]\d+)$", bn)
        if m:
            models.append((m.group(1), m.group(2), m.group(3), fp))
    return models


def find_log(game, algo, iters_str):
    """Return log path or None."""
    expected = f"{game}_strategy_{algo}_{iters_str}.txt"
    fp = os.path.join(LOGS_DIR, expected)
    return fp if os.path.isfile(fp) else None


def run_visualization(log_path, game, algo, iters_str):
    """Run src/visualize.py for a single log."""
    out_dir = os.path.join(VIS_DIR, f"{game}_{algo}_{iters_str}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(SRC, "visualize.py"),
           "--log-dir", LOGS_DIR, "--output-dir", VIS_DIR,
           f"{game}_strategy_{algo}_{iters_str}.txt"]
    subprocess.run(cmd, cwd=_ROOT)


def run_exploitability(pkl_path, game, algo, iters_str):
    """Run src/check_exploit.py for a tabular model, save report."""
    model_name = f"{game}_{algo}_{iters_str}"
    out_txt = os.path.join(EXP_DIR, f"{model_name}.txt")
    cmd = [sys.executable, os.path.join(SRC, "check_exploit.py"),
           model_name, "--game", game]
    result = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n[STDERR]\n" + result.stderr)
    print(f"  exploit → {out_txt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Single model name like leduc_cfr_plus_2e+06")
    args = parser.parse_args()

    if args.model:
        m = re.match(r"^([a-z]+)_([a-z_]+)_(\de[+-]\d+)$", args.model)
        if not m:
            print(f"Bad model name: {args.model}")
            return
        models = [(m.group(1), m.group(2), m.group(3),
                   os.path.join(MODELS_DIR, args.model + ".pkl"))]
    else:
        models = discover_models()

    if not models:
        print("No models found.")
        return

    print(f"Found {len(models)} model(s)\n")

    for game, algo, iters_str, pkl_path in models:
        print(f"── {game}_{algo}_{iters_str} ──")
        log_path = find_log(game, algo, iters_str)

        # Visualization (from log)
        if log_path:
            run_visualization(log_path, game, algo, iters_str)
        else:
            print(f"  [skip viz] no log at {log_path}")

        # Exploitability (tabular only, from model)
        if algo in TABULAR_ALGOS and os.path.isfile(pkl_path):
            run_exploitability(pkl_path, game, algo, iters_str)
        else:
            print(f"  [skip exploit] not tabular or no .pkl")

        print()


if __name__ == "__main__":
    main()
