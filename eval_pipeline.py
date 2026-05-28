"""Evaluation pipeline — auto-discovers models, runs visualization + exploitability.

Output structure:
    eval/{game}_{algo}_{iters}/
        visualizations/   ← game_value.png (plus Kuhn per-infoset PNGs)
        exploitability/   ← {model_name}.txt

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

TABULAR_ALGOS = {"cfr", "cfr_plus", "dcfr", "pdcfr_plus", "deep_cfr", "deep_cfr_paper", "mccfr"}

# Known game names (longest first so expanded_leduc is tried before leduc)
_GAME_NAMES = sorted(["kuhn", "leduc", "expanded_leduc", "river_poker"], key=len, reverse=True)

_MODEL_RE = re.compile(r"^(.+)_(\de[+-]\d+)$")


def _parse_model_filename(filename: str):
    """Parse '{game}_{algo}_{iters}.pkl' using known game-name prefixes."""
    bn = filename.removesuffix(".pkl") if filename.endswith(".pkl") else filename
    for game in _GAME_NAMES:
        prefix = game + "_"
        if bn.startswith(prefix):
            rest = bn[len(prefix):]
            m = _MODEL_RE.match(rest)
            if m:
                algo, iters_str = m.group(1), m.group(2)
                return game, algo, iters_str
    return None


def discover_models():
    models = []
    for fp in sorted(glob.glob(os.path.join(MODELS_DIR, "*.pkl"))):
        bn = os.path.basename(fp)
        parsed = _parse_model_filename(bn)
        if parsed is None:
            continue
        game, algo, iters_str = parsed
        base_algo = algo.replace("_best", "")
        is_best = algo.endswith("_best")
        models.append((game, algo, base_algo, iters_str, fp, is_best))
    return models


def find_log(game, algo, iters_str):
    fp = os.path.join(LOGS_DIR, f"{game}_strategy_{algo}_{iters_str}.txt")
    return fp if os.path.isfile(fp) else None


def run_visualization(log_path, game, base_algo, iters_str, eval_name):
    out_dir = os.path.join(EVAL_DIR, eval_name, "visualizations")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(SRC, "visualize.py"),
                    "--log-dir", LOGS_DIR, "--output-dir", EVAL_DIR,
                    f"{game}_strategy_{base_algo}_{iters_str}.txt"],
                   cwd=_ROOT)


def run_exploitability(pkl_path, game, algo, iters_str):
    model_name = f"{game}_{algo}_{iters_str}"
    out_dir = os.path.join(EVAL_DIR, model_name, "exploitability")
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, f"{model_name}.txt")
    cmd = [sys.executable, os.path.join(SRC, "check_exploit.py"),
           model_name, "--game", game]
    result = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, timeout=300)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n" + result.stderr)
    print(f"  exploit → {out_txt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Single model name like leduc_cfr_plus_2e+06")
    args = parser.parse_args()

    if args.model:
        parsed = _parse_model_filename(args.model)
        if parsed is None:
            print(f"Bad model name: {args.model}")
            return
        game, algo, iters_str = parsed
        base_algo = algo.replace("_best", "")
        is_best = algo.endswith("_best")
        models = [(game, algo, base_algo, iters_str,
                   os.path.join(MODELS_DIR, args.model + ".pkl"), is_best)]
    else:
        models = discover_models()

    if not models:
        print("No models found.")
        return

    print(f"Found {len(models)} model(s)\n")

    for game, algo, base_algo, iters_str, pkl_path, is_best in models:
        name = f"{game}_{algo}_{iters_str}"
        tag = " [BEST]" if is_best else ""
        print(f"── {name}{tag} ──")

        log_path = find_log(game, base_algo, iters_str)
        if log_path:
            run_visualization(log_path, game, base_algo, iters_str, name)
        else:
            print(f"  [skip viz] no log")

        if base_algo in TABULAR_ALGOS and os.path.isfile(pkl_path):
            run_exploitability(pkl_path, game, algo, iters_str)
        else:
            print(f"  [skip exploit] not tabular or no .pkl")
        print()


if __name__ == "__main__":
    main()
