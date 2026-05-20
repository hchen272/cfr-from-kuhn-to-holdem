import pickle
import os


def save_model(node_map, iterations, algorithm):
    """
    Save trained node map.
    """

    model_dir = "models"

    os.makedirs(model_dir, exist_ok=True)

    filename = f"kuhn_{algorithm}_{iterations:.0e}.pkl"

    filepath = os.path.join(model_dir, filename)

    with open(filepath, "wb") as f:
        pickle.dump(node_map, f)

    print(f"Model saved to {filepath}")

def load_model(iterations, algorithm):

    filename = f"kuhn_{algorithm}_{iterations:.0e}.pkl"

    filepath = os.path.join("models", filename)

    with open(filepath, "rb") as f:
        node_map = pickle.load(f)

    print(f"Model loaded from {filepath}")

    return node_map


def save_strategy_txt(
    node_map,
    iter_now,
    avg_value,
    iterations,
    algorithm
):
    """
    Save strategy snapshots.
    """

    log_dir = "logs"

    os.makedirs(log_dir, exist_ok=True)

    filename = (
        f"strategy_{algorithm}_{iterations:.0e}.txt"
    )

    filepath = os.path.join(
        log_dir,
        filename
    )

    # overwrite at beginning
    # append afterwards
    mode = "w" if iter_now == 0 else "a"

    with open(
        filepath,
        mode,
        encoding="utf-8"
    ) as f:

        # write header once
        if iter_now == 0:

            f.write(
                "=== CFR STRATEGY SNAPSHOT ===\n\n"
            )

            return

        # snapshot section
        f.write("=" * 50 + "\n\n")

        f.write(
            f"Iterations: {iter_now}\n"
        )

        for infoset in sorted(node_map):

            avg_strategy = (
                node_map[infoset]
                .get_average_strategy()
            )

            f.write(
                f"{infoset}: "
                f"{avg_strategy}\n"
            )

        f.write(
            f"Average game value: "
            f"{avg_value:.4f}\n\n"
        )