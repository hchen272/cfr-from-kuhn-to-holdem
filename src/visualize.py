import os
import re

import matplotlib.pyplot as plt


def plot_strategy_evolution(filepath, algo, iters=None):
    """
    Plot each infoset strategy separately.

    Save all figures into:
    visualizations/
    """

    strategy_history = {}

    current_iter = None

    # read log file
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # parse iteration
            if line.startswith("Iterations:"):

                current_iter = int(
                    line.split(":")[1]
                    .strip()
                )

            # parse strategy
            elif ":" in line and "[" in line:
                parts = line.split(":")
                infoset = parts[0].strip()
                strategy_text = parts[1]
                numbers = re.findall(
                    r"[-+]?\d*\.\d+|\d+",
                    strategy_text
                )

                if len(numbers) >= 2:
                    # first action probability
                    check_prob = float(numbers[0])

                    if infoset not in strategy_history:
                        strategy_history[infoset] = {
                            "iters": [],
                            "values": []
                        }
                    strategy_history[infoset]["iters"].append(
                        current_iter
                    )
                    strategy_history[infoset]["values"].append(
                        check_prob
                    )

    # create output folder
    output_dir = f"visualizations/{algo}_{iters:.0e}"
    os.makedirs(output_dir, exist_ok=True)

    # create one figure per infoset
    for infoset, data in strategy_history.items():

        plt.figure(figsize=(8, 5))
        plt.plot(
            data["iters"],
            data["values"]
        )
        plt.xlabel("Iterations")
        plt.ylabel("Check / Pass Probability")
        plt.title(
            f"Strategy Evolution - {infoset}"
        )
        plt.grid(True)
        # save figure
        filepath = os.path.join(
            output_dir,
            f"{infoset}.png"
        )
        plt.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print(
            f"Saved visualization: {filepath}"
        )

if __name__ == "__main__":
    filepath = "logs/strategy_cfr_1e+07.txt"
    algo = "cfr"

    plot_strategy_evolution(
        filepath,
        algo,
        iters=10000000000
    )