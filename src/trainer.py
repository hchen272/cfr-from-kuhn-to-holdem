import argparse
import random
from game import *
from utils import save_strategy_txt, save_model

class Trainer:

    def __init__(self, algorithm='cfr'):
        """
        algorithm: 'cfr' or 'cfr_plus'
        """
        self.algorithm = algorithm

    def train(self, iterations):
        # Select the correct CFR variant and node map
        if self.algorithm == 'cfr':
            from cfr import cfr, node_map
        elif self.algorithm == 'cfr_plus':
            from cfr_plus import cfr_plus as cfr, node_map_plus as node_map
        elif self.algorithm == 'dcfr':
            from dcfr import dcfr as cfr, node_map_dcfr as node_map
        elif self.algorithm == 'pdcfr_plus':
            from pdcfr_plus import pdcfr_plus as cfr, node_map_pdcfr as node_map
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Clean logs (write header)
        save_strategy_txt(node_map, 0, 0, iterations, self.algorithm)

        total_util = 0.0

        for i in range(iterations):
            # Shuffle and deal cards
            cards = CARDS.copy()
            random.shuffle(cards)
            cards = (cards[0], cards[1])

            # Run one iteration of CFR/CFR+
            total_util += cfr(cards, "", 1, 1)

            # Periodic logging
            if (i + 1) % 100000 == 0:
                avg_value = round(total_util / (i + 1), 4)
                print(f"Iteration {i+1}")
                print(f"Average game value: {avg_value}")
                print("-" * 40)
                save_strategy_txt(node_map, i + 1, avg_value, iterations, self.algorithm)

        # Final output
        print("\n=== FINAL STRATEGIES ===\n")
        for infoset in sorted(node_map):
            node = node_map[infoset]
            avg_strategy = node.get_average_strategy()
            print(f"{infoset}: {avg_strategy}")

        save_model(node_map, iterations, self.algorithm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kuhn Poker CFR Trainer")
    parser.add_argument(
        "--algo", "-a",
        type=str,
        default="cfr",
        choices=["cfr", "cfr_plus", "dcfr", "pdcfr_plus"],
        help="Algorithm: 'cfr', 'cfr_plus', 'dcfr', or 'pdcfr_plus'"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10000000,
        help="Number of CFR iterations"
    )
    args = parser.parse_args()

    trainer = Trainer(algorithm=args.algo)
    trainer.train(args.iterations)