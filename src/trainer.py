import argparse
import random
from game_selector import get_game
from utils import save_strategy_txt, save_model

class Trainer:

    def __init__(self, algorithm='cfr', game_name='kuhn'):
        """
        algorithm: 'cfr', 'cfr_plus', 'dcfr', 'pdcfr_plus', or 'deep_cfr'
        game_name: 'kuhn' or 'leduc' (when implemented)
        """
        self.algorithm = algorithm
        self.game_name = game_name
        self.game = get_game(game_name)

    def train(self, iterations):
        # ---- Deep CFR uses a completely different training loop ----
        if self.algorithm == 'deep_cfr':
            from neural.train import train_deep_cfr
            # Deep CFR default: 1M (tabular defaults to 10M)
            dcfr_iters = 1000000 if iterations == 10000000 else iterations
            train_deep_cfr(iterations=dcfr_iters, log_prefix=self.algorithm,
                           game_name=self.game_name)
            return

        # ---- Tabular algorithms ------------------------------------
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
        save_strategy_txt(node_map, 0, 0, iterations, self.algorithm,
                          game_name=self.game_name)

        total_util = 0.0
        game = self.game

        for i in range(iterations):
            # Deal cards using the game
            cards = game.deal_cards()

            # Run one iteration of CFR/CFR+
            total_util += cfr(game, cards, "", 1, 1)

            # Periodic logging
            if (i + 1) % 100000 == 0:
                avg_value = round(total_util / (i + 1), 4)
                print(f"Iteration {i+1}")
                print(f"Average game value: {avg_value}")
                print("-" * 40)
                save_strategy_txt(node_map, i + 1, avg_value, iterations,
                                  self.algorithm, game_name=self.game_name)

        # Final output
        print("\n=== FINAL STRATEGIES ===\n")
        for infoset in sorted(node_map):
            node = node_map[infoset]
            avg_strategy = node.get_average_strategy()
            print(f"{infoset}: {avg_strategy}")

        save_model(node_map, iterations, self.algorithm,
                   game_name=self.game_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFR Trainer")
    parser.add_argument(
        "--algo", "-a",
        type=str,
        default="cfr",
        choices=["cfr", "cfr_plus", "dcfr", "pdcfr_plus", "deep_cfr"],
        help="Algorithm: 'cfr', 'cfr_plus', 'dcfr', 'pdcfr_plus', or 'deep_cfr'"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=10000000,
        help="Number of CFR iterations"
    )
    parser.add_argument(
        "--game", "-g",
        type=str,
        default="kuhn",
        choices=["kuhn", "leduc"],
        help="Game: 'kuhn' (default) or 'leduc' (when implemented)"
    )
    args = parser.parse_args()

    trainer = Trainer(algorithm=args.algo, game_name=args.game)
    trainer.train(args.iterations)