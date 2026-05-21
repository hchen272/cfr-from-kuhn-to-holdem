import argparse
import random
from game_selector import get_game
from utils import save_strategy_txt, save_model


class Trainer:

    def __init__(self, algorithm='cfr', game_name='kuhn'):
        self.algorithm = algorithm
        self.game_name = game_name
        self.game = get_game(game_name)

        # Build pre-computed game tree (integer-encoded, payoff-cached)
        from tabular.game_tree import GameTree
        self.tree = GameTree(self.game)

    def _converted_node_map(self, raw_map):
        """Convert integer-keyed node_map → string-keyed for logging."""
        out = {}
        for iid, node in raw_map.items():
            key = self.tree.infoset_str(iid)
            out[key] = node
        return out

    def train(self, iterations):
        # ---- Deep CFR uses a completely different training loop ----
        if self.algorithm == 'deep_cfr':
            from neural.train import train_deep_cfr
            dcfr_iters = 1000000 if iterations == 10000000 else iterations
            train_deep_cfr(iterations=dcfr_iters, log_prefix=self.algorithm,
                           game_name=self.game_name)
            return

        # ---- Tabular tree-based algorithms ----
        from tabular.cfr_tree import (cfr_tree, cfr_plus_tree,
                                      dcfr_tree, pdcfr_plus_tree)

        node_map = {}  # local, keyed by integer infoset ID

        # Select algorithm function
        if self.algorithm == 'cfr':
            cfr_fn = cfr_tree
            extra = {}
        elif self.algorithm == 'cfr_plus':
            cfr_fn = cfr_plus_tree
            extra = {}
        elif self.algorithm == 'dcfr':
            cfr_fn = dcfr_tree
            extra = {'iter_cnt_ref': [0], 'alpha': 1.5, 'beta': 0.0, 'gamma': 2.0}
        elif self.algorithm == 'pdcfr_plus':
            cfr_fn = pdcfr_plus_tree
            extra = {'iter_cnt_ref': [0], 'alpha': 1.5, 'beta': 0.0, 'gamma': 2.0}
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Clean logs (write header)
        conv = self._converted_node_map(node_map)
        save_strategy_txt(conv, 0, 0, iterations, self.algorithm,
                          game_name=self.game_name)

        total_util = 0.0
        game = self.game
        root_hid = 0  # empty history always gets ID 0

        for i in range(iterations):
            cards = game.deal_cards()
            comm_rank = ""
            if self.tree._comm_ranks:
                comm = getattr(game, '_comm', None)
                if comm is not None:
                    comm_rank = comm[0] if isinstance(comm, tuple) else comm

            # Increment iter counter for DCFR/PDCFR+
            if 'iter_cnt_ref' in extra:
                extra['iter_cnt_ref'][0] += 1

            total_util += cfr_fn(self.tree, cards, comm_rank,
                                 root_hid, 1.0, 1.0, node_map, **extra)

            if (i + 1) % 100000 == 0:
                avg_value = round(total_util / (i + 1), 4)
                print(f"Iteration {i+1}")
                print(f"Average game value: {avg_value}")
                print("-" * 40)
                conv = self._converted_node_map(node_map)
                save_strategy_txt(conv, i + 1, avg_value, iterations,
                                  self.algorithm, game_name=self.game_name)

        # Final output
        print("\n=== FINAL STRATEGIES ===\n")
        conv = self._converted_node_map(node_map)
        for infoset in sorted(conv):
            node = conv[infoset]
            avg_strategy = node.get_average_strategy()
            print(f"{infoset}: {avg_strategy}")

        save_model(conv, iterations, self.algorithm,
                   game_name=self.game_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFR Trainer")
    parser.add_argument(
        "--algo", "-a",
        type=str, default="cfr",
        choices=["cfr", "cfr_plus", "dcfr", "pdcfr_plus", "deep_cfr"],
        help="Algorithm: 'cfr', 'cfr_plus', 'dcfr', 'pdcfr_plus', or 'deep_cfr'"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int, default=10000000,
        help="Number of CFR iterations"
    )
    parser.add_argument(
        "--game", "-g",
        type=str, default="kuhn",
        choices=["kuhn", "leduc"],
        help="Game: 'kuhn' (default) or 'leduc' (when implemented)"
    )
    args = parser.parse_args()

    trainer = Trainer(algorithm=args.algo, game_name=args.game)
    trainer.train(args.iterations)
