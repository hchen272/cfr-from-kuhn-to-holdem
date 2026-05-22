import argparse
import random
import numpy as np
from game_selector import get_game
from utils import save_strategy_txt, save_model


class Trainer:

    def __init__(self, algorithm='cfr', game_name='kuhn',
                 alternate=False):
        self.algorithm = algorithm
        self.game_name = game_name
        self.alternate = alternate
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

    def _batch_deals(self):
        """Return list of (cards, comm_rank, weight) covering all instances.

        Weights are proportional to the number of card-instance deals sharing
        the same rank pattern, so that regret deltas can be averaged in an
        instance-weighted sense.

        Leduc (6-card deck: J₁J₂ Q₁Q₂ K₁K₂ → 120 ordered deals):

            P0≠P1, comm≠P0≠P1  (e.g. J-Q-K) : 6 patterns × 8 = 48  → w=8
            P0≠P1, comm=P0     (e.g. J-Q-J) : 6 patterns × 4 = 24  → w=4
            P0≠P1, comm=P1     (e.g. J-Q-Q) : 6 patterns × 4 = 24  → w=4
            P0=P1              (e.g. J-J-Q) : 6 patterns × 4 = 24  → w=4
                                          total = 120
        """
        game = self.game
        if game.name == 'kuhn':
            ranks = ['J', 'Q', 'K']
            deals = []
            for p0 in ranks:
                for p1 in ranks:
                    if p1 != p0:
                        deals.append(((p0, p1), '', 1))
            return deals
        elif game.name == 'leduc':
            ranks = ['J', 'Q', 'K']
            deals = []
            # P0 ≠ P1 — 18 rank combos
            for p0 in ranks:
                for p1 in ranks:
                    if p1 == p0:
                        continue
                    for comm in self.tree._comm_ranks:
                        if comm != p0 and comm != p1:
                            w = 8  # all-different: 2×2×2 = 8 instances
                        else:
                            w = 4  # comm matches P0 or P1: 2×2×1 = 4
                        deals.append(((p0, p1), comm, w))
            # P0 ＝ P1 — 6 rank combos (previously missing)
            for r in ranks:
                for comm in self.tree._comm_ranks:
                    if comm == r:
                        continue
                    # 2 ways to assign suits × 4 remaining comm cards = 8,
                    # but split across the 2 comm ranks → 4 per deal
                    deals.append(((r, r), comm, 4))
            return deals
        else:
            raise ValueError(f"Unknown game: {game.name}")

    def train(self, iterations, batch=False):
        # ---- Deep CFR uses a completely different training loop ----
        if self.algorithm == 'deep_cfr':
            from neural.train import train_deep_cfr
            dcfr_iters = 1000000 if iterations == 10000000 else iterations
            train_deep_cfr(iterations=dcfr_iters, log_prefix=self.algorithm,
                           game_name=self.game_name,
                           alternate=self.alternate)
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
            extra = {'iter_cnt_ref': [0]}
        elif self.algorithm == 'dcfr':
            cfr_fn = dcfr_tree
            extra = {'iter_cnt_ref': [0], 'alpha': 1.5, 'beta': 0.0, 'gamma': 2.0}
        elif self.algorithm == 'pdcfr_plus':
            cfr_fn = pdcfr_plus_tree
            extra = {'iter_cnt_ref': [0], 'alpha': 1.5, 'beta': 0.0, 'gamma': 2.0}
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Pre-compute batch deals list if needed
        if batch:
            batch_deals = self._batch_deals()
            n_batch = len(batch_deals)
            total_weight = sum(w for _, _, w in batch_deals)
            is_weighted = self.algorithm == 'cfr_plus'  # C2 enabled for CFR+
            print(f"Batch mode: {n_batch} rank combos ({total_weight} instances)"
                  f" per iteration ({self.game.name})"
                  f"{' [C2 weighted]' if is_weighted else ''}")
        else:
            batch_deals = None
            n_batch = 1
            total_weight = 1
            is_weighted = False

        # Clean logs (write header)
        conv = self._converted_node_map(node_map)
        save_strategy_txt(conv, 0, 0, iterations, self.algorithm,
                          game_name=self.game_name)

        total_util = 0.0
        game = self.game
        root_hid = 0  # empty history always gets ID 0

        for i in range(iterations):
            # Alternating updates: swap every iteration (paper's standard)
            if self.alternate:
                extra['update_player'] = i % 2  # 0 → P0, 1 → P1
            else:
                extra['update_player'] = -1  # both

            # Increment iter counter for DCFR/PDCFR+ (once per outer iter)
            if 'iter_cnt_ref' in extra:
                extra['iter_cnt_ref'][0] += 1

            iter_util = 0.0

            if batch:
                # ── C2 batch traversal: accumulate, don't apply ──
                if is_weighted:
                    # Reset accumulators
                    for node in node_map.values():
                        node.reset_batch()

                for cards, comm_rank, *rest in batch_deals:
                    game._comm = (comm_rank, 0) if comm_rank else game._comm
                    dw = rest[0] if rest else 1.0
                    if is_weighted:
                        iter_util += dw * cfr_fn(
                            self.tree, cards, comm_rank,
                            root_hid, 1.0, 1.0, node_map,
                            deal_weight=dw, **extra)
                    else:
                        iter_util += cfr_fn(
                            self.tree, cards, comm_rank,
                            root_hid, 1.0, 1.0, node_map, **extra)

                if is_weighted:
                    # ── C2 post-batch: accumulate σ_t (before regret update),
                    #     then apply averaged regret deltas ──
                    lw = float(extra.get('iter_cnt_ref', [1])[0])
                    for node in node_map.values():
                        if node._batch_weight > 0:
                            avg_reach = (node._batch_reach
                                         / node._batch_weight)
                            # accumulate σ_t first (strategy from *current* regrets)
                            node.get_strategy(avg_reach,
                                              linear_weight=lw,
                                              accumulate=True)
                            # then update regrets → σ_{t+1}
                            avg_delta = node._batch_delta / node._batch_weight
                            node.regret_sum = np.maximum(
                                0.0, node.regret_sum + avg_delta)
                    iter_util /= total_weight
                else:
                    iter_util /= n_batch
            else:
                cards = game.deal_cards()
                comm_rank = ""
                if self.tree._comm_ranks:
                    comm = getattr(game, '_comm', None)
                    if comm is not None:
                        comm_rank = comm[0] if isinstance(comm, tuple) else comm
                iter_util = cfr_fn(self.tree, cards, comm_rank,
                                   root_hid, 1.0, 1.0, node_map, **extra)

            total_util += iter_util

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
    parser.add_argument(
        "--alternate",
        action="store_true",
        help="Enable alternating updates (P0 / P1 every 50k iters)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enumerate all possible (cards, comm) combos each iteration"
    )
    args = parser.parse_args()

    trainer = Trainer(algorithm=args.algo, game_name=args.game,
                      alternate=args.alternate)
    trainer.train(args.iterations, batch=args.batch)
