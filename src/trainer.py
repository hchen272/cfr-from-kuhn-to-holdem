import argparse
import random
import copy
from collections import deque
import numpy as np
from game_selector import get_game
from utils import save_strategy_txt, save_model

# ── Variance-based checkpoint threshold ──
# When nash_value is unknown (None), a rolling-window standard deviation
# below this threshold triggers a convergence checkpoint.
_CONVERGENCE_STD_THRESHOLD = 0.003


class Trainer:

    def __init__(self, algorithm='cfr', game_name='kuhn',
                 alternate=False):
        self.algorithm = algorithm
        self.game_name = game_name
        self.alternate = alternate
        self.game = get_game(game_name)

        # Build pre-computed game tree (integer-encoded, payoff-cached)
        from algo.tabular.game_tree import GameTree
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

        General formula for SUIT_COUNT = s:
            all-three-different: w = s³
            any-equality:        w = s²(s−1)

        E.g. Leduc (s=2): w=8 (all-diff), w=4 (any match); 120 ordered deals.
        Expanded Leduc (s=2, 4 ranks): w=8, w=4; 336 ordered deals.
        """
        game = self.game
        ranks = getattr(game, 'RANKS', ['J', 'Q', 'K'])
        s = getattr(game, 'SUIT_COUNT', 1)

        if not hasattr(game, '_community_rank'):
            # Kuhn-style: no community card
            deals = []
            for p0 in ranks:
                for p1 in ranks:
                    if p1 != p0:
                        deals.append(((p0, p1), '', 1))
            return deals

        # Leduc-style: community card enumeration
        deals = []
        # P0 ≠ P1
        for p0 in ranks:
            for p1 in ranks:
                if p1 == p0:
                    continue
                for comm in self.tree._comm_ranks:
                    if comm != p0 and comm != p1:
                        w = s ** 3
                    else:
                        w = s * s * (s - 1)
                    deals.append(((p0, p1), comm, w))
        # P0 ＝ P1
        for r in ranks:
            for comm in self.tree._comm_ranks:
                if comm == r:
                    continue
                w = s * s * (s - 1)
                deals.append(((r, r), comm, w))
        return deals

    def _train_mccfr(self, iterations):
        """External Sampling MCCFR — tabular regret/strategy storage."""
        from collections import deque
        from algo.mccfr.mccfr_tree import mccfr_tree

        game = self.game
        node_map = {}
        total_util = 0.0
        nash = getattr(game, 'nash_value', None)
        best_dist = float('inf')
        best_node_map = None
        best_iter = 0
        window = max(20, iterations // 50)
        recent_vals = deque(maxlen=window)

        print(f" game: {game.name}  |  algo: mccfr  |  iters: {iterations:,}")

        conv = self._converted_node_map(node_map)
        save_strategy_txt(conv, 0, 0, iterations, "mccfr", game_name=game.name)

        for t in range(1, iterations + 1):
            cards = game.deal_cards()
            traverser = t % 2
            p0r, p1r = cards

            # Enumerate community cards with correct probability
            s = getattr(game, 'SUIT_COUNT', 1)
            remain = {r: s for r in getattr(game, 'RANKS', [])}
            remain[p0r] -= 1
            remain[p1r] -= 1
            total_rem = sum(remain.values())

            iter_util = 0.0
            for cr, cnt in remain.items():
                if cnt > 0:
                    prob = cnt / total_rem
                    u = mccfr_tree(self.tree, cards, cr, 0, 1.0, 1.0,
                                   node_map, traverser, update_player=traverser)
                    iter_util += prob * u

            total_util += iter_util
            recent_vals.append(iter_util)

            if t % max(1, iterations // 100) == 0:
                avg_value = round(total_util / t, 6)
                cur_value = round(iter_util, 6)
                roll_avg = sum(recent_vals) / len(recent_vals)
                pct = t / iterations * 100
                print(f"[{pct:5.1f}%] Iter {t:>12,}  |  avg: {avg_value:+.6f}  "
                      f"cur: {cur_value:+.6f}  roll: {roll_avg:+.6f}")
                conv = self._converted_node_map(node_map)
                save_strategy_txt(conv, t, avg_value, iterations, "mccfr",
                                  game_name=game.name, iter_value=cur_value)

                if nash is not None:
                    d = abs(cur_value - nash)
                    if d < best_dist:
                        best_dist = d
                        best_iter = t
                        best_node_map = {k: copy.deepcopy(v) for k, v in conv.items()}
                        print(f"  >>> checkpoint (dist={d:.6f})")
                elif len(recent_vals) >= window // 2:
                    # Variance-based convergence checkpoint
                    roll_std = float(np.std(list(recent_vals)))
                    if roll_std < _CONVERGENCE_STD_THRESHOLD and roll_std < best_dist:
                        best_dist = roll_std
                        best_iter = t
                        best_node_map = {k: copy.deepcopy(v) for k, v in conv.items()}
                        print(f"  >>> checkpoint (std={roll_std:.6f}, roll_avg={roll_avg:+.6f})")

        print(f"\n=== FINAL STRATEGIES ===\n")
        conv = self._converted_node_map(node_map)
        for infoset in sorted(conv):
            print(f"{infoset}: {conv[infoset].get_average_strategy()}")
        print(f"Avg game value: {total_util / iterations:+.4f}")

        save_model(conv, iterations, "mccfr", game_name=game.name)
        if best_node_map is not None:
            save_model(best_node_map, best_iter, "mccfr_best", game_name=game.name)
            metric = "dist" if nash is not None else "std"
            print(f"Best checkpoint: iter {best_iter} ({metric}={best_dist:.6f})")

    def train(self, iterations, batch=False):
        # ---- Double DQN ---- (episode-based RL)
        if self.algorithm == 'ddqn':
            from algo.ddqn.train import train_ddqn
            train_ddqn(iterations=iterations, game_name=self.game_name,
                       log_prefix=self.algorithm)
            return

        # ---- DQN ---- (episode-based RL)
        if self.algorithm == 'dqn':
            from algo.dqn.train import train_dqn
            train_dqn(iterations=iterations, game_name=self.game_name,
                      log_prefix=self.algorithm)
            return

        # ---- NFSP Dual (bilateral) ---- (episode-based RL, both players learn)
        if self.algorithm == 'nfsp_dual':
            from algo.nfsp_dual.train import train_nfsp_dual
            train_nfsp_dual(iterations=iterations, game_name=self.game_name,
                            log_prefix=self.algorithm)
            return

        # ---- NFSP ---- (episode-based RL)
        if self.algorithm == 'nfsp':
            from algo.nfsp.train import train_nfsp
            train_nfsp(iterations=iterations, game_name=self.game_name,
                       log_prefix=self.algorithm)
            return

        # ---- Deep CFR (paper spec) —— external sampling + from-scratch ----
        if self.algorithm == 'deep_cfr_paper':
            from algo.deep_cfr.train import train
            train(iterations=iterations, game_name=self.game_name)
            return

        # ---- MCCFR (tabular, external sampling) ----
        if self.algorithm == 'mccfr':
            self._train_mccfr(iterations)
            return

        # ---- Deep CFR (original) uses a completely different training loop ----
        if self.algorithm == 'deep_cfr':
            from algo.neural.train import train_deep_cfr
            dcfr_iters = 1000000 if iterations == 10000000 else iterations
            train_deep_cfr(iterations=dcfr_iters, log_prefix=self.algorithm,
                           game_name=self.game_name,
                           alternate=self.alternate)
            return

        # ---- Tabular tree-based algorithms ----
        from algo.tabular.cfr_tree import (cfr_tree, cfr_plus_tree,
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

        # ── config summary ──────────────────────────────────────────
        print(f" game: {self.game.name}  |  algo: {self.algorithm}  "
              f"|  iters: {iterations:,}  |  alternate: {self.alternate}  "
              f"|  batch: {batch}")

        # Clean logs (write header)
        conv = self._converted_node_map(node_map)
        save_strategy_txt(conv, 0, 0, iterations, self.algorithm,
                          game_name=self.game_name)

        total_util = 0.0
        game = self.game
        root_hid = 0
        nash = getattr(game, 'nash_value', None)
        best_distance = float('inf')
        best_iter = 0
        best_node_map = None
        window = max(20, iterations // 50)
        recent_vals = deque(maxlen=window)

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
                    if comm_rank:
                        game._comm = (comm_rank, 0)
                    dw = rest[0] if rest else 1.0
                    if is_weighted:
                        iter_util += dw * cfr_fn(
                            self.tree, cards, comm_rank,
                            root_hid, 1.0, 1.0, node_map,
                            deal_weight=dw, **extra)
                    else:
                        iter_util += dw * cfr_fn(
                            self.tree, cards, comm_rank,
                            root_hid, dw, dw, node_map, **extra)

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
                    iter_util /= total_weight
            else:
                cards = game.deal_cards()
                if self.tree._comm_ranks:
                    # ── enumerate all community cards with correct probability ──
                    p0r, p1r = cards
                    s = getattr(game, 'SUIT_COUNT', 1)
                    remain = {r: s for r in getattr(game, 'RANKS', [])}
                    remain[p0r] -= 1
                    remain[p1r] -= 1
                    total_rem = sum(remain.values())
                    iter_util = 0.0
                    for cr in self.tree._comm_ranks:
                        if remain[cr] == 0:
                            continue
                        prob = remain[cr] / total_rem
                        game._comm = (cr, 0)
                        iter_util += prob * cfr_fn(
                            self.tree, cards, cr,
                            root_hid, prob, prob, node_map, **extra)
                else:
                    iter_util = cfr_fn(self.tree, cards, "",
                                       root_hid, 1.0, 1.0, node_map, **extra)

            total_util += iter_util
            recent_vals.append(iter_util)

            if (i + 1) % max(1, iterations // 100) == 0:
                avg_value = round(total_util / (i + 1), 6)
                cur_value = round(iter_util, 6)
                roll_avg = sum(recent_vals) / len(recent_vals) if recent_vals else 0.0
                pct = (i + 1) / iterations * 100
                print(f"[{pct:5.1f}%] Iter {i+1:>12,}  |  avg: {avg_value:+.6f}  cur: {cur_value:+.6f}  roll: {roll_avg:+.6f}")
                conv = self._converted_node_map(node_map)
                save_strategy_txt(conv, i + 1, avg_value, iterations,
                                  self.algorithm, game_name=self.game_name,
                                  iter_value=cur_value)
                # ── checkpoint ──
                if nash is not None:
                    d = abs(cur_value - nash)
                    if d < best_distance:
                        best_distance = d
                        best_iter = i + 1
                        best_node_map = {k: copy.deepcopy(v) for k, v in conv.items()}
                        print(f"  >>> checkpoint (dist={d:.6f})")
                elif len(recent_vals) >= window // 2:
                    # Variance-based convergence checkpoint
                    roll_std = float(np.std(list(recent_vals)))
                    if roll_std < _CONVERGENCE_STD_THRESHOLD and roll_std < best_distance:
                        best_distance = roll_std
                        best_iter = i + 1
                        best_node_map = {k: copy.deepcopy(v) for k, v in conv.items()}
                        print(f"  >>> checkpoint (std={roll_std:.6f}, roll_avg={roll_avg:+.6f})")

        # Final output
        print("\n=== FINAL STRATEGIES ===\n")
        conv = self._converted_node_map(node_map)
        for infoset in sorted(conv):
            node = conv[infoset]
            avg_strategy = node.get_average_strategy()
            print(f"{infoset}: {avg_strategy}")

        save_model(conv, iterations, self.algorithm,
                   game_name=self.game_name)

        if best_node_map is not None:
            save_model(best_node_map, best_iter, self.algorithm + "_best",
                       game_name=self.game_name)
            metric = "dist" if nash is not None else "std"
            print(f"Best checkpoint: iter {best_iter} ({metric}={best_distance:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFR Trainer")
    parser.add_argument(
        "--algo", "-a",
        type=str, default="cfr",
        choices=["cfr", "cfr_plus", "dcfr", "pdcfr_plus", "deep_cfr",
                 "deep_cfr_paper", "dqn", "ddqn", "nfsp", "nfsp_dual", "mccfr"],
        help="Algorithm: cfr, cfr_plus, dcfr, pdcfr_plus, deep_cfr, deep_cfr_paper, dqn, ddqn, nfsp, nfsp_dual, mccfr"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int, default=10000000,
        help="Number of CFR iterations"
    )
    parser.add_argument(
        "--game", "-g",
        type=str, default="kuhn",
        choices=["kuhn", "leduc", "expanded_leduc"],
        help="Game: 'kuhn', 'leduc', or 'expanded_leduc'"
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
