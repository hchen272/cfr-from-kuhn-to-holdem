"""
Exploitability analysis for Leduc / Kuhn.
Loads a trained CFR model and computes Best-Response exploitability.

Usage:
    python src/validate_env.py leduc_cfr_plus_5e+06
    python src/validate_env.py kuhn_cfr_plus_5e+04  --game kuhn
"""
import sys, os, pickle, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src') if '__file__' in dir() else 'src')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Need to set src as first path
script_dir = os.path.dirname(__file__)
if script_dir:
    sys.path.insert(0, os.path.join(script_dir, '..'))

# ──────────────────────────────────────────────────────────────────────
#  1. Load strategy
# ──────────────────────────────────────────────────────────────────────

def load_model(model_name, models_dir='models'):
    """Load a pickled strategy_sum dict."""
    path = os.path.join(models_dir, f"{model_name}.pkl")
    if not os.path.exists(path):
        # try with game prefix
        for f in os.listdir(models_dir):
            if model_name in f and f.endswith('.pkl'):
                path = os.path.join(models_dir, f)
                break
    print(f"Loading {path} ...")
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    if isinstance(data, dict):
        return data
    # Some saves wrap in another structure
    return data

# ──────────────────────────────────────────────────────────────────────
#  2. Best-Response computation
# ──────────────────────────────────────────────────────────────────────

def compute_br(avg_strategies, tree, game, br_player):
    """Compute the best-response value for *br_player* against the
    opponent's average strategy stored in *avg_strategies*.

    avg_strategies : dict  iid -> np.array (average strategy probs)

    Returns the game value FROM THE PERSPECTIVE OF br_player.
    """
    total = 0.0
    n_deals = 0

    # Enumerate ALL possible card deals for exact expectation
    deck = game.deck if hasattr(game, 'deck') else None
    all_cards = game.all_cards if hasattr(game, 'all_cards') else None

    from itertools import permutations

    if game.name == 'kuhn':
        cards_list = ['J','Q','K']
        for p0_card in cards_list:
            for p1_card in cards_list:
                if p1_card == p0_card:
                    continue
                cards = (p0_card, p1_card)
                val = _cfr_br_value(tree, cards, 0, br_player, avg_strategies)
                total += val
                n_deals += 1
        # average over all 6 deals
        return total / n_deals

    elif game.name == 'leduc':
        # Leduc: 6 cards, deal 2 to players + 1 community
        leduc_cards = ['J','J','Q','Q','K','K']
        from itertools import permutations
        seen = set()
        for perm in permutations(leduc_cards, 3):
            deal = (perm[0], perm[1], perm[2])
            if deal in seen:
                continue
            seen.add(deal)
            cards = (perm[0], perm[1])
            comm = perm[2]
            game._comm = (comm, 0)
            val = _cfr_br_value(tree, cards, 0, br_player, avg_strategies, comm_rank=comm)
            total += val
            n_deals += 1
        return total / n_deals

    else:
        raise ValueError(f"Unknown game: {game.name}")


def _cfr_br_value(tree, cards, hid, br_player, avg_strategies,
                  comm_rank=""):
    """Recursive best-response value."""
    node_info = tree.nodes[hid]
    player = node_info.player

    if node_info.is_terminal:
        pay = tree.get_payoff(hid, cards)
        return pay if player == 0 else -pay

    if player == br_player:
        # BR: choose the action that maximises value
        best_val = -float('inf')
        for a in node_info.legal_actions:
            child_hid = node_info.child_for(a, comm_rank)
            if child_hid is None:
                continue
            if player == 0:
                v = -_cfr_br_value(tree, cards, child_hid, br_player,
                                   avg_strategies, comm_rank)
            else:
                v = -_cfr_br_value(tree, cards, child_hid, br_player,
                                   avg_strategies, comm_rank)
            if v > best_val:
                best_val = v
        return best_val
    else:
        # Opponent: follow average strategy
        iid = tree.infoset_id(cards[player], hid)
        if iid in avg_strategies:
            strat = avg_strategies[iid]
        else:
            strat = np.ones(tree.num_actions) / tree.num_actions
        val = 0.0
        for a in node_info.legal_actions:
            child_hid = node_info.child_for(a, comm_rank)
            if child_hid is None:
                continue
            if player == 0:
                v = -_cfr_br_value(tree, cards, child_hid, br_player,
                                   avg_strategies, comm_rank)
            else:
                v = -_cfr_br_value(tree, cards, child_hid, br_player,
                                   avg_strategies, comm_rank)
            val += strat[a] * v
        return val


# ──────────────────────────────────────────────────────────────────────
#  3. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", nargs='?', default="leduc_cfr_plus_5e+06")
    parser.add_argument("--game", "-g", default="leduc", choices=["kuhn", "leduc"])
    args = parser.parse_args()

    from game_selector import get_game
    from tabular.game_tree import GameTree

    game = get_game(args.game)
    tree = GameTree(game)
    print(f"Game: {game.name}, tree nodes: {len(tree.nodes)}")

    # Load model
    strategy_data = load_model(args.model_name)
    
    # The model stores strategy_sum dict; extract average strategies
    avg_strategies = {}
    if isinstance(strategy_data, dict):
        # strategy_sum: keys are infoset strings, values are arrays
        # We need to convert to IID -> np.array average strategy
        for key, val in strategy_data.items():
            # key could be string infoset or integer IID
            if isinstance(key, int):
                avg = val
                # Normalize if it's a sum
                total = np.sum(val)
                if total > 0:
                    avg = val / total
                else:
                    avg = np.ones_like(val) / len(val)
                avg_strategies[key] = avg
            else:
                # string key: convert to IID
                if hasattr(tree, 'infoset_id'):
                    iid = tree.infoset_id(key[0], 0)  # approximate
                    avg_strategies[iid] = val
    
    # If keys are strings, try to convert using infoset_str
    if not avg_strategies:
        # Try as Node-like dict
        for infoset_key, node in strategy_data.items():
            if hasattr(node, 'get_average_strategy'):
                s = node.get_average_strategy()
                avg_strategies[infoset_key] = s

    print(f"Loaded {len(avg_strategies)} infosets")
    
    # Compute BR for both players
    print("\n--- Computing exploitability ---")
    v0 = compute_br(avg_strategies, tree, game, br_player=0)
    v1 = compute_br(avg_strategies, tree, game, br_player=1)
    
    print(f"P0 game value vs avg:  {v0:+.6f}")
    print(f"P1 game value vs avg:  {v1:+.6f}")
    print(f"Exploitability:         {v0 - v1:+.6f}  (should be ~0 at Nash)")
    print(f"Game value (P0 persp):  {(v0 - v1)/2:+.6f}")
    
    # Full Nash exploitability check
    br0_vs_br1 = compute_br(avg_strategies, tree, game, br_player=0)
    # Recompute so it's BR0 against sigma1
    # Actually v0 is already P0's BR value against avg

    print(f"\n=== Nash Distance ===")
    nash = getattr(game, 'nash_value', None)
    if nash is not None:
        print(f"Nash value: {nash}")
    print(f"Exploitability = max(0, {v0:.4f} - {(-v1):.4f}) / 2 = {(v0 - (-v1))/2:.4f}")


if __name__ == '__main__':
    main()
