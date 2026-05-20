"""
Backward-compatibility re-export of Kuhn Poker game constants and functions.

New code should use ``game_selector.get_game("kuhn")`` instead.
"""
from games.kuhn import KuhnGame

_game = KuhnGame()

# Re-export module-level constants for backward compatibility
CARDS = _game.CARDS
PASS = _game.PASS
BET = _game.BET
ACTIONS = _game.ACTIONS
NUM_ACTIONS = _game.num_actions

# Re-export functions
is_terminal = _game.is_terminal
get_payoff = _game.get_payoff
get_legal_actions = _game.get_legal_actions
card_rank = _game.card_rank
deal_cards = _game.deal_cards
