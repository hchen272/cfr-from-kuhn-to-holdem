"""
On-the-fly game tree for Texas Hold'em.

Unlike ``src/algo/tabular/game_tree.py`` which pre-computes the entire BFS
tree, this module traverses the game dynamically.  Each node is a history
string; children are computed on demand via ``game.build_next_history``.

Design
------
- No pre-computed nodes — ``child_history(history, action)`` is a pure
  function of the game rules.
- No payoff cache — ``get_payoff(history, cards)`` evaluates the 7-card
  hand at terminal nodes on the fly (backed by the LRU-cached hand
  evaluator).
- Infoset keys combine the abstraction bucket, community cards, and
  action history.

Interface (mirrors GameTree where possible)
-------------------------------------------
- ``is_terminal(history)`` → bool
- ``legal_actions(history)`` → list[int]
- ``child_history(history, action_idx)`` → str
- ``get_payoff(history, cards)`` → float
- ``infoset_key(bucket, history)`` → str   (unique per infoset)
- ``player(history)`` → int                (0 or 1)

Usage
-----
>>> game = TexasHoldemGame()
>>> tree = OnTheFlyTree(game, CardAbstraction(n_buckets=50))
>>> hist = ""
>>> while not tree.is_terminal(hist):
...     actions = tree.legal_actions(hist)
...     a = actions[0]
...     hist = tree.child_history(hist, a)
"""
import sys
import os

if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import TexasHoldemGame, _SEP, _card_str, CALL, RAISE, FOLD


class OnTheFlyTree:
    """On-the-fly traversal helper for Texas Hold'em.

    Parameters
    ----------
    game : TexasHoldemGame
        Game instance (with cards already dealt).
    abstraction : CardAbstraction
        Maps hole cards → bucket ids.
    """

    def __init__(self, game, abstraction=None):
        self.game = game
        self.abstraction = abstraction
        self.num_actions = game.num_actions

        # Action mapping (same as GameTree for compatibility)
        self._a2i = {a: i for i, a in enumerate(game.ACTIONS)}  # {'c':0,'r':1,'f':2}
        self._i2a = {i: a for i, a in enumerate(game.ACTIONS)}  # {0:'c',1:'r',2:'f'}

    # ── Core delegation ───────────────────────────────────────────────

    def is_terminal(self, history: str) -> bool:
        return self.game.is_terminal(history)

    def legal_actions(self, history: str) -> list:
        """Return list of action *indices* (not chars)."""
        chars = self.game.get_legal_actions(history)
        return [self._a2i[a] for a in chars]

    def child_history(self, history: str, action_idx: int) -> str:
        """Return the history after taking *action_idx*."""
        a = self._i2a.get(action_idx, 'c')
        return self.game.build_next_history(history, a)

    def player(self, history: str) -> int:
        return self.game.current_player(history)

    def get_payoff(self, history: str, cards: tuple) -> float:
        """Payoff from P0's perspective (terminal only)."""
        return self.game.get_payoff(history, cards)

    # ── Infoset key ───────────────────────────────────────────────────

    def infoset_key(self, bucket: int, history: str) -> str:
        """Build a unique string key for an information set.

        Format: ``"<bucket>|<history>"``

        The history already encodes all actions and round transitions.
        Community cards are implicitly included via the game state (set
        before payoff evaluation).  For infoset purposes we embed visible
        community cards from each section of history.

        Returns a compact key used as dict key for Node lookups.
        """
        # Embed visible community cards based on round progress
        parts = history.split(_SEP)
        ri = history.count(_SEP)  # current round index (0=preflop, 1=flop, ...)

        key_parts = [str(bucket)]

        # Preflop actions
        key_parts.append(parts[0] if len(parts) > 0 else '')

        # Flop: add community cards if we're past preflop
        if ri >= 1 and len(self.game._community_cards) >= 3:
            comm_3 = ''.join(_card_str(c) for c in self.game._community_cards[:3])
            key_parts.append(comm_3)
            key_parts.append(parts[1] if len(parts) > 1 else '')

        # Turn
        if ri >= 2 and len(self.game._community_cards) >= 4:
            comm_1 = _card_str(self.game._community_cards[3])
            key_parts.append(comm_1)
            key_parts.append(parts[2] if len(parts) > 2 else '')

        # River
        if ri >= 3 and len(self.game._community_cards) >= 5:
            comm_1 = _card_str(self.game._community_cards[4])
            key_parts.append(comm_1)
            key_parts.append(parts[3] if len(parts) > 3 else '')

        return _SEP.join(key_parts)

    # ── Round info ────────────────────────────────────────────────────

    def current_round(self, history: str) -> int:
        """0=preflop, 1=flop, 2=turn, 3=river."""
        return self.game._current_round(history)


# ── Smoke test ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from abstraction import CardAbstraction

    print("Building abstraction...")
    absn = CardAbstraction(n_buckets=50, n_mc_samples=2000)

    game = TexasHoldemGame()
    tree = OnTheFlyTree(game, absn)

    # Deal cards
    cards = game.deal_cards()
    p0h, p1h = cards
    p0b = absn.bucket_id(p0h)
    p1b = absn.bucket_id(p1h)

    print(f"P0: {_card_str(p0h[0])} {_card_str(p0h[1])} → bucket {p0b}")
    print(f"P1: {_card_str(p1h[0])} {_card_str(p1h[1])} → bucket {p1b}")
    print(f"Community: {[_card_str(c) for c in game._community_cards]}")

    # Walk a sample hand
    hist = ""
    print(f"\nWalking tree:")
    while not tree.is_terminal(hist):
        legal = tree.legal_actions(hist)
        p = tree.player(hist)
        b = p0b if p == 0 else p1b
        key = tree.infoset_key(b, hist)
        print(f"  player={p}  bucket={b}  round={tree.current_round(hist)}  "
              f"hist='{hist}'  legal={legal}  infoset='{key}'")
        if not legal:
            break
        a = legal[0]
        hist = tree.child_history(hist, a)

    payoff = tree.get_payoff(hist, cards)
    print(f"\nTerminal: hist='{hist}'")
    print(f"Payoff (P0): {payoff:+.4f}")
    print(f"Infoset P0: {tree.infoset_key(p0b, hist)}")
    print(f"Infoset P1: {tree.infoset_key(p1b, hist)}")
