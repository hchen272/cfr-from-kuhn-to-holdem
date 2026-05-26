"""
Kuhn Poker — a minimal 3-card, single-round poker game.

Rules
-----
- Deck: J, Q, K (3 cards, no suits).
- Each player antes 1 chip and receives one private card.
- One betting round: pass-or-bet with at most one raise per player.
- If both pass (pp) → showdown, high card wins 1 chip.
- If bet and call (bb / pbb) → showdown, high card wins 2 chips.
- If bet and fold (bp / pbp) → bettor wins the pot (1 chip profit).

Information sets: 12 (3 cards × 4 betting histories).
"""

from games import Game


class KuhnGame(Game):
    """Kuhn Poker implementation."""

    CARDS = ["J", "Q", "K"]
    RANKS = ["J", "Q", "K"]
    SUIT_COUNT = 1
    PASS = "p"
    BET = "b"
    ACTIONS = [PASS, BET]

    # ── Metadata ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "kuhn"

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def feature_dim(self) -> int:
        return 5

    @property
    def nash_value(self) -> float:
        return -1.0 / 18.0   # ≈ -0.0556

    # ── State queries ───────────────────────────────────────────────────

    def is_terminal(self, history: str) -> bool:
        """Terminal histories: pp, bp, bb, pbp, pbb."""
        return history in ("pp", "bp", "bb", "pbp", "pbb")

    def get_legal_actions(self, history: str) -> list:
        if self.is_terminal(history):
            return []
        return self.ACTIONS

    def card_rank(self, card: str) -> int:
        return {"J": 0, "Q": 1, "K": 2}.get(card, -1)

    def get_payoff(self, history: str, cards: tuple) -> float:
        """Payoff from player-0's perspective."""
        player0, player1 = cards

        if history == "pp":
            return 1 if self.card_rank(player0) > self.card_rank(player1) else -1

        if history == "bp":
            return 1

        if history == "pbp":
            return -1

        if history in ("bb", "pbb"):
            if self.card_rank(player0) > self.card_rank(player1):
                return 2
            else:
                return -2

        raise ValueError(f"Invalid terminal history: {history}")

    # ── Deal ────────────────────────────────────────────────────────────

    def deal_cards(self) -> tuple:
        import random
        deck = self.CARDS.copy()
        random.shuffle(deck)
        return deck[0], deck[1]

    # ── Feature encoding (for Deep CFR) ────────────────────────────────

    def infoset_to_features(self, infoset: str):
        """
        Convert an infoset string into a fixed-size feature vector.

        Features (dim = 5)::
            [is_J, is_Q, is_K, first_action, second_action]

        Action slots: -1 = no action yet, 0 = pass/check/fold, 1 = bet/call.
        """
        card = infoset[0]
        history = infoset[1:]

        # Card one-hot
        card_vec = [1.0 if card == c else 0.0 for c in ("J", "Q", "K")]

        # History encoding (up to 2 slots)
        slot = [-1.0, -1.0]
        for i, action in enumerate(history):
            slot[i] = 0.0 if action == "p" else 1.0

        return card_vec + slot  # 3 + 2 = 5 floats
