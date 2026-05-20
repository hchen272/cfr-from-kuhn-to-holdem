"""
Leduc Hold'em — a medium-sized poker game with 2 betting rounds.

Rules
-----
- Deck: 6 cards (K♠ K♥ Q♠ Q♥ J♠ J♥) — 2 suits, 3 ranks.
- Each player antes 1 chip and receives one private card.
- One community card is dealt after the first betting round.
- Two betting rounds (pre-flop / flop), fixed limit:
  - Round 1 bet size: 2 chips.
  - Round 2 bet size: 4 chips.
  - Maximum 2 raises per round.
- Hand evaluation:
    1. Pair (hole matches community) > high card.
    2. Both pairs → higher pair wins.
    3. No pair → higher card wins.
    4. Same rank → split pot.

History encoding
----------------
``build_next_history`` inserts the community card rank automatically
when the first round completes, producing a history like::

    "cc|Kc"   (round 1: c-c, community K, round 2: c)

The ``|com_rank`` separator is 2 characters, preserving player-turn parity
(``len(history) % 2`` = current player).

Cards tuple ``(p0_rank, p1_rank)`` (same shape as Kuhn Poker).

Infoset key (used by node map):
    ``player_rank + history``
    e.g. ``"Jcc"`` (round 1) or ``"Jcc|Kc"`` (round 2).
"""

import random
from games import Game

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANKS = ["J", "Q", "K"]
SUITS = [0, 1]

CALL = "c"
RAISE = "r"
FOLD = "f"

BET_SIZE_R1 = 2
BET_SIZE_R2 = 4
ANTE = 1
MAX_RAISES = 2

_SEP = "|"  # round separator (1 char for the sep + 1 char for community rank = 2 chars total, preserving parity)


def _make_deck():
    return [(r, s) for r in RANKS for s in SUITS]


def _rank_of(card):
    return card[0] if isinstance(card, tuple) else card


# ---------------------------------------------------------------------------
# LeducGame
# ---------------------------------------------------------------------------
class LeducGame(Game):
    """Leduc Hold'em implementation."""

    ACTIONS = [CALL, RAISE, FOLD]

    @property
    def name(self) -> str:
        return "leduc"

    @property
    def num_actions(self) -> int:
        return 3

    @property
    def feature_dim(self) -> int:
        return 20

    # ── Deal ────────────────────────────────────────────────────────────

    def deal_cards(self):
        """Return ``(p0_rank, p1_rank)`` (community stored internally)."""
        deck = _make_deck()
        random.shuffle(deck)
        self._p0 = deck[0]
        self._p1 = deck[1]
        self._comm = deck[2]
        return (_rank_of(deck[0]), _rank_of(deck[1]))

    # ── History helpers ─────────────────────────────────────────────────

    @staticmethod
    def _r1_actions(history: str) -> str:
        """Extract round-1 actions."""
        if _SEP not in history:
            return history
        return history.split(_SEP)[0]

    @staticmethod
    def _r2_actions(history: str) -> str:
        """Extract round-2 actions (empty string if still in round 1)."""
        if _SEP not in history:
            return ""
        parts = history.split(_SEP)
        # parts[1] = community_rank + r2_actions  (e.g., "Kc")
        return parts[1][1:] if len(parts[1]) > 1 else ""

    @staticmethod
    def _community_rank(history: str) -> str:
        """Extract community card rank from history (empty if round 1)."""
        if _SEP not in history:
            return ""
        parts = history.split(_SEP)
        return parts[1][0] if parts[1] else ""

    def _is_r1_complete(self, history: str) -> bool:
        """Round 1 is complete when both have acted and last action is call."""
        r1 = self._r1_actions(history)
        if not r1:
            return False
        # Round 1 ends when both players have acted and last action is call
        # Minimum 2 actions for both to act
        if len(r1) >= 2 and r1[-1] == CALL:
            return True
        return False

    def _is_r2_terminal(self, history: str) -> bool:
        """Check if round 2 has ended with both calling."""
        r2 = self._r2_actions(history)
        if not r2:
            return False
        return len(r2) >= 2 and r2[-1] == CALL

    def _count_raises(self, actions: str) -> int:
        return actions.count(RAISE)

    def build_next_history(self, history: str, action: str) -> str:
        """Extend history, automatically inserting community card
        when transitioning from round 1 to round 2.

        Uses ``|community_rank`` (2 chars) as separator to preserve
        player-turn parity (``len(history) % 2`` = current player).
        """
        if _SEP not in history:
            if self._is_r1_complete(history):
                # Already at round-1 boundary (e.g., "cc"):
                # insert community card before the action
                return history + _SEP + _rank_of(self._comm) + action
            # Check if adding this action completes round 1
            next_r1 = history + action
            if self._is_r1_complete(next_r1):
                return next_r1 + _SEP + _rank_of(self._comm)
        return history + action

    def _pot_size(self, history: str) -> int:
        r1 = self._r1_actions(history)
        r2 = self._r2_actions(history)
        pot = 2 * ANTE
        for a in r1:
            if a in (CALL, RAISE):
                pot += BET_SIZE_R1
        for a in r2:
            if a in (CALL, RAISE):
                pot += BET_SIZE_R2
        return pot

    # ── State queries ───────────────────────────────────────────────────

    def is_terminal(self, history: str) -> bool:
        r1 = self._r1_actions(history)
        r2 = self._r2_actions(history)
        if FOLD in r1 or FOLD in r2:
            return True
        if r2 and self._is_r2_terminal(history):
            return True
        return False

    def get_legal_actions(self, history: str) -> list:
        if self.is_terminal(history):
            return []

        r1 = self._r1_actions(history)
        r2 = self._r2_actions(history)

        if not r2 and not self._is_r1_complete(history):
            # ---- Round 1 ----
            actions = r1
            n_raises = self._count_raises(actions)

            if not actions:
                return [CALL, RAISE]

            if len(actions) == 1:
                legal = [CALL, FOLD]
                if n_raises < MAX_RAISES:
                    legal.append(RAISE)
                return legal

            # 2+ actions in round 1
            last = actions[-1]
            if last == RAISE and n_raises < MAX_RAISES:
                return [CALL, RAISE, FOLD]
            elif last == RAISE:
                return [CALL, FOLD]
            elif last == CALL:
                return [CALL, RAISE]  # transition to round 2
            return []
        else:
            # ---- Round 2 ----
            actions = r2
            n_raises = self._count_raises(actions)

            if not actions:
                return [CALL, RAISE]

            if len(actions) == 1:
                legal = [CALL, FOLD]
                if n_raises < MAX_RAISES:
                    legal.append(RAISE)
                return legal

            last = actions[-1]
            if last == RAISE and n_raises < MAX_RAISES:
                return [CALL, RAISE, FOLD]
            elif last == RAISE:
                return [CALL, FOLD]
            elif last == CALL:
                return []  # round 2 complete
            return []

    def card_rank(self, card) -> int:
        if isinstance(card, tuple):
            card = card[0]
        return {"J": 0, "Q": 1, "K": 2}.get(card, -1)

    def _hand_rank(self, player_rank: str, community_rank: str) -> int:
        return 1 if player_rank == community_rank else 0

    def get_payoff(self, history: str, cards: tuple) -> float:
        p0_rank, p1_rank = cards
        com_rank = self._community_rank(history)
        if not com_rank:
            # Fallback: use internal stored community card
            com_rank = _rank_of(self._comm)
        r1 = self._r1_actions(history)
        r2 = self._r2_actions(history)

        # ---- Fold cases ----
        if FOLD in r1:
            # P0 folded
            if len(r1) - 1 == 0 and r1[-1] == FOLD:
                return -ANTE
            # P1 folded
            return ANTE

        if FOLD in r2:
            # Determine who folded based on position in round 2
            fold_pos = len(r1) + r2.index(FOLD)
            if fold_pos % 2 == 0:
                return -self._pot_size(history) // 2
            else:
                return self._pot_size(history) // 2

        # ---- Showdown ----
        p0_hand = self._hand_rank(p0_rank, com_rank)
        p1_hand = self._hand_rank(p1_rank, com_rank)

        if p0_hand > p1_hand:
            return self._pot_size(history) // 2
        elif p1_hand > p0_hand:
            return -self._pot_size(history) // 2
        else:
            p0_val = self.card_rank(p0_rank)
            p1_val = self.card_rank(p1_rank)
            if p0_val > p1_val:
                return self._pot_size(history) // 2
            elif p1_val > p0_val:
                return -self._pot_size(history) // 2
            else:
                return 0

    # ── Feature encoding ───────────────────────────────────────────────

    def infoset_to_features(self, infoset: str):
        """Feature vector for Deep CFR (dim = 20)."""
        my_rank = infoset[0]

        # Extract community rank and action history from infoset
        # Format: "Jcc|Kc" (player_rank + r1_actions + | + community + r2_actions)
        hist_part = infoset[1:]  # everything after player card
        if _SEP in hist_part:
            parts = hist_part.split(_SEP)
            r1_hist = parts[0]
            rest = parts[1]  # e.g., "Kc"
            com_rank = rest[0] if rest else ""
            r2_hist = rest[1:] if len(rest) > 1 else ""
        else:
            r1_hist = hist_part
            com_rank = ""
            r2_hist = ""

        # My card one-hot
        my_vec = [1.0 if my_rank == c else 0.0 for c in RANKS]

        # Community one-hot
        com_vec = [1.0 if com_rank == c else 0.0 for c in RANKS]

        # Round 1 action slots (up to 6)
        r1_slots = [-1.0] * 6
        for i, a in enumerate(r1_hist):
            r1_slots[i] = 0.0 if a == CALL else 1.0

        # Round 2 action slots
        r2_slots = [-1.0] * 6
        for i, a in enumerate(r2_hist):
            r2_slots[i] = 0.0 if a == CALL else 1.0

        # Pot size normalised
        pot = self._pot_size(infoset[1:])
        pot_feat = min(pot / 30.0, 1.0)

        return my_vec + com_vec + r1_slots + r2_slots + [pot_feat]
