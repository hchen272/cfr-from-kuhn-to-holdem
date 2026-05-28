"""
River Poker — simplified 1-street Hold'em with 2 hole cards per player.

Rules
-----
- Deck: 8 cards (A♠A♥ K♠K♥ Q♠Q♥ J♠J♥) — 2 suits, 4 ranks.
- 2 private hole cards per player, 1 community card.
- Two betting rounds (pre / post community), fixed limit:
  - Round 1 bet size: 2 chips (= BB).
  - Round 2 bet size: 4 chips (= 2×BB).
  - Maximum 1 bet + 2 raises per round (MAX_RAISES=3).
- Blinds: P0 = SB (1 chip), P1 = BB (2 chips).
- Hand evaluation (3 cards: 2 hole + 1 community):
    1. Pair (two cards of same rank) > high card.
    2. Higher pair wins.
    3. No pair → compare highest card, then second-highest.
    4. Identical → split pot.

Player hands are represented as sorted 2-char rank strings (e.g. "JQ", "AK", "JJ").
This keeps infoset keys consistent and human-readable.

History encoding
----------------
Same as Leduc: ``"cc|Kc"`` — ``|`` + community card rank separates rounds
(2 chars, preserving player-turn parity).

Infoset key:
    ``sorted_hole_cards + history``
    e.g. ``"JQcc|K"`` (R2, community K, P0 has J and Q).

Payoffs normalised by BB (= 2).  Nash value unknown (~ −0.08 to −0.12 estimated).
"""

import random
from games import Game

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANKS = ["J", "Q", "K", "A"]
SUITS = [0, 1]

CALL = "c"
RAISE = "r"
FOLD = "f"

BET_SIZE_R1 = 2
BET_SIZE_R2 = 4
BB = 2
SB = 1
MAX_RAISES = 3

_SEP = "|"


def _make_deck():
    return [(r, s) for r in RANKS for s in SUITS]


def _rank_of(card):
    return card[0] if isinstance(card, tuple) else card


def _hand_str(card1, card2):
    """Return sorted 2-char hand string, e.g. ('Q','J') → 'JQ'."""
    a, b = _rank_of(card1), _rank_of(card2)
    return a + b if RANKS.index(a) <= RANKS.index(b) else b + a


def _hand_ranks(hand_str):
    """Return list of 2 individual rank chars from hand string."""
    return [hand_str[0], hand_str[1]]


# All possible 2-card hand strings (rank-based, suits ignored)
_ALL_HAND_TYPES = []
for i, r1 in enumerate(RANKS):
    for j, r2 in enumerate(RANKS):
        if i <= j:
            _ALL_HAND_TYPES.append(r1 + r2)


# ---------------------------------------------------------------------------
# RiverPokerGame
# ---------------------------------------------------------------------------
class RiverPokerGame(Game):
    """River Poker — 2 hole cards, 1 community card, 2 betting rounds."""

    RANKS = ["J", "Q", "K", "A"]
    SUIT_COUNT = 2
    HAND_TYPES = _ALL_HAND_TYPES   # 10 possible rank-based 2-card hands
    ACTIONS = [CALL, RAISE, FOLD]

    @property
    def name(self) -> str:
        return "river_poker"

    @property
    def num_actions(self) -> int:
        return 3

    @property
    def feature_dim(self) -> int:
        # 10 (hand one-hot) + 4 (community one-hot) + 6 (R1) + 6 (R2) + 1 (pot) = 27
        return 27

    @property
    def nash_value(self) -> float:
        return None  # unknown — use variance-based checkpointing

    # ── Deal ────────────────────────────────────────────────────────────

    def deal_cards(self):
        """Return ``(p0_hand_str, p1_hand_str)`` — sorted 2-char strings."""
        deck = _make_deck()
        random.shuffle(deck)
        self._p0_c1 = deck[0]
        self._p0_c2 = deck[1]
        self._p1_c1 = deck[2]
        self._p1_c2 = deck[3]
        self._comm = deck[4]
        p0_str = _hand_str(deck[0], deck[1])
        p1_str = _hand_str(deck[2], deck[3])
        return (p0_str, p1_str)

    # ── History helpers ─────────────────────────────────────────────────

    @staticmethod
    def _r1_actions(history: str) -> str:
        if _SEP not in history:
            return history
        return history.split(_SEP)[0]

    @staticmethod
    def _r2_actions(history: str) -> str:
        if _SEP not in history:
            return ""
        parts = history.split(_SEP)
        return parts[1][1:] if len(parts[1]) > 1 else ""

    @staticmethod
    def _community_rank(history: str) -> str:
        if _SEP not in history:
            return ""
        parts = history.split(_SEP)
        return parts[1][0] if parts[1] else ""

    def _is_r1_complete(self, history: str) -> bool:
        r1 = self._r1_actions(history)
        if not r1:
            return False
        return len(r1) >= 2 and r1[-1] == CALL

    def _is_r2_terminal(self, history: str) -> bool:
        r2 = self._r2_actions(history)
        if not r2:
            return False
        return len(r2) >= 2 and r2[-1] == CALL

    def _count_raises(self, actions: str) -> int:
        return actions.count(RAISE)

    def _player_investment(self, history: str, player: int) -> int:
        raised = [SB, BB]
        r1 = self._r1_actions(history)
        r2 = self._r2_actions(history)

        for i, a in enumerate(r1):
            p = i % 2
            if a == RAISE:
                raised[p] = max(raised) + BET_SIZE_R1
            elif a == CALL:
                raised[p] = max(raised)

        for i, a in enumerate(r2):
            p = i % 2
            if a == RAISE:
                raised[p] = max(raised) + BET_SIZE_R2
            elif a == CALL:
                raised[p] = max(raised)

        return raised[player]

    def build_next_history(self, history: str, action: str) -> str:
        if _SEP not in history:
            if self._is_r1_complete(history):
                return history + _SEP + _rank_of(self._comm) + action
            next_r1 = history + action
            if self._is_r1_complete(next_r1):
                return next_r1 + _SEP + _rank_of(self._comm)
        return history + action

    def _pot_size(self, history: str) -> int:
        raised = [SB, BB]
        r1 = self._r1_actions(history)
        r2 = self._r2_actions(history)
        for i, a in enumerate(r1):
            p = i % 2
            if a == RAISE:
                raised[p] = max(raised) + BET_SIZE_R1
            elif a == CALL:
                raised[p] = max(raised)
        for i, a in enumerate(r2):
            p = i % 2
            if a == RAISE:
                raised[p] = max(raised) + BET_SIZE_R2
            elif a == CALL:
                raised[p] = max(raised)
        return sum(raised)

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
            actions = r1
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
                return [CALL, RAISE]
            return []
        else:
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
                return []
            return []

    def card_rank(self, card) -> int:
        return {r: i for i, r in enumerate(self.RANKS)}.get(card, -1)

    # ── Hand evaluation ─────────────────────────────────────────────────

    def _evaluate_hand(self, hand_str: str, community_rank: str):
        """Evaluate best hand from 2 hole + 1 community card.

        Returns (hand_type, main_rank, kicker_rank) where:
          hand_type: 1 = pair, 0 = high card
          main_rank: rank index of the pair (or -1 for high card)
          kicker_rank: highest non-pair rank index (or -1 if no kicker)
        """
        cards = [hand_str[0], hand_str[1], community_rank]
        ranks = sorted([self.card_rank(c) for c in cards], reverse=True)

        # Check for pair
        if cards[0] == cards[1] or cards[0] == community_rank:
            pr = self.card_rank(cards[0])
            kickers = [r for r in ranks if r != pr]
            return (1, pr, max(kickers) if kickers else -1)
        elif cards[1] == community_rank:
            pr = self.card_rank(cards[1])
            kickers = [r for r in ranks if r != pr]
            return (1, pr, max(kickers) if kickers else -1)
        else:
            return (0, -1, ranks[0])

    def get_payoff(self, history: str, cards: tuple) -> float:
        """cards = (p0_hand_str, p1_hand_str) e.g. ('JQ', 'KA')."""
        p0_hand, p1_hand = cards
        com_rank = self._community_rank(history)
        if not com_rank:
            com_rank = _rank_of(self._comm)
        r1 = self._r1_actions(history)
        r2 = self._r2_actions(history)

        p0_inv = self._player_investment(history, 0)
        p1_inv = self._player_investment(history, 1)

        # ---- Fold ----
        if FOLD in r1:
            fold_idx = r1.index(FOLD)
            raw = float(-p0_inv) if fold_idx % 2 == 0 else float(p1_inv)
            return raw / BB

        if FOLD in r2:
            fold_idx = len(r1) + r2.index(FOLD)
            raw = float(-p0_inv) if fold_idx % 2 == 0 else float(p1_inv)
            return raw / BB

        # ---- Showdown ----
        p0_type, p0_main, p0_kicker = self._evaluate_hand(p0_hand, com_rank)
        p1_type, p1_main, p1_kicker = self._evaluate_hand(p1_hand, com_rank)

        # Compare hand types
        if p0_type > p1_type:
            raw = float(p1_inv)
        elif p1_type > p0_type:
            raw = float(-p0_inv)
        elif p0_type == 1:  # both have pairs
            if p0_main > p1_main:
                raw = float(p1_inv)
            elif p1_main > p0_main:
                raw = float(-p0_inv)
            elif p0_kicker > p1_kicker:
                raw = float(p1_inv)
            elif p1_kicker > p0_kicker:
                raw = float(-p0_inv)
            else:
                raw = float(p1_inv - p0_inv) / 2.0
        else:  # both high card — compare kickers (highest cards)
            p0_cards = sorted([self.card_rank(c) for c in [p0_hand[0], p0_hand[1], com_rank]], reverse=True)
            p1_cards = sorted([self.card_rank(c) for c in [p1_hand[0], p1_hand[1], com_rank]], reverse=True)
            # Compare highest, then second-highest
            winner = 0
            for i in range(2):  # compare top 2 cards; 3rd would be community (same for both)
                if p0_cards[i] > p1_cards[i]:
                    winner = 0; break
                elif p1_cards[i] > p0_cards[i]:
                    winner = 1; break
            else:
                winner = -1  # tie (split pot)
            if winner == 0:
                raw = float(p1_inv)
            elif winner == 1:
                raw = float(-p0_inv)
            else:
                raw = float(p1_inv - p0_inv) / 2.0

        return raw / BB

    # ── Feature encoding ───────────────────────────────────────────────

    def infoset_to_features(self, infoset: str):
        """Feature vector for Deep CFR (dim = 27)."""
        # infoset format: "JQ_cc|K" or "JQ_" etc
        # Hand one-hot
        hand_vec = [1.0 if infoset[:2] == ht else 0.0 for ht in _ALL_HAND_TYPES]  # 10

        hist_part = infoset[2:]  # skip the 2-char hand
        if _SEP in hist_part:
            parts = hist_part.split(_SEP)
            r1_hist = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            com_rank = rest[0] if rest else ""
            r2_hist = rest[1:] if len(rest) > 1 else ""
        else:
            r1_hist = hist_part
            com_rank = ""
            r2_hist = ""

        com_vec = [1.0 if com_rank == c else 0.0 for c in self.RANKS]  # 4

        r1_slots = [-1.0] * 6
        for i, a in enumerate(r1_hist):
            r1_slots[i] = 0.0 if a == CALL else 1.0

        r2_slots = [-1.0] * 6
        for i, a in enumerate(r2_hist):
            r2_slots[i] = 0.0 if a == CALL else 1.0

        pot = self._pot_size(hist_part)
        pot_feat = min(pot / 30.0, 1.0)

        return hand_vec + com_vec + r1_slots + r2_slots + [pot_feat]
