"""
Texas Hold'em — Fixed-Limit, heads-up, 52-card deck.

Game rules
----------
- 52-card deck (4 suits × 13 ranks, 2–A).
- 2 hole cards per player, 5 community cards (flop 3 / turn 1 / river 1).
- 4 betting rounds: preflop, flop, turn, river.
- Fixed-limit: small bet = 2 (flop), big bet = 4 (turn / river).
  Preflop raise = 2 (matching the small bet).
- MAX_RAISES = 4 per round.
- Blinds: P0 = SB (1 chip), P1 = BB (2 chips).
- Payoffs normalised by BB (= 2).

History encoding
----------------
``"<preflop>|<flop>|<turn>|<river>"`` — actions only, one section per round,
sections separated by ``|``.

Community cards are stored externally (``game._community_cards``) to keep
the history string simple and parity-agnostic.

Infoset key (built by OnTheFlyTree)::

    "<hole_cards>|<preflop>|<visible_community>|<flop>|<visible_community>|
     <turn>|<visible_community>|<river>"

Example: ``"AKs|crrc|AhKhQh|ccr|Jh|cr|Th|cc"``

Hand evaluation
---------------
Standard 7-card evaluation (best 5 of 7).  Rankings (descending):
    9 — straight-flush (incl. royal)
    8 — four of a kind
    7 — full house
    6 — flush
    5 — straight
    4 — three of a kind
    3 — two pair
    2 — one pair
    1 — high card
"""
import sys
import os
import random

# Allow running as script or as package
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_eval import evaluate_7 as _eval_7

# ── Constants ───────────────────────────────────────────────────────────
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['h', 'd', 'c', 's']
_RANK_ORDER = {r: i for i, r in enumerate(RANKS)}

CALL = 'c'
RAISE = 'r'
FOLD = 'f'

BET_SMALL = 2          # flop bet / preflop raise increment
BET_BIG = 4            # turn & river bet
BB = 2
SB = 1
MAX_RAISES = 4         # per round

_SEP = '|'

# Round indices
PREFLOP, FLOP, TURN, RIVER = range(4)

# ── Deck ─────────────────────────────────────────────────────────────────


def _make_deck():
    """Return list of 52 (rank_char, suit_char) tuples."""
    return [(r, s) for r in RANKS for s in SUITS]


def _card_str(card):
    """'Ah', 'Td', etc."""
    return card[0] + card[1]


def _rank_of(card):
    return card[0] if isinstance(card, tuple) else card


# ── Game class ───────────────────────────────────────────────────────────


class TexasHoldemGame:
    """Fixed-Limit heads-up Texas Hold'em.

    Implements a Game-ABC-compatible interface but stands alone in the
    ``holdem/`` module (no import from ``src.games``).
    """

    # Class-level constants (Game ABC compatibility)
    CARDS = []
    ACTIONS = [CALL, RAISE, FOLD]
    RANKS = RANKS
    SUIT_COUNT = 4
    # No HAND_TYPES — hole cards are full (rank, suit) tuples; abstraction
    # will be handled by ``abstraction.py``.

    def __init__(self):
        self._deck = _make_deck()
        self._community_cards = []
        self._comm = ()       # compatibility shim

    # ── Properties (Game ABC) ─────────────────────────────────────────

    @property
    def name(self) -> str:
        return "texas_holdem"

    @property
    def num_actions(self) -> int:
        return 3

    @property
    def feature_dim(self) -> int:
        # Placeholder — Deep CFR not targeted yet
        return 256

    @property
    def nash_value(self) -> float:
        return None  # unknown

    # ── Deal ──────────────────────────────────────────────────────────

    def deal_cards(self):
        """Shuffle & deal. Returns ``(p0_hole, p1_hole)`` — each a
        2-tuple of (rank, suit) pairs, e.g. ``(('A','h'), ('K','h'))``.
        """
        deck = list(self._deck)
        random.shuffle(deck)
        self._p0_hole = (deck[0], deck[1])
        self._p1_hole = (deck[2], deck[3])
        # 5 community cards: flop, turn, river
        self._community_cards = list(deck[4:9])
        self._comm = ()
        return (self._p0_hole, self._p1_hole)

    # ── History helpers ───────────────────────────────────────────────

    @staticmethod
    def _round_sections(history: str):
        """Return list of per-round action strings.  Always length 4."""
        parts = history.split(_SEP)
        while len(parts) < 4:
            parts.append('')
        return parts[:4]

    @staticmethod
    def _current_round(history: str) -> int:
        """Which betting round are we in? 0=preflop … 3=river."""
        # Count how many section separators are present
        sep_count = history.count(_SEP)
        return sep_count

    def _round_actions(self, history: str, round_idx: int) -> str:
        return self._round_sections(history)[round_idx]

    def _total_actions(self, history: str) -> int:
        return sum(len(s) for s in self._round_sections(history))

    def current_player(self, history: str) -> int:
        """P0 or P1 based on total actions taken."""
        return self._total_actions(history) % 2

    @staticmethod
    def _count_raises(actions: str) -> int:
        return actions.count(RAISE)

    def _bet_size(self, round_idx: int) -> int:
        """Bet size for *round_idx*."""
        if round_idx == PREFLOP:
            return BET_SMALL   # preflop raise = 2
        elif round_idx == FLOP:
            return BET_SMALL   # 2
        else:
            return BET_BIG     # turn / river = 4

    # ── Investment / Pot ─────────────────────────────────────────────

    def _player_investment(self, history: str, player: int) -> int:
        """Chips *player* has put in so far."""
        # Each player starts with their blind
        invested = [SB, BB]  # P0, P1

        for ri in range(4):
            actions = self._round_actions(history, ri)
            bet = self._bet_size(ri)
            for i, a in enumerate(actions):
                p = i % 2
                if a == RAISE:
                    invested[p] = max(invested) + bet
                elif a == CALL:
                    invested[p] = max(invested)

        return invested[player]

    def _pot_size(self, history: str) -> int:
        return (self._player_investment(history, 0) +
                self._player_investment(history, 1))

    # ── History building ─────────────────────────────────────────────

    def build_next_history(self, history: str, action: str) -> str:
        """Append *action* to history.  Insert round separator if the
        round just completed."""
        parts = self._round_sections(history)
        cur = self._current_round(history)
        current_actions = parts[cur]

        # Check if adding this action completes the round
        next_actions = current_actions + action
        round_done = self._round_complete(next_actions, cur)

        if round_done and cur < RIVER:
            # move to next round
            parts[cur] = next_actions
            return _SEP.join(parts[:cur + 2])  # one more section
        else:
            parts[cur] = next_actions
            # rebuild without trailing empty sections
            while parts and parts[-1] == '':
                parts.pop()
            return _SEP.join(parts)

    def _round_complete(self, actions: str, round_idx: int) -> bool:
        """Has the round ended with two consecutive call-equivalent actions?"""
        if not actions:
            return False
        if FOLD in actions:
            return True  # fold ends the round (and game)
        if len(actions) < 2:
            return False
        # Round complete when last two actions are both calls/checks
        # AND enough actions have happened (minimum 2 for initial round
        # where blinds are already posted)
        if round_idx == PREFLOP:
            # Preflop: blinds are posted; P1(BB) already has 1 chip committed.
            # P0(SB) acts first.  After P0 calls/raises and P1 responds,
            # if the last two are calls → round done.
            # But also P0 can fold immediately → 1 action with fold = done.
            return actions[-1] == CALL and len(actions) >= 2
        else:
            # Postflop: first two can be check-check (cc)
            if actions[-1] == CALL:
                # Need: at least 2 actions AND no pending raise
                if len(actions) >= 2 and actions[-2] == CALL:
                    return True
                # Or: facing a raise, then call ends
                if RAISE in actions:
                    last_raise = actions.rfind(RAISE)
                    if len(actions) > last_raise + 1 and actions[-1] == CALL:
                        # After a raise, any call ends the round
                        return True
            return False

    # ── State queries ────────────────────────────────────────────────

    def is_terminal(self, history: str) -> bool:
        parts = self._round_sections(history)
        # Fold anywhere ends the hand
        for p in parts:
            if FOLD in p:
                return True
        ri = self._current_round(history)
        if ri < RIVER:
            return False
        # River round must be complete
        return self._round_complete(parts[RIVER], RIVER)

    def get_legal_actions(self, history: str) -> list:
        if self.is_terminal(history):
            return []

        ri = self._current_round(history)
        actions = self._round_actions(history, ri)
        n_raises = self._count_raises(actions)

        if ri == PREFLOP:
            # Blinds posted: P0(SB) at 1, P1(BB) at 2
            if not actions:
                # P0 first to act: call (match BB), raise, or fold
                return [CALL, RAISE, FOLD]
            if len(actions) == 1:
                # P1 facing P0's action
                if actions[0] == CALL:
                    # P0 just called → P1 can check (call) or raise
                    return [CALL, RAISE]
                elif actions[0] == RAISE:
                    # P0 raised → P1 can call, raise, fold
                    if n_raises < MAX_RAISES:
                        return [CALL, RAISE, FOLD]
                    else:
                        return [CALL, FOLD]
                elif actions[0] == FOLD:
                    return []
            # More actions → general case
            last = actions[-1]
            if last == RAISE:
                if n_raises < MAX_RAISES:
                    return [CALL, RAISE, FOLD]
                else:
                    return [CALL, FOLD]
            elif last == CALL:
                # After a call, if facing a raise earlier → need to respond
                # Otherwise the round should be complete
                return [CALL, RAISE]
            return []
        else:
            # Postflop rounds
            if not actions:
                return [CALL, RAISE]  # check or bet
            # Facing action
            last = actions[-1]
            if last == RAISE:
                if n_raises < MAX_RAISES:
                    return [CALL, RAISE, FOLD]
                else:
                    return [CALL, FOLD]
            elif last == CALL:
                if len(actions) == 1:
                    # Facing a single call? Can't happen normally
                    return [CALL, RAISE]
                elif actions[-2] == CALL and len(actions) >= 2:
                    # Two consecutive calls → round complete
                    return []
                # Otherwise this call was responding to a raise
                return [CALL, RAISE]
            return []

    # ── Payoff ────────────────────────────────────────────────────────

    def get_payoff(self, history: str, cards: tuple) -> float:
        """cards = (p0_hole, p1_hole) — each a 2-tuple of (rank, suit)."""
        p0_hole, p1_hole = cards
        parts = self._round_sections(history)

        p0_inv = self._player_investment(history, 0)
        p1_inv = self._player_investment(history, 1)

        # ── Fold ──
        for ri, acts in enumerate(parts):
            if FOLD in acts:
                fold_idx = sum(len(parts[r]) for r in range(ri)) + acts.index(FOLD)
                raw = float(-p0_inv) if fold_idx % 2 == 0 else float(p1_inv)
                return raw / BB

        # ── Showdown ──
        # Determine visible community cards
        visible_comm = []
        if len(parts) > 1 and parts[1]:   # flop actions exist
            visible_comm = self._community_cards[:3]
        if len(parts) > 2 and parts[2]:   # turn actions exist
            visible_comm = self._community_cards[:4]
        if len(parts) > 3 and parts[3]:   # river actions exist
            visible_comm = self._community_cards[:5]

        p0_best = _eval_7(list(p0_hole) + visible_comm)
        p1_best = _eval_7(list(p1_hole) + visible_comm)

        if p0_best > p1_best:
            raw = float(p1_inv)
        elif p1_best > p0_best:
            raw = float(-p0_inv)
        else:
            raw = float(p1_inv - p0_inv) / 2.0  # split pot

        return raw / BB

    # ── Misc ──────────────────────────────────────────────────────────

    def card_rank(self, card) -> int:
        return _RANK_ORDER.get(_rank_of(card), -1)

    def infoset_to_features(self, infoset: str):
        """Placeholder — neural features not implemented yet."""
        raise NotImplementedError("Deep CFR features not yet implemented for Texas Hold'em")


# ── Quick smoke test ─────────────────────────────────────────────────────
if __name__ == '__main__':
    game = TexasHoldemGame()
    print(f"Game: {game.name}")
    print(f"Actions: {game.ACTIONS}")
    print(f"Num actions: {game.num_actions}")

    game.deal_cards()
    p0h, p1h = game._p0_hole, game._p1_hole
    comm = game._community_cards
    print(f"P0: {_card_str(p0h[0])} {_card_str(p0h[1])}")
    print(f"P1: {_card_str(p1h[0])} {_card_str(p1h[1])}")
    print(f"Community: {[_card_str(c) for c in comm]}")

    # Walk a sample hand
    hist = ""
    print(f"\n--- Walk sample hand ---")
    print(f"Start: hist='{hist}', player={game.current_player(hist)}")
    while not game.is_terminal(hist):
        legal = game.get_legal_actions(hist)
        print(f"  hist='{hist}'  player={game.current_player(hist)}  "
              f"round={game._current_round(hist)}  legal={legal}")
        if not legal:
            break
        # Pick first legal action for deterministic walk
        a = legal[0]
        hist = game.build_next_history(hist, a)
    print(f"Final: hist='{hist}'  terminal={game.is_terminal(hist)}")
    payoff = game.get_payoff(hist, (p0h, p1h))
    print(f"Payoff (P0): {payoff:+.4f}")
