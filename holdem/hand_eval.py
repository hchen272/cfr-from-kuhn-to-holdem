"""
Fast 7-card Texas Hold'em hand evaluator.

Evaluates best 5-card hand from 7 cards (2 hole + 5 community).
Returns a comparable tuple ``(hand_rank, tiebreakers…)``.

Strategy
--------
- Brute-force over C(7,5)=21 combos for correctness.
- LRU cache on the 7-card tuple to accelerate repeated evaluations during
  training / MC sampling.

Hand rankings (descending)
--------------------------
    9 — straight-flush (incl. royal flush)
    8 — four of a kind
    7 — full house
    6 — flush
    5 — straight
    4 — three of a kind
    3 — two pair
    2 — one pair
    1 — high card

Usage
-----
>>> from hand_eval import evaluate_7
>>> rank = evaluate_7(cards_7)  # cards_7: iterable of (rank_char, suit_char)
>>> rank
(2, 10, 7, 5, 2)  # one pair of tens, kickers 7, 5, 2
"""
from functools import lru_cache
from itertools import combinations as _combos

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['h', 'd', 'c', 's']
_RANK_ORDER = {r: i for i, r in enumerate(RANKS)}


def rank_index(card):
    """Numeric rank index: '2'→0 … 'A'→12."""
    r = card[0] if isinstance(card, tuple) else card
    return _RANK_ORDER.get(r, -1)


# ── Core evaluator ────────────────────────────────────────────────────────


def _eval_5(ranks, suits):
    """Evaluate exactly 5 cards.

    Args:
        ranks: list of 5 rank chars  (e.g. ['A','K','Q','J','T'])
        suits: list of 5 suit chars   (e.g. ['h','h','h','h','h'])

    Returns:
        tuple: ``(hand_rank, tiebreaker0, tiebreaker1, …)``
        Lexicographic comparison yields correct hand ordering.
    """
    is_flush = len(set(suits)) == 1

    # Count occurrences of each rank
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1
    counts_sorted = sorted(rank_counts.values(), reverse=True)

    # Ranks sorted by frequency (desc) then rank value (desc)
    by_freq = sorted(rank_counts.items(),
                     key=lambda x: (-x[1], -_RANK_ORDER[x[0]]))

    # All ranks in descending order (for high-card / flush tiebreaker)
    sorted_ranks = sorted(ranks, key=lambda r: _RANK_ORDER[r], reverse=True)
    sorted_vals = tuple(_RANK_ORDER[r] for r in sorted_ranks)

    # Straight detection
    rank_vals = sorted(set(_RANK_ORDER[r] for r in ranks))
    is_straight = False
    straight_high = 0
    if len(rank_vals) == 5:
        if rank_vals[-1] - rank_vals[0] == 4:
            is_straight = True
            straight_high = rank_vals[-1]
    # Wheel: A-2-3-4-5
    if not is_straight and set(rank_vals) == {0, 1, 2, 3, 12}:
        is_straight = True
        straight_high = 3   # 5-high

    # ── Categorize ──
    # 9 — straight-flush
    if is_flush and is_straight:
        return (9, straight_high)

    # 8 — four of a kind
    if counts_sorted == [4, 1]:
        quad_r = _RANK_ORDER[by_freq[0][0]]
        kicker = _RANK_ORDER[by_freq[1][0]]
        return (8, quad_r, kicker)

    # 7 — full house
    if counts_sorted == [3, 2]:
        trips_r = _RANK_ORDER[by_freq[0][0]]
        pair_r = _RANK_ORDER[by_freq[1][0]]
        return (7, trips_r, pair_r)

    # 6 — flush
    if is_flush:
        return (6,) + sorted_vals

    # 5 — straight
    if is_straight:
        return (5, straight_high)

    # 4 — three of a kind
    if counts_sorted == [3, 1, 1]:
        trips_r = _RANK_ORDER[by_freq[0][0]]
        kickers = tuple(sorted(
            (_RANK_ORDER[r] for r in ranks if r != by_freq[0][0]),
            reverse=True))
        return (4, trips_r) + kickers

    # 3 — two pair
    if counts_sorted == [2, 2, 1]:
        p0, p1 = sorted([_RANK_ORDER[by_freq[0][0]],
                         _RANK_ORDER[by_freq[1][0]]], reverse=True)
        kicker = _RANK_ORDER[by_freq[2][0]]
        return (3, p0, p1, kicker)

    # 2 — one pair
    if counts_sorted == [2, 1, 1, 1]:
        pair_r = _RANK_ORDER[by_freq[0][0]]
        kickers = tuple(sorted(
            (_RANK_ORDER[r] for r in ranks if r != by_freq[0][0]),
            reverse=True))
        return (2, pair_r) + kickers

    # 1 — high card
    return (1,) + sorted_vals


def _evaluate_7_uncached(cards_7):
    """Brute-force best of C(7,5)=21 combos."""
    best = (0,)
    for combo in _combos(cards_7, 5):
        r = tuple(c[0] for c in combo)   # ranks
        s = tuple(c[1] for c in combo)   # suits
        val = _eval_5(r, s)
        if val > best:
            best = val
    return best


# ── Cached public API ─────────────────────────────────────────────────────

_CARDS_52 = [(r, s) for r in RANKS for s in ['h', 'd', 'c', 's']]
_CARD_ID = {c: i for i, c in enumerate(_CARDS_52)}


def _canonical_7(cards_7):
    """Convert any 7-card iterable to a sorted tuple of card ids (cache key)."""
    ids = tuple(sorted(_CARD_ID[c] for c in cards_7))
    return ids


@lru_cache(maxsize=65536)
def evaluate_7_cached(card_ids):
    """Cached version — *card_ids* is a sorted tuple of 7 int card indices."""
    cards = [_CARDS_52[i] for i in card_ids]
    return _evaluate_7_uncached(cards)


def evaluate_7(cards_7):
    """Evaluate best 5-card hand from 7 cards (with LRU cache).

    Args:
        cards_7: iterable of 7 ``(rank_char, suit_char)`` tuples.

    Returns:
        ``(hand_rank, tiebreaker0, …)`` — comparable tuple.
    """
    key = _canonical_7(cards_7)
    return evaluate_7_cached(key)


# ── Comparison helper ─────────────────────────────────────────────────────

def compare_hands(cards_p0, cards_p1, community):
    """Compare two players' hands.

    Returns:
        > 0 if P0 wins,
        < 0 if P1 wins,
        = 0 if tie (split pot).
    """
    r0 = evaluate_7(list(cards_p0) + list(community))
    r1 = evaluate_7(list(cards_p1) + list(community))
    if r0 > r1:
        return 1
    elif r1 > r0:
        return -1
    return 0


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Royal flush vs four of a kind
    rf = [('A', 'h'), ('K', 'h'), ('Q', 'h'), ('J', 'h'), ('T', 'h'),
          ('2', 'd'), ('3', 'c')]
    fk = [('A', 's'), ('A', 'd'), ('A', 'c'), ('A', 'h'),
          ('K', 's'), ('2', 'h'), ('3', 's')]

    print("Royal Flush:", evaluate_7(rf))
    print("Four of a Kind:", evaluate_7(fk))
    print("Royal > Four:", evaluate_7(rf) > evaluate_7(fk))

    # Wheel straight (A-2-3-4-5)
    wheel = [('A', 'h'), ('2', 'd'), ('3', 'c'), ('4', 's'), ('5', 'h'),
             ('9', 'd'), ('T', 'c')]
    ace_high = [('A', 'h'), ('K', 'd'), ('Q', 'c'), ('J', 's'), ('T', 'h'),
                ('9', 'd'), ('8', 'c')]
    print("Wheel (5-high straight):", evaluate_7(wheel))
    print("Ace-high straight:", evaluate_7(ace_high))
    print("Ace-high > Wheel:", evaluate_7(ace_high) > evaluate_7(wheel))

    # Two pair tiebreaker
    two_pair_a = [('A', 'h'), ('A', 'd'), ('K', 'c'), ('K', 's'), ('Q', 'h'),
                  ('3', 'd'), ('2', 'c')]
    two_pair_b = [('A', 's'), ('A', 'c'), ('K', 'h'), ('K', 'd'), ('J', 's'),
                  ('4', 'h'), ('2', 'd')]
    print("Two Pair A (AAKKQ):", evaluate_7(two_pair_a))
    print("Two Pair B (AAKKJ):", evaluate_7(two_pair_b))
    print("A > B (Q kicker):", evaluate_7(two_pair_a) > evaluate_7(two_pair_b))

    # Cache stats
    info = evaluate_7_cached.cache_info()
    print(f"\nCache info: {info}")
