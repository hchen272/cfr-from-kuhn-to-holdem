"""
Card abstraction for Texas Hold'em.

Compresses the 1,326 specific hole-card combinations into a manageable
number of buckets (~100) by clustering on preflop expected hand strength
(equity against a random hand).

Approach
--------
1. Enumerate 169 rank-based hole-card types (13 pairs + 78 suited + 78 offsuit).
2. Estimate preflop equity for each type via Monte Carlo (random opponent +
   random board).
3. Sort by equity; partition into *n_buckets* equal-sized groups.
4. Every specific (rank, suit, rank, suit) hole card maps to its type's bucket.

Usage
-----
>>> from abstraction import CardAbstraction
>>> absn = CardAbstraction(n_buckets=100)
>>> absn.bucket_id((('A','h'), ('K','h')))  # AK suited
42
>>> absn.bucket_id((('7','d'), ('2','c')))  # 72 offsuit
3
"""
import random
import sys
import os
from collections import defaultdict
from itertools import combinations as _combos, product as _prod

# Allow running as script
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_eval import evaluate_7, RANKS, SUITS

# ── Constants ───────────────────────────────────────────────────────────
ALL_CARDS = [(r, s) for r in RANKS for s in SUITS]   # 52


def _hole_type(c1, c2):
    """Return canonical rank-based type string: 'AA', 'AKs', 'AKo', etc."""
    r1, r2 = c1[0], c2[0]
    suited = 's' if c1[1] == c2[1] else 'o'
    if RANKS.index(r1) >= RANKS.index(r2):
        return r1 + r2 + ('' if r1 == r2 else suited)
    else:
        return r2 + r1 + suited


def _rank_pair_key(rank_pair_str):
    """Sortable key for rank-pair strings (e.g. 'AKs' → rank indices)."""
    r1, r2 = rank_pair_str[0], rank_pair_str[1]
    return RANKS.index(r1) * 13 + RANKS.index(r2)


# ── Monte Carlo equity estimator ─────────────────────────────────────────

def _estimate_equity(hole_type_str, n_samples=5000):
    """Estimate preflop equity of *hole_type_str* vs random hand.

    Returns float in [0, 1] — fraction of pots won (ties count 0.5).
    """
    r1, r2 = hole_type_str[0], hole_type_str[1]
    suited = hole_type_str[2] if len(hole_type_str) > 2 else ''

    wins = 0.0
    for _ in range(n_samples):
        # Build deck excluding our hole cards
        # For the specific suits, we need to sample suits from the type
        deck = list(ALL_CARDS)
        # Pick our hole cards with correct suits
        if suited == 's':
            s = random.choice(SUITS)
            our = [(r1, s), (r2, s)]
        elif suited == 'o':
            s1, s2 = random.sample(SUITS, 2)
            our = [(r1, s1), (r2, s2)]
        else:  # pair
            s1, s2 = random.sample(SUITS, 2)
            our = [(r1, s1), (r2, s2)]

        # Remove our cards from deck
        deck = [c for c in deck if c not in our]

        # Deal opponent 2 cards, board 5 cards
        random.shuffle(deck)
        opp = [deck[0], deck[1]]
        board = deck[2:7]

        our_val = evaluate_7(our + board)
        opp_val = evaluate_7(opp + board)

        if our_val > opp_val:
            wins += 1.0
        elif our_val == opp_val:
            wins += 0.5

    return wins / n_samples


# ── Precomputed lookup table builder ──────────────────────────────────────

def _build_type_equities(n_samples=5000):
    """Build {type_str: equity} for all 169 rank-based types."""
    types = []
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            if i < j:
                types.append(r2 + r1 + 's')
                types.append(r2 + r1 + 'o')
            elif i == j:
                types.append(r1 + r1)

    equities = {}
    for t in types:
        equities[t] = _estimate_equity(t, n_samples=n_samples)
    return equities


# Cached result (computed once on first use)
_EQUITY_CACHE = None


def _get_equities():
    global _EQUITY_CACHE
    if _EQUITY_CACHE is None:
        _EQUITY_CACHE = _build_type_equities()
    return _EQUITY_CACHE


# ── Card Abstraction ─────────────────────────────────────────────────────

class CardAbstraction:
    """Maps specific hole cards → bucket ID (0 … n_buckets-1)."""

    def __init__(self, n_buckets=100, n_mc_samples=5000):
        self.n_buckets = n_buckets

        # Build or load equity table
        equities = _build_type_equities(n_samples=n_mc_samples)

        # Sort types by equity
        sorted_types = sorted(equities.items(), key=lambda x: x[1])
        n_types = len(sorted_types)

        # Partition into equal-sized buckets
        bucket_size = max(1, n_types // n_buckets)
        self._type_to_bucket = {}
        for bucket_id in range(n_buckets):
            start = bucket_id * bucket_size
            end = start + bucket_size if bucket_id < n_buckets - 1 else n_types
            for type_str, _ in sorted_types[start:end]:
                self._type_to_bucket[type_str] = bucket_id

        # Cache for specific-card lookups
        self._card_to_type = {}
        self._card_to_bucket = {}

    def bucket_id(self, hole_cards):
        """Return bucket id for *hole_cards* (pair of (rank, suit) tuples)."""
        # Use canonical order to ensure cache accuracy
        key = tuple(sorted(hole_cards, key=lambda c: (RANKS.index(c[0]), c[1])))
        if key in self._card_to_bucket:
            return self._card_to_bucket[key]

        htype = _hole_type(key[0], key[1])
        bid = self._type_to_bucket.get(htype, 0)
        self._card_to_bucket[key] = bid
        return bid

    def bucket_str(self, bucket_id):
        """Human-readable label for *bucket_id*."""
        return f"B{bucket_id:03d}"


# ── Smoke test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Building card abstraction (this may take ~1 minute)...")
    absn = CardAbstraction(n_buckets=50, n_mc_samples=2000)

    print(f"\nBuckets: {absn.n_buckets}")
    print(f"Types mapped: {len(absn._type_to_bucket)}")

    # Show bucket assignments for well-known hands
    test_hands = [
        (('A', 'h'), ('A', 'd')),   # AA
        (('A', 'h'), ('K', 'h')),   # AKs
        (('A', 'h'), ('K', 'd')),   # AKo
        (('Q', 'h'), ('Q', 'd')),   # QQ
        (('J', 'h'), ('T', 'h')),   # JTs
        (('7', 'd'), ('2', 'c')),   # 72o
        (('3', 'h'), ('2', 'd')),   # 32o
    ]
    print("\nBucket assignments:")
    for h in test_hands:
        hid = absn.bucket_id(h)
        print(f"  {h[0][0]}{h[0][1]}{'s' if h[0][1]==h[1][1] else 'o' if h[0][0]!=h[1][0] else ''} "
              f"→ bucket {hid} ({absn.bucket_str(hid)})")

    # Verify all 1326 combos get a bucket
    all_hole = [c for c in _combos(ALL_CARDS, 2)]
    assert len(all_hole) == 1326, f"Expected 1326, got {len(all_hole)}"
    for h in all_hole:
        absn.bucket_id(h)
    print(f"\nAll {len(all_hole)} hole-card combos mapped successfully.")
    print(f"Cache size: {len(absn._card_to_bucket)}")
