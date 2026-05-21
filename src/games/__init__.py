"""
Game abstraction layer.

Defines the ``Game`` abstract base class that all poker games must implement.
Each game provides a uniform interface for:
  - Game metadata (name, num_actions, feature_dim)
  - State queries (is_terminal, get_legal_actions, get_payoff, card_rank)
  - Deal / action constants (deal_cards, ACTIONS, CARDS)
"""

from abc import ABC, abstractmethod


class Game(ABC):
    """Abstract base class for poker games used by CFR variants."""

    # ── Class-level constants (also accessible as instance attributes) ──
    CARDS: list = []
    ACTIONS: list = []

    # ── Metadata ────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short game identifier, e.g. ``'kuhn'`` or ``'leduc'``."""

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Number of legal actions at each decision point."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of the feature vector for neural-network input."""

    @property
    def nash_value(self) -> float:
        """Known Nash equilibrium game value for P0 (or None if unknown)."""
        return None

    # ── State queries ──────────────────────────────────────────────────

    @abstractmethod
    def is_terminal(self, history: str) -> bool:
        """Return ``True`` if *history* represents a terminal state."""

    @abstractmethod
    def get_payoff(self, history: str, cards: tuple) -> float:
        """Return payoff from player-0's perspective."""

    @abstractmethod
    def get_legal_actions(self, history: str) -> list:
        """Return list of legal actions at *history*."""

    @abstractmethod
    def card_rank(self, card: str) -> int:
        """Return numeric rank of *card* (higher = stronger)."""

    # ── Deal / setup ───────────────────────────────────────────────────

    @abstractmethod
    def deal_cards(self) -> tuple:
        """Shuffle and deal cards; return ``(player0_card, player1_card)``."""

    def build_next_history(self, history: str, action: str) -> str:
        """
        Append *action* to *history*, handling round transitions.

        The default implementation concatenates directly (suitable for
        single-round games like Kuhn Poker). Multi-round games override
        this to insert community-card or round-separator information.
        """
        return history + action

    # ── Feature encoding (for Deep CFR) ────────────────────────────────

    @abstractmethod
    def infoset_to_features(self, infoset: str):
        """Convert an infoset string to a fixed-size numeric feature vector."""
