"""
Pre-computed game tree with integer encoding.

Builds a complete enumeration of all reachable histories for a poker game,
assigning integer IDs.  During training, CFR uses integer lookups instead of
string parsing.  Log output is converted back to human-readable strings.

Handles games with community cards (Leduc) by forking at the R1→R2 transition.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Any, Union


@dataclass
class TreeNode:
    """Pre-computed information about a single history node."""
    hid: int
    history_str: str
    player: int
    is_terminal: bool
    legal_actions: list = field(default_factory=list)
    # child_hids[action_idx] → child_hid  (int), or [hid_J, hid_K, hid_Q] (list)
    child_hids: dict = field(default_factory=dict)

    def child_for(self, action_idx: int, comm_rank: str = "") -> Union[int, None]:
        """Return child_hid, handling community-card forks."""
        v = self.child_hids.get(action_idx)
        if v is None:
            return None
        if isinstance(v, int):
            return v
        idx = {'J': 0, 'Q': 1, 'K': 2}.get(comm_rank, 0)
        return v[idx] if idx < len(v) else v[0]


class GameTree:
    """Pre-computed game tree for fast tabular / deep CFR."""

    def __init__(self, game: Any):
        self.game = game
        self.num_actions = game.num_actions

        # history_id → TreeNode
        self.nodes: dict[int, TreeNode] = {}

        # string ↔ id
        self._hist_to_id: dict[str, int] = {}
        self._id_to_hist: dict[int, str] = {}

        # payoff cache: (hid, p0r, p1r) → float
        self._payoff_cache: dict[tuple, float] = {}

        # infoset mapping: (card_char, hid) → iid
        self._iid_cache: dict[tuple, int] = {}
        self._id_to_infoset: dict[int, str] = {}
        self._next_iid: int = 0

        # community rank per hid
        self._comm_of: dict[int, str] = {}

        # action char ↔ index
        self._a2i = {a: i for i, a in enumerate(game.ACTIONS)}
        self._i2a = {i: a for i, a in enumerate(game.ACTIONS)}

        self._has_comm = hasattr(game, '_community_rank')
        self._comm_ranks = ['J', 'Q', 'K'] if self._has_comm else []
        
        # Ensure _comm exists (LeducGame needs it for tree build)
        if self._has_comm and not hasattr(game, '_comm'):
            game._comm = ('K', 0)

        self._build()

    # ── public helpers ──────────────────────────────────────────────

    def get_payoff(self, hid: int, cards: tuple) -> float:
        return self._payoff_cache.get((hid, cards[0], cards[1]), 0.0)

    def infoset_id(self, card: str, hid: int) -> int:
        key = (card, hid)
        if key in self._iid_cache:
            return self._iid_cache[key]
        iid = self._next_iid
        self._next_iid += 1
        self._iid_cache[key] = iid
        self._id_to_infoset[iid] = f"{card}{self._id_to_hist[hid]}"
        return iid

    def infoset_str(self, iid: int) -> str:
        return self._id_to_infoset.get(iid, f"?{iid}")

    def history_str(self, hid: int) -> str:
        return self._id_to_hist.get(hid, "")

    def comm_rank_of(self, hid: int) -> str:
        return self._comm_of.get(hid, "")

    @property
    def num_infosets(self) -> int:
        return len(self._iid_cache)

    # ── build ───────────────────────────────────────────────────────

    def _build(self):
        """BFS, forking for community cards at R1→R2 boundary."""
        # ── first pass: assign IDs for all histories ───
        queue = deque()
        queue.append(("", ""))  # (history_str, comm_rank)
        visited: set[str] = set()
        next_id = 0

        while queue:
            hist, comm = queue.popleft()
            if hist in visited:
                continue
            visited.add(hist)

            hid = next_id
            self._hist_to_id[hist] = hid
            self._id_to_hist[hid] = hist
            self._comm_of[hid] = comm
            next_id += 1

            legal_strs = self.game.get_legal_actions(hist)
            player = len(hist) % 2
            is_term = self.game.is_terminal(hist)

            # enqueue children (strings only at this stage)
            for a_str in legal_strs:
                if self._has_comm:
                    sep_was = '|' in hist
                    if not sep_was:
                        # might be R1→R2 transition — fork
                        for cr in self._comm_ranks:
                            self.game._comm = (cr, 0)
                            child = self.game.build_next_history(hist, a_str)
                            child_comm = ''
                            if '|' in child:
                                parts = child.split('|')
                                if len(parts) > 1 and parts[1]:
                                    child_comm = parts[1][0]
                            if child not in visited:
                                queue.append((child, child_comm))
                    else:
                        self.game._comm = (comm, 0) if comm else ('K', 0)
                        child = self.game.build_next_history(hist, a_str)
                        if child not in visited:
                            queue.append((child, comm))
                else:
                    child = self.game.build_next_history(hist, a_str)
                    if child not in visited:
                        queue.append((child, ""))

            self.nodes[hid] = TreeNode(
                hid=hid, history_str=hist, player=player,
                is_terminal=is_term,
                legal_actions=[self._a2i[a] for a in legal_strs],
                child_hids={},
            )

        # ── second pass: resolve child_hids (str → int) ───
        for hid, node in self.nodes.items():
            hist = self._id_to_hist[hid]
            sep_was = '|' in hist
            for a_str in self.game.get_legal_actions(hist):
                ai = self._a2i[a_str]
                if self._has_comm and not sep_was:
                    # R1→R2 forking: store list [hid_J, hid_K, hid_Q]
                    children = []
                    for cr in self._comm_ranks:
                        self.game._comm = (cr, 0)
                        child = self.game.build_next_history(hist, a_str)
                        children.append(self._hist_to_id.get(child, -1))
                    node.child_hids[ai] = children
                else:
                    # need comm context for R2 histories
                    comm = self._comm_of.get(hid, '')
                    if self._has_comm:
                        self.game._comm = (comm, 0) if comm else ('K', 0)
                    child = self.game.build_next_history(hist, a_str)
                    node.child_hids[ai] = self._hist_to_id.get(child, -1)

        # ── pre-compute payoffs ───
        ranks = ["J", "Q", "K"]
        for hid, node in self.nodes.items():
            if not node.is_terminal:
                continue
            for p0r in ranks:
                for p1r in ranks:
                    self._payoff_cache[(hid, p0r, p1r)] = \
                        self.game.get_payoff(node.history_str, (p0r, p1r))

    def __repr__(self) -> str:
        return (f"GameTree(histories={len(self.nodes)}, "
                f"infosets={self.num_infosets})")
