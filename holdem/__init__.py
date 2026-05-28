"""
holdem/ — Texas Hold'em module.

Independent from src/; uses on-the-fly game-tree traversal instead of
pre-computed BFS.  Shares only ``src.algo.tabular.node.Node`` and
``src.utils`` for model persistence.
"""
