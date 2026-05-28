"""
Game selector — returns a ``Game`` instance by name.

Usage::

    from game_selector import get_game

    game = get_game("kuhn")          # → KuhnGame singleton
    game = get_game("leduc")         # → LeducGame (when implemented)
"""

_GAME_CACHE = {}


def get_game(game_name: str):
    """Return a cached ``Game`` instance for the given *game_name*.

    Parameters
    ----------
    game_name : str
        One of ``'kuhn'``, ``'leduc'`` (when implemented).

    Returns
    -------
    Game
        A singleton game instance.
    """
    if game_name in _GAME_CACHE:
        return _GAME_CACHE[game_name]

    if game_name == "kuhn":
        from games.kuhn import KuhnGame
        game = KuhnGame()
    elif game_name == "leduc":
        from games.leduc import LeducGame
        game = LeducGame()
    elif game_name == "expanded_leduc":
        from games.expanded_leduc import ExpandedLeducGame
        game = ExpandedLeducGame()
    elif game_name == "river_poker":
        from games.river_poker import RiverPokerGame
        game = RiverPokerGame()
    else:
        raise ValueError(f"Unknown game: {game_name!r}. "
                         f"Available: kuhn, leduc, expanded_leduc, river_poker")

    _GAME_CACHE[game_name] = game
    return game
