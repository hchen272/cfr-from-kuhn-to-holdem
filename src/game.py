import random


# Kuhn Poker deck
CARDS = ['J', 'Q', 'K']

# Available actions
PASS = 'p'
BET = 'b'

ACTIONS = [PASS, BET]


def deal_cards():
    """
    Shuffle and deal 1 card to each player.

    Returns:
        tuple: (player0_card, player1_card)
    """
    deck = CARDS.copy()
    random.shuffle(deck)

    return deck[0], deck[1]


def is_terminal(history):
    """
    Determine whether the game is over.

    Terminal histories in Kuhn Poker:

    pp     -> both players checked
    bp     -> bet + fold
    bb     -> bet + call
    pbp    -> check + bet + fold
    pbb    -> check + bet + call
    """

    if history in ['pp', 'bp', 'bb', 'pbp', 'pbb']:
        return True

    return False


def get_legal_actions(history):
    """
    Return legal actions for current state.

    Kuhn Poker only has:
        p = pass/check/fold
        b = bet/call
    """

    if is_terminal(history):
        return []

    return ACTIONS


def card_rank(card):
    """
    Convert card into numeric strength.
    """

    ranks = {
        'J': 0,
        'Q': 1,
        'K': 2
    }

    return ranks[card]


def get_payoff(history, cards):
    """
    Compute payoff from player0's perspective.

    Args:
        history (str):
            betting history

        cards (tuple):
            (player0_card, player1_card)

    Returns:
        int:
            positive -> player0 wins
            negative -> player0 loses
    """

    player0, player1 = cards

    terminal_pass = history in ['pp', 'bp', 'pbp']
    double_bet = history in ['bb', 'pbb']

    # showdown after checks
    if history == 'pp':

        if card_rank(player0) > card_rank(player1):
            return 1
        else:
            return -1

    # someone folded after bet
    if history == 'bp':
        return 1

    if history == 'pbp':
        return -1

    # called bet -> pot size = 2
    if double_bet:

        if card_rank(player0) > card_rank(player1):
            return 2
        else:
            return -2

    raise ValueError(f"Invalid terminal history: {history}")


if __name__ == "__main__":

    # Test environment

    cards = deal_cards()

    print("Cards:", cards)

    tests = ['pp', 'bp', 'bb', 'pbp', 'pbb']

    for history in tests:

        payoff = get_payoff(history, cards)

        print(f"History: {history}")
        print(f"Payoff: {payoff}")
        print("-" * 30)