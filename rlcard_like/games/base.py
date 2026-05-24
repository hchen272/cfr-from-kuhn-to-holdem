class Card:
    ''' Card stores suit and rank. '''
    valid_suit = ['S', 'H', 'D', 'C']
    valid_rank = ['J', 'Q', 'K']

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return NotImplemented

    def __hash__(self):
        return Card.valid_rank.index(self.rank) + 100 * Card.valid_suit.index(self.suit)

    def __str__(self):
        return self.rank + self.suit

    def get_index(self):
        return self.suit + self.rank
