from rlcard_like.games.base import Card

class LeducholdemDealer:
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = [Card('S', 'J'), Card('H', 'J'),
                     Card('S', 'Q'), Card('H', 'Q'),
                     Card('S', 'K'), Card('H', 'K')]
        self.shuffle()
        self.pot = 0

    def shuffle(self):
        self.np_random.shuffle(self.deck)

    def deal_card(self):
        return self.deck.pop()
