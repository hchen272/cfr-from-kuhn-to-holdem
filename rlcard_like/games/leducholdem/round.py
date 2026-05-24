class LimitHoldemRound:
    ''' Generic limit hold'em betting round. '''

    def __init__(self, raise_amount, allowed_raise_num, num_players, np_random):
        self.np_random = np_random
        self.game_pointer = None
        self.raise_amount = raise_amount
        self.allowed_raise_num = allowed_raise_num
        self.num_players = num_players
        self.have_raised = 0
        self.not_raise_num = 0
        self.raised = [0 for _ in range(num_players)]
        self.player_folded = None

    def start_new_round(self, game_pointer, raised=None):
        self.game_pointer = game_pointer
        self.have_raised = 0
        self.not_raise_num = 0
        self.raised = raised if raised else [0] * self.num_players

    def proceed_round(self, players, action):
        if action == 'call':
            diff = max(self.raised) - self.raised[self.game_pointer]
            self.raised[self.game_pointer] = max(self.raised)
            players[self.game_pointer].in_chips += diff
            self.not_raise_num += 1
        elif action == 'raise':
            diff = max(self.raised) - self.raised[self.game_pointer] + self.raise_amount
            self.raised[self.game_pointer] = max(self.raised) + self.raise_amount
            players[self.game_pointer].in_chips += diff
            self.have_raised += 1
            self.not_raise_num = 1
        elif action == 'fold':
            players[self.game_pointer].status = 'folded'
            self.player_folded = True
        elif action == 'check':
            self.not_raise_num += 1
        self.game_pointer = (self.game_pointer + 1) % self.num_players
        while players[self.game_pointer].status == 'folded':
            self.game_pointer = (self.game_pointer + 1) % self.num_players
        return self.game_pointer

    def get_legal_actions(self):
        full = ['call', 'raise', 'fold', 'check']
        if self.have_raised >= self.allowed_raise_num:
            full.remove('raise')
        if self.raised[self.game_pointer] < max(self.raised):
            full.remove('check')
        if self.raised[self.game_pointer] == max(self.raised):
            full.remove('call')
        return full

    def is_over(self):
        return self.not_raise_num >= self.num_players


class LeducholdemRound(LimitHoldemRound):
    ''' Leduc-specific round: 2->4 raise amount in R2. '''
    pass
