class LeducholdemPlayer:
    def __init__(self, player_id, np_random):
        self.np_random = np_random
        self.player_id = player_id
        self.status = 'alive'
        self.hand = None
        self.in_chips = 0

    def get_state(self, public_card, all_chips, legal_actions):
        state = {}
        state['hand'] = self.hand.get_index() if self.hand else None
        state['public_card'] = public_card.get_index() if public_card else None
        state['all_chips'] = all_chips
        state['my_chips'] = self.in_chips
        state['legal_actions'] = legal_actions
        return state
