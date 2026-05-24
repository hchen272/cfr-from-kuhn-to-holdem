import numpy as np
from copy import copy

from rlcard_like.games.leducholdem.dealer import LeducholdemDealer
from rlcard_like.games.leducholdem.player import LeducholdemPlayer
from rlcard_like.games.leducholdem.judger import LeducholdemJudger
from rlcard_like.games.leducholdem.round import LeducholdemRound


class LeducholdemGame:
    ''' Leduc Hold'em — two-player, two-round, fixed-limit poker.

    Card deck: J♠ J♥ Q♠ Q♥ K♠ K♥  (6 cards, 3 ranks × 2 suits).
    Each player antes 1 chip (small blind / big blind).
    One community card is dealt after round 1.
    Bet sizes: 2 chips (R1), 4 chips (R2).  Max 2 raises per round.
    '''

    NUM_PLAYERS = 2
    NUM_ACTIONS = 3            # call/check, raise, fold
    NASH_VALUE = -0.0855       # P0 expected value at Nash

    def __init__(self, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()

        self.small_blind = 1
        self.big_blind = 2

        self.raise_amount = self.big_blind       # 2
        self.allowed_raise_num = 2

        self.num_players = self.NUM_PLAYERS
        self.num_actions = self.NUM_ACTIONS

    # ── init / step ───────────────────────────────────────────────

    def init_game(self):
        self.dealer = LeducholdemDealer(self.np_random)
        self.players = [LeducholdemPlayer(i, self.np_random)
                        for i in range(self.num_players)]
        self.judger = LeducholdemJudger(self.np_random)

        for i in range(self.num_players):
            self.players[i].hand = self.dealer.deal_card()

        s = self.np_random.randint(0, self.num_players)
        b = (s + 1) % self.num_players
        self.players[b].in_chips = self.big_blind
        self.players[s].in_chips = self.small_blind
        self.public_card = None
        self.game_pointer = s

        self.round = LeducholdemRound(
            raise_amount=self.raise_amount,
            allowed_raise_num=self.allowed_raise_num,
            num_players=self.num_players,
            np_random=self.np_random)
        self.round.start_new_round(
            game_pointer=self.game_pointer,
            raised=[p.in_chips for p in self.players])
        self.round_counter = 0
        self.history = []

        return self.get_state(self.game_pointer), self.game_pointer

    def step(self, action):
        # accept int (from CFR agent) or str
        if isinstance(action, int):
            action = ['call', 'raise', 'fold'][action]
        if self.allow_step_back:
            snap = (copy(self.round), copy(self.round.raised),
                    self.game_pointer, self.round_counter,
                    copy(self.dealer.deck), copy(self.public_card),
                    [copy(p) for p in self.players],
                    [copy(p.hand) for p in self.players])
            self.history.append(snap)

        self.game_pointer = self.round.proceed_round(self.players, action)

        if self.round.is_over():
            if self.round_counter == 0:
                self.public_card = self.dealer.deal_card()
                self.round.raise_amount = 2 * self.raise_amount  # 2 → 4
            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        return self.get_state(self.game_pointer), self.game_pointer

    def step_back(self):
        if not self.history:
            return False
        (self.round, r_raised, self.game_pointer, self.round_counter,
         d_deck, self.public_card, self.players, ps_hand) = self.history.pop()
        self.round.raised = r_raised
        self.dealer.deck = d_deck
        for i, h in enumerate(ps_hand):
            self.players[i].hand = h
        return True

    # ── queries ───────────────────────────────────────────────────

    def get_player_id(self):
        return self.game_pointer

    def is_over(self):
        alive = sum(1 for p in self.players if p.status == 'alive')
        if alive == 1:
            return True
        return self.round_counter >= 2

    def get_payoffs(self):
        chips = self.judger.judge_game(self.players, self.public_card)
        return np.array(chips) / self.big_blind

    def get_legal_actions(self):
        return self.round.get_legal_actions()

    def get_state(self, player_id):
        chips = [p.in_chips for p in self.players]
        legal_actions = self.get_legal_actions()
        state = self.players[player_id].get_state(self.public_card, chips, legal_actions)
        state['current_player'] = self.game_pointer
        # Build obs string like rlcard: hand|public|history
        hand = state['hand'] or '--'
        pub  = state['public_card'] or '--'
        state['obs'] = f"{hand}|{pub}"
        state['raw_legal_actions'] = legal_actions
        state['legal_actions'] = {self._action2id[a]: None for a in legal_actions}
        return state

    # ── helpers ───────────────────────────────────────────────────

    @property
    def _action2id(self):
        return {'call': 0, 'raise': 1, 'fold': 2, 'check': 0}
