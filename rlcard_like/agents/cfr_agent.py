"""CFR Agent — chance-sampling (vanilla) CFR, following rlcard's pattern.

Traverses the full game tree once per player per iteration.
Uses rlcard's standard regret-matching and linear averaging."""
import numpy as np
import collections
import os
import pickle

from rlcard_like.utils.utils import remove_illegal


class CFRAgent:
    def __init__(self, env, model_path='./cfr_model'):
        self.use_raw = False
        self.env = env
        self.model_path = model_path
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)
        self.regrets = collections.defaultdict(np.array)
        self.iteration = 0

    # ── training ──────────────────────────────────────────────────

    def train(self):
        self.iteration += 1
        for player_id in range(self.env.num_players):
            self.env.init_game()
            probs = np.ones(self.env.num_players)
            self.traverse_tree(probs, player_id)
        self.update_policy()

    def traverse_tree(self, probs, player_id):
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)
        obs, legal_actions = self._get_state(current_player)
        action_probs = self._action_probs(obs, legal_actions, self.policy)

        for action in legal_actions:
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob

            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if current_player != player_id:
            return state_utility

        # ── record regret and average strategy ──
        player_prob = probs[current_player]
        counterfactual_prob = (
            np.prod(probs[:current_player]) *
            np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]

        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)

        for action in legal_actions:
            action_prob = action_probs[action]
            regret = counterfactual_prob * (
                action_utilities[action][current_player] - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += (
                self.iteration * player_prob * action_prob)

        return state_utility

    # ── regret matching ───────────────────────────────────────────

    def update_policy(self):
        for obs in self.regrets:
            self.policy[obs] = self._regret_matching(obs)

    def _regret_matching(self, obs):
        regret = self.regrets[obs]
        pos_sum = sum(r for r in regret if r > 0)
        probs = np.zeros(self.env.num_actions)
        if pos_sum > 0:
            for a in range(self.env.num_actions):
                probs[a] = max(0.0, regret[a] / pos_sum)
        else:
            probs[:] = 1.0 / self.env.num_actions
        return probs

    def _action_probs(self, obs, legal_actions, policy):
        if obs not in policy:
            probs = np.ones(self.env.num_actions) / self.env.num_actions
            self.policy[obs] = probs
        else:
            probs = policy[obs]
        return remove_illegal(probs, legal_actions)

    # ── evaluation ────────────────────────────────────────────────

    def eval_step(self, state):
        obs = state['obs']
        legal_keys = list(state['legal_actions'].keys())
        probs = self._action_probs(obs, legal_keys, self.average_policy)
        action = np.random.choice(len(probs), p=probs)
        info = {'probs': {state['raw_legal_actions'][i]:
                          float(probs[legal_keys[i]])
                          for i in range(len(legal_keys))}}
        return action, info

    def _get_state(self, player_id):
        state = self.env.get_state(player_id)
        return state['obs'], list(state['legal_actions'].keys())

    # ── persist ───────────────────────────────────────────────────

    def save(self):
        os.makedirs(self.model_path, exist_ok=True)
        for name, obj in [('policy', self.policy),
                          ('average_policy', self.average_policy),
                          ('regrets', self.regrets),
                          ('iteration', self.iteration)]:
            with open(os.path.join(self.model_path, f'{name}.pkl'), 'wb') as f:
                pickle.dump(obj, f)

    def load(self):
        if not os.path.exists(self.model_path):
            return
        with open(os.path.join(self.model_path, 'policy.pkl'), 'rb') as f:
            self.policy = pickle.load(f)
        with open(os.path.join(self.model_path, 'average_policy.pkl'), 'rb') as f:
            self.average_policy = pickle.load(f)
        with open(os.path.join(self.model_path, 'regrets.pkl'), 'rb') as f:
            self.regrets = pickle.load(f)
        with open(os.path.join(self.model_path, 'iteration.pkl'), 'rb') as f:
            self.iteration = pickle.load(f)
