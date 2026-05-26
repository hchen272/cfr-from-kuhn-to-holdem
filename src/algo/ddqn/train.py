"""Double DQN training for poker games.

Differs from DQN by decoupling action selection and evaluation:
    target = r + γ * Q_target(s', argmax_a Q(s', a))
This reduces the over-estimation bias of standard DQN."""
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_selector import get_game
from algo.dqn.model import QNetwork, _hidden_dim
from utils import save_strategy_txt, save_model


class _FakeNode:
    """Duck-typed Node for save_strategy_txt compatibility."""
    def __init__(self, avg):
        self._avg = avg
    def get_average_strategy(self):
        return self._avg


# ═══════════════════════════════════════════════════════════════════
#  Replay buffer
# ═══════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        entry = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.pos] = entry
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, r, ns, d = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a),
                np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════
#  Hyper-parameters
# ═══════════════════════════════════════════════════════════════════

BATCH_SIZE       = 256
BUFFER_CAPACITY  = 100_000
LEARNING_RATE    = 0.0001
GAMMA            = 0.99
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY        = 50_000           # steps over which ε decays
TARGET_UPDATE    = 1_000            # update target network every N steps
TRAIN_START      = 1_000            # start training after buffer has N entries


# ═══════════════════════════════════════════════════════════════════
#  Double DQN Agent
# ═══════════════════════════════════════════════════════════════════

class DDQNAgent:
    def __init__(self, game, device=None):
        self.game = game
        self.device = device or torch.device("cpu")
        hdim = _hidden_dim(game.name)

        self.q_net = QNetwork(game.feature_dim, hdim, game.num_actions).to(self.device)
        self.target_net = QNetwork(game.feature_dim, hdim, game.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(BUFFER_CAPACITY)
        self.steps = 0

    def _features(self, card, history):
        infoset = card + history
        return np.array(self.game.infoset_to_features(infoset), dtype=np.float32)

    def select_action(self, card, history, legal_actions, epsilon):
        if random.random() < epsilon:
            return random.choice(legal_actions)
        feats = torch.from_numpy(self._features(card, history)).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(feats.unsqueeze(0))[0].cpu().numpy()
        # pick best among legal actions
        best = max(legal_actions, key=lambda a: q_vals[a])
        return best

    def train_step(self):
        if len(self.buffer) < TRAIN_START:
            return
        s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)

        s_t  = torch.from_numpy(s).to(self.device)
        ns_t = torch.from_numpy(ns).to(self.device)
        r_t  = torch.from_numpy(r).to(self.device)
        d_t  = torch.from_numpy(d).to(self.device)

        q_vals = self.q_net(s_t)
        q_pred = q_vals.gather(1, torch.from_numpy(a).unsqueeze(1).to(self.device)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online net selects, target net evaluates
            best_actions = self.q_net(ns_t).argmax(1, keepdim=True)
            next_q = self.target_net(ns_t).gather(1, best_actions).squeeze(1)
            target = r_t + GAMMA * next_q * (1 - d_t)

        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════

def train_ddqn(iterations, game_name="kuhn", log_prefix="ddqn"):
    game = get_game(game_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDQNAgent(game, device)

    print(f"Double DQN  |  game: {game.name}  |  episodes: {iterations:,}  |  device: {device}")

    # Enumerate infosets from game tree nodes × card ranks
    from algo.tabular.game_tree import GameTree
    tree = GameTree(game)
    infoset_keys = []
    for hid, node in tree.nodes.items():
        if node.is_terminal:
            continue
        hist = tree.history_str(hid)
        for rank in ['J', 'Q', 'K']:
            infoset_keys.append(rank + hist)

    def _strategy_map():
        """Build {infoset_str: FakeNode} from Q-network (softmax over Q)."""
        node_map = {}
        for key in infoset_keys:
            card = key[0]
            hist = key[1:]
            try:
                feats = np.array(game.infoset_to_features(card + hist), dtype=np.float32)
            except Exception:
                continue
            with torch.no_grad():
                q = agent.q_net(torch.from_numpy(feats).unsqueeze(0).to(device))[0].cpu().numpy()
            probs = np.exp(q - q.max())
            probs /= probs.sum()
            node_map[key] = _FakeNode(probs)
        return node_map

    # Log header
    save_strategy_txt(_strategy_map(), 0, 0, iterations, log_prefix, game_name=game_name)

    total_reward = 0.0
    best_reward = -float("inf")
    best_net_state = None
    snapshot_every = max(1, iterations // 100)

    for episode in range(1, iterations + 1):
        cards = game.deal_cards()
        history = ""
        ep_reward = 0.0

        while not game.is_terminal(history):
            player = len(history) % 2
            legal = game.get_legal_actions(history)
            legal_ids = [game.ACTIONS.index(a) for a in legal]

            epsilon = EPS_END + (EPS_START - EPS_END) * \
                      max(0.0, 1.0 - agent.steps / EPS_DECAY)

            if player == 0:
                action_id = agent.select_action(cards[0], history, legal_ids, epsilon)
                next_hist = game.build_next_history(history, game.ACTIONS[action_id])

                if game.is_terminal(next_hist):
                    reward = game.get_payoff(next_hist, cards)
                    done = 1.0
                else:
                    reward = 0.0
                    done = 0.0

                feats = agent._features(cards[0], history)
                next_feats = agent._features(cards[0], next_hist) if not done else np.zeros_like(feats)
                agent.buffer.add(feats, action_id, reward, next_feats, done)
                agent.train_step()
                ep_reward += reward
            else:
                # opponent: random play
                action_id = random.choice(legal_ids)

            history = game.build_next_history(history, game.ACTIONS[action_id])

        total_reward += ep_reward

        if episode % snapshot_every == 0:
            avg = total_reward / episode
            print(f"  episode {episode:>8,}  |  avg reward: {avg:+.4f}")
            save_strategy_txt(_strategy_map(), episode, avg, iterations,
                              log_prefix, game_name=game_name)

            if avg > best_reward:
                best_reward = avg
                best_net_state = {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()}

    # Restore best
    if best_net_state is not None:
        agent.q_net.load_state_dict(best_net_state)
        print(f"\n[CHECKPOINT] Restored best (avg reward = {best_reward:+.4f})")

    save_model({"q_net": best_net_state}, iterations, log_prefix, game_name=game_name)
    print(f"Average reward: {total_reward / iterations:+.4f}")
