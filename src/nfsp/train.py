"""Neural Fictitious Self-Play (NFSP).

Heinrich & Silver (2016):
  - Best-response agent:  DQN (ε-greedy)
  - Average-policy agent: supervised network trained on past behaviour
  - Anticipatory parameter η: probability of using best-response vs average
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_selector import get_game
from dqn.model import QNetwork, _hidden_dim
from utils import save_strategy_txt, save_model


class _FakeNode:
    def __init__(self, avg): self._avg = avg
    def get_average_strategy(self): return self._avg


# ═══════════════════════════════════════════════════════════════════
#  Policy network (for average strategy)
# ═══════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    """MLP: feature_dim → hidden → hidden → num_actions → softmax (in forward)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.out(h)   # raw logits; softmax applied by caller


# ═══════════════════════════════════════════════════════════════════
#  Reservoir buffer for supervised learning (stores (state, action))
# ═══════════════════════════════════════════════════════════════════

class ReservoirBuffer:
    """Fixed-capacity buffer with reservoir sampling."""

    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0

    def add(self, state, action):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action))
        else:
            idx = random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = (state, action)
        self.n_seen += 1

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, n)
        s = np.array([b[0] for b in batch], dtype=np.float32)
        a = np.array([b[1] for b in batch], dtype=np.int64)
        return s, a

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════
#  Hyper-parameters
# ═══════════════════════════════════════════════════════════════════

RL_BATCH_SIZE    = 256            # DQN mini-batch
SL_BATCH_SIZE    = 256            # policy network mini-batch
RL_BUFFER_SIZE   = 100_000        # DQN replay buffer
SL_BUFFER_SIZE   = 200_000        # reservoir buffer for avg policy
RL_LR            = 0.0001         # DQN learning rate
SL_LR            = 0.001          # policy net learning rate
GAMMA            = 0.99
EPS_START        = 0.06
EPS_END          = 0.06
TARGET_UPDATE    = 1_000
ETA              = 0.1            # anticipatory parameter (fraction of BR)
RL_TRAIN_EVERY   = 1              # train DQN every N steps
SL_TRAIN_EVERY   = 1              # train policy net every N steps
TRAIN_START      = 1_000          # start training after buffer fills


# ═══════════════════════════════════════════════════════════════════
#  NFSP Agent
# ═══════════════════════════════════════════════════════════════════

class NFSPAgent:
    def __init__(self, game, device=None):
        self.game = game
        self.device = device or torch.device("cpu")
        hdim = _hidden_dim(game.name)
        na = game.num_actions

        # ── DQN (best response) ──
        self.q_net = QNetwork(game.feature_dim, hdim, na).to(self.device)
        self.target_net = QNetwork(game.feature_dim, hdim, na).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.rl_optim = optim.Adam(self.q_net.parameters(), lr=RL_LR)
        self.rl_loss = nn.MSELoss()
        self.rl_buffer = []           # simple growing buffer for DQN
        self.rl_buf_pos = 0

        # ── SL (average policy) ──
        self.policy_net = PolicyNetwork(game.feature_dim, hdim, na).to(self.device)
        self.sl_optim = optim.Adam(self.policy_net.parameters(), lr=SL_LR)
        self.sl_loss = nn.CrossEntropyLoss()
        self.sl_buffer = ReservoirBuffer(SL_BUFFER_SIZE)

        self.steps = 0

    def _features(self, card, history):
        infoset = card + history
        return np.array(self.game.infoset_to_features(infoset), dtype=np.float32)

    # ── action selection ──────────────────────────────────────

    def select_action(self, card, history, legal_ids):
        """ε-greedy over Q-values (for DQN)."""
        if random.random() < EPS_START:
            return random.choice(legal_ids)
        feats = torch.from_numpy(self._features(card, history)).to(self.device)
        with torch.no_grad():
            q = self.q_net(feats.unsqueeze(0))[0].cpu().numpy()
        return max(legal_ids, key=lambda a: q[a])

    def select_action_avg(self, card, history, legal_ids):
        """Sample from average-policy network."""
        feats = torch.from_numpy(self._features(card, history)).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(feats.unsqueeze(0))[0].cpu().numpy()
        # mask illegal actions
        mask = np.full(self.game.num_actions, -1e9)
        for a in legal_ids:
            mask[a] = logits[a]
        probs = np.exp(mask - mask.max())
        probs /= probs.sum()
        return np.random.choice(self.game.num_actions, p=probs)

    def act(self, card, history, legal_ids):
        """NFSP dual-mode action: η → best response, 1-η → average."""
        if random.random() < ETA:
            return self.select_action(card, history, legal_ids), "rl"
        else:
            return self.select_action_avg(card, history, legal_ids), "sl"

    # ── training ──────────────────────────────────────────────

    def rl_add(self, s, a, r, ns, d):
        entry = (s, a, r, ns, d)
        if len(self.rl_buffer) < RL_BUFFER_SIZE:
            self.rl_buffer.append(entry)
        else:
            self.rl_buffer[self.rl_buf_pos] = entry
        self.rl_buf_pos = (self.rl_buf_pos + 1) % RL_BUFFER_SIZE

    def rl_train_step(self):
        if len(self.rl_buffer) < TRAIN_START:
            return
        batch = random.sample(self.rl_buffer, min(RL_BATCH_SIZE, len(self.rl_buffer)))
        s, a, r, ns, d = zip(*batch)
        s_t  = torch.from_numpy(np.array(s, dtype=np.float32)).to(self.device)
        a_t  = torch.tensor(a).unsqueeze(1).to(self.device)
        r_t  = torch.tensor(r, dtype=torch.float32).to(self.device)
        ns_t = torch.from_numpy(np.array(ns, dtype=np.float32)).to(self.device)
        d_t  = torch.tensor(d, dtype=torch.float32).to(self.device)

        q_pred = self.q_net(s_t).gather(1, a_t).squeeze(1)
        with torch.no_grad():
            target = r_t + GAMMA * self.target_net(ns_t).max(1)[0] * (1 - d_t)
        loss = self.rl_loss(q_pred, target)
        self.rl_optim.zero_grad()
        loss.backward()
        self.rl_optim.step()

        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def sl_train_step(self):
        if len(self.sl_buffer) < TRAIN_START // 10:
            return
        s, a = self.sl_buffer.sample(SL_BATCH_SIZE)
        s_t = torch.from_numpy(s).to(self.device)
        a_t = torch.from_numpy(a).to(self.device)
        logits = self.policy_net(s_t)
        loss = self.sl_loss(logits, a_t)
        self.sl_optim.zero_grad()
        loss.backward()
        self.sl_optim.step()

    # ── average strategy (for logging) ────────────────────────

    def get_average_policy(self, card, history):
        feats = torch.from_numpy(self._features(card, history)).to(self.device)
        with torch.no_grad():
            logits = self.policy_net(feats.unsqueeze(0))[0].cpu().numpy()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return probs


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════

def train_nfsp(iterations, game_name="kuhn", log_prefix="nfsp"):
    game = get_game(game_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = NFSPAgent(game, device)

    print(f"NFSP  |  game: {game.name}  |  episodes: {iterations:,}  "
          f"|  η={ETA}  |  device: {device}")

    # Build game tree to enumerate infosets
    from tabular.game_tree import GameTree
    tree = GameTree(game)
    # Enumerate infosets from game tree nodes × card ranks
    infoset_keys = []
    for hid, node in tree.nodes.items():
        if node.is_terminal:
            continue
        hist = tree.history_str(hid)
        for rank in ['J', 'Q', 'K']:
            infoset_keys.append(rank + hist)

    def _strategy_map():
        """Build {infoset_str: FakeNode} from policy network."""
        node_map = {}
        for key in infoset_keys:
            card = key[0]
            hist = key[1:]
            try:
                feats = np.array(game.infoset_to_features(card + hist), dtype=np.float32)
            except Exception:
                continue
            with torch.no_grad():
                logits = agent.policy_net(torch.from_numpy(feats).unsqueeze(0).to(device))[0].cpu().numpy()
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            node_map[key] = _FakeNode(probs)
        return node_map

    # Log header
    save_strategy_txt(_strategy_map(), 0, 0, iterations, log_prefix, game_name=game_name)

    total_reward = 0.0
    best_reward = -float("inf")
    best_state = None
    snapshot_every = max(1, iterations // 100)

    for episode in range(1, iterations + 1):
        cards = game.deal_cards()
        history = ""
        ep_reward = 0.0

        # Store trajectory for this episode
        trajectory = []  # (player, card, history, action_id, mode)

        while not game.is_terminal(history):
            player = len(history) % 2
            legal = game.get_legal_actions(history)
            legal_ids = [game.ACTIONS.index(a) for a in legal]

            if player == 0:
                action_id, mode = agent.act(cards[0], history, legal_ids)
                trajectory.append((cards[0], history, action_id, mode))
            else:
                # Opponent: use average policy
                if random.random() < 0.5:
                    feats = agent._features(cards[1], history)
                    feats_t = torch.from_numpy(feats).to(device)
                    with torch.no_grad():
                        logits = agent.policy_net(feats_t.unsqueeze(0))[0].cpu().numpy()
                    mask = np.full(game.num_actions, -1e9)
                    for a in legal_ids:
                        mask[a] = logits[a]
                    probs = np.exp(mask - mask.max())
                    probs /= probs.sum()
                    action_id = np.random.choice(game.num_actions, p=probs)
                else:
                    action_id = random.choice(legal_ids)

            history = game.build_next_history(history, game.ACTIONS[action_id])

        # ── episode done, compute rewards and store ──
        payoff = game.get_payoff(history, cards)
        ep_reward = payoff  # from P0 perspective

        for card_0, hist, a_id, mode in trajectory:
            # Build features and next features for this step
            next_hist = game.build_next_history(hist, game.ACTIONS[a_id])
            done = game.is_terminal(next_hist)
            reward = payoff if done else 0.0

            s = agent._features(card_0, hist)
            ns = agent._features(card_0, next_hist) if not done else np.zeros_like(s)

            agent.rl_add(s, a_id, reward, ns, 1.0 if done else 0.0)

            # Store in SL buffer only when using average-policy mode
            if mode == "sl":
                agent.sl_buffer.add(s, a_id)

        # ── train ──
        if episode % RL_TRAIN_EVERY == 0:
            agent.rl_train_step()
        if episode % SL_TRAIN_EVERY == 0:
            agent.sl_train_step()

        total_reward += ep_reward

        if episode % snapshot_every == 0:
            avg = total_reward / episode
            print(f"  episode {episode:>8,}  |  avg reward: {avg:+.4f}  "
                  f"|  rl_buf: {len(agent.rl_buffer)},  sl_buf: {len(agent.sl_buffer)}")
            save_strategy_txt(_strategy_map(), episode, avg, iterations,
                              log_prefix, game_name=game_name)
            if avg > best_reward:
                best_reward = avg
                best_state = {
                    "q_net": {k: v.cpu().clone() for k, v in agent.q_net.state_dict().items()},
                    "policy_net": {k: v.cpu().clone() for k, v in agent.policy_net.state_dict().items()},
                }

    if best_state is not None:
        agent.q_net.load_state_dict(best_state["q_net"])
        agent.policy_net.load_state_dict(best_state["policy_net"])
        print(f"\n[CHECKPOINT] Restored best (avg reward = {best_reward:+.4f})")

    save_model(best_state, iterations, log_prefix, game_name=game_name)
    print(f"Average reward: {total_reward / iterations:+.4f}")
