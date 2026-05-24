"""Bilateral Neural Fictitious Self-Play — both P0 and P1 learn.

Each player maintains their own DQN (best-response) and policy network
(average strategy).  They play against each other's average policy,
creating the self-play dynamics needed to converge toward Nash.
"""
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_selector import get_game
from dqn.model import QNetwork, _hidden_dim
from utils import save_strategy_txt, save_model
from nfsp_dual.logger import save_dual_log


# ═══════════════════════════════════════════════════════════════════
#  Policy network
# ═══════════════════════════════════════════════════════════════════

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)   # raw logits


# ═══════════════════════════════════════════════════════════════════
#  Reservoir buffer (for SL)
# ═══════════════════════════════════════════════════════════════════

class ReservoirBuffer:
    def __init__(self, capacity=200_000):
        self.buffer = []
        self.capacity = capacity
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
        return (np.array([b[0] for b in batch], dtype=np.float32),
                np.array([b[1] for b in batch], dtype=np.int64))

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════
#  Hyper-parameters
# ═══════════════════════════════════════════════════════════════════

RL_BUFFER_SIZE = 100_000
SL_BUFFER_SIZE = 200_000
RL_BATCH       = 256
SL_BATCH       = 256
RL_LR          = 0.0001
SL_LR          = 0.001
GAMMA          = 0.99
EPS            = 0.06         # fixed ε for DQN
ETA            = 0.1          # P(best-response) in NFSP
TARGET_UPDATE  = 1_000
TRAIN_START    = 1_000
RL_EVERY       = 1
SL_EVERY       = 1


# ═══════════════════════════════════════════════════════════════════
#  Per-player NFSP agent
# ═══════════════════════════════════════════════════════════════════

class NFSPPlayer:
    """One player's DQN + policy-network bundle."""
    def __init__(self, game, device):
        hdim = _hidden_dim(game.name)
        na = game.num_actions
        self.q_net     = QNetwork(game.feature_dim, hdim, na).to(device)
        self.target_net = QNetwork(game.feature_dim, hdim, na).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.policy_net = PolicyNetwork(game.feature_dim, hdim, na).to(device)
        self.rl_optim = optim.Adam(self.q_net.parameters(), lr=RL_LR)
        self.sl_optim = optim.Adam(self.policy_net.parameters(), lr=SL_LR)

        self.rl_buffer = []
        self.rl_buf_pos = 0
        self.sl_buffer = ReservoirBuffer(SL_BUFFER_SIZE)
        self.steps = 0

    # ── action selection ──────────────────────────────────────

    def select_br(self, feats_t, legal_ids):
        """Best-response: ε-greedy over Q-values."""
        if random.random() < EPS:
            return random.choice(legal_ids)
        with torch.no_grad():
            q = self.q_net(feats_t.unsqueeze(0))[0].cpu().numpy()
        return max(legal_ids, key=lambda a: q[a])

    def select_avg(self, feats_t, legal_ids):
        """Average policy: sample from policy_net."""
        na = len(legal_ids)  # total num_actions needed for masking
        with torch.no_grad():
            logits = self.policy_net(feats_t.unsqueeze(0))[0].cpu().numpy()
        # Need full logits size — mask to legal
        mask = np.full(logits.shape, -1e9)
        for a in legal_ids:
            mask[a] = logits[a]
        probs = np.exp(mask - mask.max())
        probs /= probs.sum()
        return np.random.choice(len(probs), p=probs)

    def act(self, feats_t, legal_ids):
        """NFSP dual-mode: η → BR, 1-η → avg."""
        if random.random() < ETA:
            return self.select_br(feats_t, legal_ids), "rl"
        else:
            return self.select_avg(feats_t, legal_ids), "sl"

    # ── training ──────────────────────────────────────────────

    def rl_add(self, s, a, r, ns, d):
        e = (s, a, r, ns, d)
        if len(self.rl_buffer) < RL_BUFFER_SIZE:
            self.rl_buffer.append(e)
        else:
            self.rl_buffer[self.rl_buf_pos] = e
        self.rl_buf_pos = (self.rl_buf_pos + 1) % RL_BUFFER_SIZE

    def rl_train(self):
        if len(self.rl_buffer) < TRAIN_START:
            return
        batch = random.sample(self.rl_buffer, min(RL_BATCH, len(self.rl_buffer)))
        s, a, r, ns, d = zip(*batch)
        s_t  = torch.tensor(np.array(s, dtype=np.float32), device=self.q_net.net[0].weight.device)
        a_t  = torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(s_t.device)
        r_t  = torch.tensor(r, dtype=torch.float32).to(s_t.device)
        ns_t = torch.tensor(np.array(ns, dtype=np.float32), device=s_t.device)
        d_t  = torch.tensor(d, dtype=torch.float32).to(s_t.device)

        q_pred = self.q_net(s_t).gather(1, a_t).squeeze(1)
        with torch.no_grad():
            target = r_t + GAMMA * self.target_net(ns_t).max(1)[0] * (1 - d_t)
        loss = nn.MSELoss()(q_pred, target)
        self.rl_optim.zero_grad()
        loss.backward()
        self.rl_optim.step()
        self.steps += 1
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def sl_train(self):
        if len(self.sl_buffer) < TRAIN_START // 10:
            return
        s, a = self.sl_buffer.sample(SL_BATCH)
        dev = self.policy_net.net[0].weight.device
        s_t = torch.tensor(s, device=dev)
        a_t = torch.tensor(a, dtype=torch.int64, device=dev)
        logits = self.policy_net(s_t)
        loss = nn.CrossEntropyLoss()(logits, a_t)
        self.sl_optim.zero_grad()
        loss.backward()
        self.sl_optim.step()

    def get_strategy(self, feats_t):
        with torch.no_grad():
            logits = self.policy_net(feats_t.unsqueeze(0))[0].cpu().numpy()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return probs


# ═══════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════

class _FakeNode:
    def __init__(self, avg): self._avg = avg
    def get_average_strategy(self): return self._avg


def _features(game, card, history):
    return np.array(game.infoset_to_features(card + history), dtype=np.float32)


def _legal_ids(game, history):
    return [game.ACTIONS.index(a) for a in game.get_legal_actions(history)]


def train_nfsp_dual(iterations, game_name="kuhn", log_prefix="nfsp_dual"):
    game = get_game(game_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p0 = NFSPPlayer(game, device)
    p1 = NFSPPlayer(game, device)

    print(f"NFSP Dual  |  game: {game.name}  |  episodes: {iterations:,}  "
          f"|  η={ETA}  |  device: {device}")

    # Enumerate infosets for logging
    from tabular.game_tree import GameTree
    tree = GameTree(game)
    infoset_keys = []
    for hid, node in tree.nodes.items():
        if node.is_terminal:
            continue
        hist = tree.history_str(hid)
        for rank in ['J', 'Q', 'K']:
            infoset_keys.append(rank + hist)

    def _strategy_map(who):
        player = p0 if who == 0 else p1
        node_map = {}
        for key in infoset_keys:
            card = key[0]; hist = key[1:]
            try:
                feats = torch.tensor(_features(game, card, hist), device=device)
            except Exception:
                continue
            node_map[key] = _FakeNode(player.get_strategy(feats))
        return node_map

    # Clean old log
    log_name = f"{game_name}_strategy_{log_prefix}_{iterations:.0e}.txt"
    log_path = os.path.join('logs', log_name)
    if os.path.exists(log_path):
        os.remove(log_path)

    save_strategy_txt(_strategy_map(0), 0, 0, iterations, log_prefix, game_name=game_name)
    # dual log header (both P0+P1)
    save_dual_log(_strategy_map(0), _strategy_map(1), 0, 0.0, iterations, game_name=game_name)

    total_reward = 0.0
    best_reward = -float("inf")
    best_state = None
    snap_every = max(1, iterations // 100)

    for ep in range(1, iterations + 1):
        cards = game.deal_cards()
        history = ""
        # trajectories: (who, card, history, action_id, mode)
        traj = []

        while not game.is_terminal(history):
            player = len(history) % 2
            agent  = p0 if player == 0 else p1
            legal  = _legal_ids(game, history)
            feats  = torch.tensor(_features(game, cards[player], history), device=device)
            action_id, mode = agent.act(feats, legal)
            traj.append((player, cards[player], history, action_id, mode))
            history = game.build_next_history(history, game.ACTIONS[action_id])

        payoff = game.get_payoff(history, cards)
        ep_reward = payoff   # P0 perspective

        for player, card, hist, a_id, mode in traj:
            agent = p0 if player == 0 else p1
            next_hist = game.build_next_history(hist, game.ACTIONS[a_id])
            done = game.is_terminal(next_hist)
            r = payoff if (player == 0) else -payoff   # reward from THIS player's view
            if not done:
                r = 0.0

            s  = _features(game, card, hist)
            ns = _features(game, card, next_hist) if not done else np.zeros_like(s)
            agent.rl_add(s, a_id, r, ns, 1.0 if done else 0.0)

            if mode == "sl":
                agent.sl_buffer.add(s, a_id)

        # Train both
        if ep % RL_EVERY == 0:
            p0.rl_train()
            p1.rl_train()
        if ep % SL_EVERY == 0:
            p0.sl_train()
            p1.sl_train()

        total_reward += ep_reward

        if ep % snap_every == 0:
            avg = total_reward / ep
            print(f"  ep {ep:>8,}  |  avg reward: {avg:+.4f}  "
                  f"|  rl_buf: {len(p0.rl_buffer)}+{len(p1.rl_buffer)}  "
                  f"|  sl_buf: {len(p0.sl_buffer)}+{len(p1.sl_buffer)}")
            save_strategy_txt(_strategy_map(0), ep, avg, iterations,
                              log_prefix, game_name=game_name)
            save_dual_log(_strategy_map(0), _strategy_map(1), ep, avg,
                          iterations, game_name=game_name)
            if avg > best_reward:
                best_reward = avg
                best_state = {
                    f"p0_{k}": v.cpu().clone() for k, v in p0.policy_net.state_dict().items()
                }

    # Save
    save_model({"p0": best_state}, iterations, log_prefix, game_name=game_name)
    print(f"Average reward: {total_reward / iterations:+.4f}")
