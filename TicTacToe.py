import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- 1. Environment ---
class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.agent = 1
        self.opponent = -1
        self.reset()

    def reset(self):
        self.board[:] = 0
        self.mover = np.random.choice([1, -1])
        return self.board.copy()

    def available_actions(self):
        return np.where(self.board == 0)[0]

    def reward_done(self):
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for i,j,k in lines:
            s = self.board[i] + self.board[j] + self.board[k]
            if s == 3:  return 1.0, True
            if s == -3: return -1.0, True
        if np.all(self.board != 0):
            return 0.0, True
        return 0.0, False

    def step(self, action, opponent_policy=None):
        if self.board[action] != 0:
            return self.board.copy(), -1.0, True

        self.board[action] = self.agent
        r, d = self.reward_done()
        if d:
            return self.board.copy(), r, True

        if opponent_policy is None:
            opp_action = np.random.choice(self.available_actions())
        else:
            opp_action = opponent_policy(self.board.copy(), self.available_actions())

        self.board[opp_action] = self.opponent
        r, d = self.reward_done()

        if d and r == -1:
            return self.board.copy(), -1.0, True

        return self.board.copy(), r, d

# --- 2. Model ---
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)

    def forward(self, x):
        return self.net(x)



# --- 3. Helpers ---
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d, next_avail):
        self.buf.append((s, a, r, ns, d, next_avail))

    def sample(self, batch):
        samples = random.sample(self.buf, batch)
        s,a,r,ns,d,na = zip(*samples)
        return (
            torch.FloatTensor(s),
            torch.LongTensor(a).unsqueeze(1),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(ns),
            torch.FloatTensor(d).unsqueeze(1),
            na
        )

    def __len__(self):
        return len(self.buf)


def select_action(model, state, avail, epsilon):
    if random.random() < epsilon:
        return random.choice(avail)

    with torch.no_grad():
        q = model(torch.FloatTensor(state).unsqueeze(0))[0]

    mask = torch.full_like(q, -1e9)
    mask[avail] = q[avail]
    return mask.argmax().item()

def train_step(model, target, optim, buffer, batch=128, gamma=0.95):
    if len(buffer) < batch:
        return None  # <- important for clean stats handling

    s, a, r, ns, d, next_avail = buffer.sample(batch)

    q_sa = model(s).gather(1, a)

    with torch.no_grad():
        q_next = target(ns)
        masked = torch.full_like(q_next, -1e9)
        for i, avail in enumerate(next_avail):
            masked[i, avail] = q_next[i, avail]

        q_max = masked.max(1, keepdim=True)[0]
        target_q = r + gamma * q_max * (1 - d)

    td_error = target_q - q_sa
    loss = nn.SmoothL1Loss()(q_sa, target_q)

    optim.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optim.step()

    # ---- stats returned here ----
    return {
        "loss": loss.item(),
        "td_error": td_error.abs().mean().item(),
        "q_mean": q_sa.mean().item(),
    }

def train():
    env = TicTacToeEnv()
    model = QNetwork()
    target = QNetwork()
    target.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    buffer = ReplayBuffer()

    steps = 0
    epsilon = 1.0

    losses = []
    ep_lengths = []

    def update_target():
        target.load_state_dict(model.state_dict())

    for episode in range(70000):
        s = env.reset()
        ep_len = 0

        if env.mover == -1:
            env.board[random.choice(env.available_actions())] = -1
            s = env.board.copy()

        done = False
        while not done:
            avail = env.available_actions()
            epsilon = max(0.01, 1.0 - steps / 40000) if episode < 50000 else 0.001

            if episode > 30000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4
            if episode > 50000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5

            a = select_action(model, s, avail, epsilon)
            ns, r, done = env.step(a)
            next_avail = env.available_actions() if not done else []

            buffer.push(s, a, r, ns, done, next_avail)
            s = ns
            steps += 1
            ep_len += 1

            stats = train_step(model, target, optimizer, buffer)

            if stats is not None:
                losses.append(stats["loss"])

        ep_lengths.append(ep_len)

        # --- periodic target update
        if episode % 500 == 0:
            update_target()

        # --- stats print
        if episode % 2000 == 0 and episode > 0:
            w, d, l = evaluate_vs_random(model)

            print(
                f"[Ep {episode:6d}] "
                f"ε={epsilon:.2f} | "
                f"Win={w:.2%} Draw={d:.2%} Loss={l:.2%} | "
                f"AvgLen={np.mean(ep_lengths[-2000:]):.2f} | "
                f"Loss={np.mean(losses[-5000:]):.4f}"
            )

    return model, target, optimizer, buffer

def evaluate_vs_random(model, games=500):
    env = TicTacToeEnv()
    wins = draws = losses = 0

    for _ in range(games):
        s = env.reset()
        done = False

        if env.mover == -1:
            env.board[random.choice(env.available_actions())] = -1
            s = env.board.copy()

        while not done:
            a = select_action(model, s, env.available_actions(), epsilon=0.0)
            s, r, done = env.step(a)

        if r == 1: wins += 1
        elif r == 0: draws += 1
        else: losses += 1

    return wins / games, draws / games, losses / games