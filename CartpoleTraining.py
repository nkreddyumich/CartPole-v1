import gym
import math
import random
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
    
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LR = 1e-3
NUM_EPISODES = 500

env = gym.make('CartPole-v1')
env.seed(0)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0

def select_action(state, steps_done):
    global EPS_START, EPS_END, EPS_DECAY
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    if random.random() < eps_threshold:
        return random.randrange(action_size)
    else:
        with torch.no_grad():
            return policy_net(state).argmax().item()
    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda d: not d, batch.done)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32) for s, d in zip(batch.next_state, batch.done) if not d]).to(device)
    
    state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state]).to(device)
    action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, device=device).float()
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if non_final_next_states.size(0) != 0:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)
    
    loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

episode_durations = []
scores = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    total_reward = 0
    for t in range(1, 10000):
        action = select_action(state, steps_done)
        steps_done += 1
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        reward = reward if not done else -10.0
        
        if not done:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            next_state_tensor = torch.zeros(state.size(), device=device)
        
        memory.push(state.cpu().numpy()[0], action, reward, next_state, done)
        
        state = next_state_tensor
        
        optimize_model()
        if done:
            episode_durations.append(t)
            scores.append(total_reward)
            break
    
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if episode % 10 == 0:
        avg_score = np.mean(scores[-10:])
        print(f"Episode {episode}\tAverage Score (last 10): {avg_score:.2f}")

torch.save(policy_net.state_dict(), "cartpoleData.pth")
print("Model saved as cartpoleData.pth")
