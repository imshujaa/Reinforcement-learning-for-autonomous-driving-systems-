import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import random
import os   

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, state_dim * 2)
        self.fc2 = nn.Linear(state_dim * 2, state_dim)
        self.actor_head = nn.Linear(state_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.actor_head(x)
        return logits


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, state_dim * 2)
        self.fc2 = nn.Linear(state_dim * 2, state_dim)
        self.value_head = nn.Linear(state_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value.squeeze(-1)


class PPO:
    def __init__(self, state_dim, action_dim, gamma=0.99, clip_eps=0.2, lr=3e-4, epochs=10, batch_size=64,
                 vf_coef=0.5, ent_coef=0.01, max_buffer_size=5000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = deque(maxlen=max_buffer_size)

    def act(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.actor(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def get_action_and_value(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.actor(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state_tensor)
        return action.item(), log_prob.item(), value.item(), entropy.item()

    def memorize(self, state, action, reward, log_prob, value, done):
        self.buffer.append((state, action, reward, log_prob, value, done))

    def compute_returns_and_advantages(self, rewards, values, dones, gamma=0.99, lam=0.95):
        advantages = []
        returns = []
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            next_value = values[step]
        returns = [adv + val for adv, val in zip(advantages, values)]
        return returns, advantages

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, old_log_probs, values, dones = zip(*self.buffer)
        self.buffer.clear()

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.tensor(old_log_probs).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        returns, advantages = self.compute_returns_and_advantages(rewards, values, dones, self.gamma)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size

                batch_states = states[start:end]
                batch_actions = actions[start:end]
                batch_old_log_probs = old_log_probs[start:end]
                batch_returns = returns[start:end]
                batch_advantages = advantages[start:end]

                logits = self.actor(batch_states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                values = self.critic(batch_states)

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy.mean()

                total_loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

                # Optimize actor
                self.actor_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                # Optimize critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
    def save_model(self, filename, env_label="default"):
        model_dir = os.path.join("models", "ADV", env_label, "PPO")
        os.makedirs(model_dir, exist_ok=True)

        actor_path = os.path.join(model_dir, f"actor_{filename}")
        critic_path = os.path.join(model_dir, f"critic_{filename}")

        try:
            torch.save(self.actor.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)
            print(f"Modelli salvati in:\n- {actor_path}\n- {critic_path}")
        except Exception as e:
            print(f"Errore nel salvataggio dei modelli PPO: {e}")

    def load_model(self, filename, env_label="default"):
        model_dir = os.path.join("models", "ADV", env_label, "PPO")

        actor_path = os.path.join(model_dir, f"actor_{filename}")
        critic_path = os.path.join(model_dir, f"critic_{filename}")

        try:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            print(f"Modelli caricati da:\n- {actor_path}\n- {critic_path}")
        except Exception as e:
            print(f"Errore nel caricamento dei modelli PPO: {e}")