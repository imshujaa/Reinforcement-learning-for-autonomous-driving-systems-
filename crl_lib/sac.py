import random
from distutils.util import strtobool
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions.categorical import Categorical
from collections import deque


class SoftQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, state_size*2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(state_size*2, state_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(state_size, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, state_size*2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(state_size*2, state_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(state_size, action_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs

        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        log_prob = F.log_softmax(logits, dim=1)
        
        return action, log_prob, action_probs


class SAC():
    def __init__(self, state_size, action_size, _q_learning_rate, _p_learning_rate, _target_entropy_scale, _autotune_alpha, _alpha, _tau, _discount, _batch_size, _buffer_size, _gradient_steps):
        self.q_lr = _q_learning_rate # learning rate of the Q network network optimizer
        self.policy_lr = _p_learning_rate # learning rate of the policy network optimizer
        self.rb = deque(maxlen=_buffer_size)

        self.autotune = _autotune_alpha
        self.target_entropy_scale = _target_entropy_scale # coefficient for scaling the autotune entropy target
        self.alpha = _alpha # Entropy regularization coefficient

        self.gamma = _discount # discount factor 
        self.tau = _tau # target smoothing coefficient 
        self.batch_size = _batch_size 
        self.buffer_size = _buffer_size 

        self.state_size = state_size
        self.action_size = action_size
        
        self.env_label = "highway"  # Default environment label, can be overridden
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.qf1 = SoftQNetwork(self.state_size, self.action_size).to(self.device)
        self.qf2 = SoftQNetwork(self.state_size, self.action_size).to(self.device)
        self.qf1_target = SoftQNetwork(self.state_size, self.action_size).to(self.device)
        self.qf2_target = SoftQNetwork(self.state_size, self.action_size).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr, eps=1e-4) 

        # Automatic entropy tuning
        if self.autotune:
            self.target_entropy = -self.target_entropy_scale * torch.log(1 / torch.tensor(self.action_size))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr, eps=1e-4)


    def act(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        actions, _, _ = self.actor.get_action(torch.Tensor(state).to(self.device))
        actions = actions.detach().cpu().numpy()
        return actions[0]
    
    def memorize(self, state, action, reward, next_state, done):
        self.rb.append((state, action, reward, next_state, done))

    
    def replay(self):

        data = random.sample(self.rb, self.batch_size)
        state, action, reward, next_state, done = zip(*data)

        state = np.vstack(state)
        next_state = np.vstack(next_state)
        action = np.array(action)
        reward = np.array(reward)[:, None]
        done = np.array(done)[:, None]
        
        state_tensor = torch.Tensor(state).to(self.device)
        next_state_tensor = torch.Tensor(next_state).to(self.device)
        action_tensor = torch.Tensor(action).to(self.device)
        action_tensor = action_tensor.to(torch.int64) 

        # CRITIC training
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.actor.get_action(next_state_tensor)
            qf1_next_target = self.qf1_target(next_state_tensor)
            qf2_next_target = self.qf2_target(next_state_tensor)
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = reward + (1 - done) * self.gamma * (min_qf_next_target.detach().cpu().numpy())

        
        # use Q-values only for the taken actions
        qf1_values = self.qf1(state_tensor)
        qf2_values = self.qf2(state_tensor)

        qf1_a_values = qf1_values.gather(1, action_tensor.view(-1, 1)).squeeze(1)
        qf2_a_values = qf2_values.gather(1, action_tensor.view(-1, 1)).squeeze(1)

        qf1_loss = F.mse_loss(qf1_a_values, torch.Tensor(next_q_value).to(self.device))
        qf2_loss = F.mse_loss(qf2_a_values, torch.Tensor(next_q_value).to(self.device))
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        _, log_pi, action_probs = self.actor.get_action(state_tensor)
        with torch.no_grad():
            qf1_values = self.qf1(state_tensor)
            qf2_values = self.qf2(state_tensor)
            min_qf_values = torch.min(qf1_values, qf2_values)
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune:
            # re-use action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha * (log_pi + self.target_entropy).detach())).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()



    def update_target(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, filename):
        model_dir = os.path.join("models", "ADV", self.env_label, "SAC")
        os.makedirs(model_dir, exist_ok=True)
        full_path = os.path.join(model_dir, filename)
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'qf1_target_state_dict': self.qf1_target.state_dict(),
            'qf2_target_state_dict': self.qf2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'alpha': self.alpha,
            'autotune': self.autotune
        }
        
        if self.autotune:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['a_optimizer_state_dict'] = self.a_optimizer.state_dict()
        
        torch.save(checkpoint, full_path)
        print(f"Model saved in: {full_path}")
    
    def load_model(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.qf1.load_state_dict(checkpoint['qf1_state_dict'])
            self.qf2.load_state_dict(checkpoint['qf2_state_dict'])
            self.qf1_target.load_state_dict(checkpoint['qf1_target_state_dict'])
            self.qf2_target.load_state_dict(checkpoint['qf2_target_state_dict'])

            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])

            self.alpha = checkpoint['alpha']
            self.autotune = checkpoint.get('autotune', False)

            if self.autotune and checkpoint.get('log_alpha') is not None:
                self.log_alpha = checkpoint['log_alpha'].to(self.device)
                self.a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])

            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

