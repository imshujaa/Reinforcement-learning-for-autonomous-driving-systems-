import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from collections import deque, namedtuple


class SoftQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln3 = nn.LayerNorm(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.fc4(x)
        return x

    def get_action(self, x):
        logits = self(x)
        # Temperature scaling per migliorare l'esplorazione
        temperature = 1.0
        logits = logits / temperature
        
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
        self.Transition = namedtuple('Transition', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
    
    def add(self, state, action, reward, next_state, done):
        transition = self.Transition(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], []
            
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get transitions
        transitions = [self.buffer[i] for i in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return transitions, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[i] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class SAC:
    def __init__(self, state_size, action_size, **kwargs):
        self.q_lr = kwargs.get('q_lr', 1e-4)
        self.policy_lr = kwargs.get('policy_lr', 1e-4)
        self.buffer_size = kwargs.get('buffer_size', 50000)
        
        # Use prioritized replay buffer if enabled
        self.use_prioritized_buffer = kwargs.get('use_prioritized_buffer', True)
        if self.use_prioritized_buffer:
            self.rb = PrioritizedReplayBuffer(self.buffer_size)
        else:
            self.rb = deque(maxlen=self.buffer_size)

        self.autotune = kwargs.get('autotune', True)
        self.target_entropy_scale = kwargs.get('target_entropy_scale', 1.0)
        self.alpha = kwargs.get('alpha', 0.2)

        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.01)
        self.batch_size = kwargs.get('batch_size', 64)
        self.gradient_steps = kwargs.get('gradient_steps', 1)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)

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
        
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.policy_lr, eps=1e-4) 

        # Automatic entropy tuning
        if self.autotune:
            self.target_entropy = -self.target_entropy_scale * torch.log(1 / torch.tensor(self.action_size))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr, eps=1e-4)

        # Learning rate scheduling
        self.lr_scheduler_actor = torch.optim.lr_scheduler.StepLR(
            self.actor_optimizer, step_size=1000, gamma=0.95)
        self.lr_scheduler_q = torch.optim.lr_scheduler.StepLR(
            self.q_optimizer, step_size=1000, gamma=0.95)

    def act(self, state):
        actions, _, _ = self.actor.get_action(torch.Tensor(state).to(self.device))
        actions = actions.detach().cpu().numpy()
        return actions[0]
    
    def memorize(self, state, action, reward, next_state, done):
        if self.use_prioritized_buffer:
            self.rb.add(state, action, reward, next_state, done)
        else:
            self.rb.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.rb) < self.batch_size:
            return
            
        if self.use_prioritized_buffer:
            transitions, indices, weights = self.rb.sample(self.batch_size)
            
            # Unpack transitions
            states = np.vstack([t.state for t in transitions])
            actions = np.array([t.action for t in transitions])
            rewards = np.array([t.reward for t in transitions])[:, None]
            next_states = np.vstack([t.next_state for t in transitions])
            dones = np.array([t.done for t in transitions])[:, None]
            
            weight_tensor = torch.FloatTensor(weights).to(self.device)
        else:
            # Standard replay buffer
            data = random.sample(self.rb, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*data)

            states = np.vstack(states)
            next_states = np.vstack(next_states)
            actions = np.array(actions)
            rewards = np.array(rewards)[:, None]
            dones = np.array(dones)[:, None]
            
            weight_tensor = torch.ones(self.batch_size).to(self.device)
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(states).to(self.device)
        action_tensor = torch.LongTensor(actions).to(self.device)
        reward_tensor = torch.FloatTensor(rewards).to(self.device)
        next_state_tensor = torch.FloatTensor(next_states).to(self.device)
        done_tensor = torch.FloatTensor(dones).to(self.device)

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
            next_q_value = reward_tensor.squeeze() + (1 - done_tensor.squeeze()) * self.gamma * min_qf_next_target

        
        # use Q-values only for the taken actions
        qf1_values = self.qf1(state_tensor)
        qf2_values = self.qf2(state_tensor)

        qf1_a_values = qf1_values.gather(1, action_tensor.view(-1, 1)).squeeze(1)
        qf2_a_values = qf2_values.gather(1, action_tensor.view(-1, 1)).squeeze(1)

        # TD errors for priority update
        if self.use_prioritized_buffer:
            td_errors1 = (qf1_a_values - next_q_value).detach().cpu().numpy()
            td_errors2 = (qf2_a_values - next_q_value).detach().cpu().numpy()
            td_errors = (td_errors1 + td_errors2) / 2
            
            # Weighted loss
            qf1_loss = (weight_tensor * F.mse_loss(qf1_a_values, next_q_value, reduction='none')).mean()
            qf2_loss = (weight_tensor * F.mse_loss(qf2_a_values, next_q_value, reduction='none')).mean()
        else:
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

        qf_loss = qf1_loss + qf2_loss

        # Update Q-networks with gradient clipping
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), 
            self.max_grad_norm
        )
        self.q_optimizer.step()

        # Update priorities
        if self.use_prioritized_buffer:
            self.rb.update_priorities(indices, td_errors)

        # ACTOR training
        _, log_pi, action_probs = self.actor.get_action(state_tensor)
        with torch.no_grad():
            qf1_values = self.qf1(state_tensor)
            qf2_values = self.qf2(state_tensor)
            min_qf_values = torch.min(qf1_values, qf2_values)
        
        if self.use_prioritized_buffer:
            actor_loss = (weight_tensor.unsqueeze(1) * action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()
        else:
            actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        if self.autotune:
            # re-use action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha * (log_pi + self.target_entropy).detach())).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Update learning rate schedulers
        self.lr_scheduler_actor.step()
        self.lr_scheduler_q.step()

    def update_target(self):
        # update the target networks
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, filename):
        model_dir = os.path.join("models", "ADV", self.env_label, "SAC")
        os.makedirs(model_dir, exist_ok=True)

        full_path = os.path.join(model_dir, filename)

        try:
            checkpoint = {
                'actor_state_dict': self.actor.state_dict(),
                'qf1_state_dict': self.qf1.state_dict(),
                'qf2_state_dict': self.qf2.state_dict(),
                'qf1_target_state_dict': self.qf1_target.state_dict(),
                'qf2_target_state_dict': self.qf2_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'q_optimizer_state_dict': self.q_optimizer.state_dict(),
                'alpha': self.alpha,
                'autotune': self.autotune,
                'state_size': self.state_size,
                'action_size': self.action_size
            }
            
            if self.autotune:
                checkpoint['log_alpha'] = self.log_alpha
                checkpoint['a_optimizer_state_dict'] = self.a_optimizer.state_dict()
            
            torch.save(checkpoint, full_path)
            print(f"Model saved in: {full_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Check if it's a new format checkpoint (dictionary) or old format (just actor state dict)
            if isinstance(checkpoint, dict) and 'actor_state_dict' in checkpoint:
                # New format - load all components
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
                    
                print(f"Model loaded from {filepath} (full checkpoint)")
            else:
                # Old format - just actor state dict
                self.actor.load_state_dict(checkpoint)
                print(f"Model loaded from {filepath} (actor only)")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
