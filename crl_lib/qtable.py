# #
# # MIT License
# # Copyright (c) 2018 Valentyn N Sichkar
# # github.com/sichkar-valentyn
# #
# # Reference to:
# # Valentyn N Sichkar. Reinforcement Learning Algorithms for global path planning // GitHub platform. DOI: 10.5281/zenodo.1317899



# Importing libraries
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Importing function from the env.py

import os 


class QTable:
    def __init__(self, state_size, action_size, _learning_rate=0.1, _discount=0.95, _num_bins=10, env_label="highway"):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = _learning_rate
        self.discount = _discount
        self.env_label = env_label  # Store environment label for dynamic paths

        # Discretization
        numBins = _num_bins
        self.bins = [
            np.linspace(-1, 1, numBins),
        ]

        self.q_table = {}

    def get_discrete_state(self, state):
        stateIndex = []
        for i in range(self.state_size):
            stateIndex.append(np.digitize(state[i], self.bins[0]) - 1)
        return tuple(stateIndex)

    def get_q_value(self, state, action):
        """Get the Q-value for a state-action pair, initializing to 0 if it doesn't exist."""
        d_state = self.get_discrete_state(state)
        return self.q_table.get(d_state + (action,), 0)

    def update_q_value(self, state, action, value):
        """Update the Q-value for a state-action pair."""
        d_state = self.get_discrete_state(state)
        self.q_table[d_state + (action,)] = value

    def get_q_table(self):
        return self.q_table

    # Function for choosing the action for the agent
    def choose_action(self, state):
        d_state = self.get_discrete_state(state)
        
        # If the state has no Q-values, initialize random values for all actions
        if not any(d_state + (a,) in self.q_table for a in range(self.action_size)):
            for a in range(self.action_size):
                self.q_table[d_state + (a,)] = np.random.uniform(low=-2, high=0)  # Random initialization

        # Choose the action with the highest Q-value
        return np.argmax([self.get_q_value(state, a) for a in range(self.action_size)])
    

    def choose_action_planning(self, state, discarded_actions, max_q = True):
    # Create a list of all actions except the discarded ones
        valid_actions = [a for a in range(self.action_size) if a not in discarded_actions]
        
        if not valid_actions:
            raise ValueError("No valid actions available after discarding.")

        if max_q:
            # Select the action among valid_actions with the highest Q-value
            action = max(valid_actions, key=lambda a: self.get_q_value(state, a))
        else:
            action = min(valid_actions, key=lambda a: self.get_q_value(state, a))
            
        return action


    # Function for learning and updating Q-table with new knowledge
    def learn(self, state, action, reward, next_state, done):
        # Ensure the current state and next state are initialized in the Q-table
        d_state = self.get_discrete_state(state)
        d_next_state = self.get_discrete_state(next_state)
        
        # Initialize random values for unvisited states
        if not any(d_state + (a,) in self.q_table for a in range(self.action_size)):
            for a in range(self.action_size):
                self.q_table[d_state + (a,)] = np.random.uniform(low=-2, high=0)

        if not any(d_next_state + (a,) in self.q_table for a in range(self.action_size)):
            for a in range(self.action_size):
                self.q_table[d_next_state + (a,)] = np.random.uniform(low=-2, high=0)

        if done:
            target = reward  # No future reward
        else:
            maxFutureQ = max(self.get_q_value(next_state, a) for a in range(self.action_size))
            target = reward + self.discount * maxFutureQ

        currentQ = self.get_q_value(state, action)
        newQ = (1 - self.learning_rate) * currentQ + self.learning_rate * target
        self.update_q_value(state, action, newQ)

    def _save_model(self, descr):
        try:
            models_dir = os.path.join("models", "ADV", self.env_label, "Q")
            os.makedirs(models_dir, exist_ok=True) 
            filename = f"table_{str(descr).replace(' ', '_').replace('.', '_').replace(':', '_')}.txt"
            filepath = os.path.join(models_dir, filename)
            with open(filepath, "w") as file:
                for key, value in self.q_table.items():
                    file.write(f"{key}: {value}\n")
        except Exception as e:
            print(f"Error saving Q-table: {e}")


    def _load_model(self, file_name):
        try:
            file_name = file_name
            self.q_table = {}  
            with open(file_name, "r") as file:
                for line in file:
                    key, value = line.strip().split(": ", 1)
                    self.q_table[eval(key)] = eval(value)
        except Exception as e:
            raise Exception(f"Error loading Q-table: {e}")