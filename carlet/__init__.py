"""
CaRLeT - Causal Reinforcement Learning for Autonomous Driving

A production-ready implementation of causal reinforcement learning
for multi-agent autonomous driving systems.
"""

__version__ = "1.0.0"
__author__ = "CaRLeT Research Team"
__license__ = "MIT"

from carlet.agents import QTableAgent, DQNAgent, PPOAgent, SACAgent
from carlet.training import CRLTrainer
from carlet.environments import HighwayMAEnv

__all__ = [
    "QTableAgent",
    "DQNAgent", 
    "PPOAgent",
    "SACAgent",
    "CRLTrainer",
    "HighwayMAEnv",
]
