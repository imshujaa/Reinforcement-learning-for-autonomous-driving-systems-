"""Base agent interface for reinforcement learning agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
from pathlib import Path


class BaseAgent(ABC):
    """Abstract base class for all RL agents in CaRLeT.
    
    This class defines the interface that all agents must implement,
    ensuring consistency across different algorithm implementations.
    
    Attributes:
        state_size: Dimension of the observation space
        action_size: Number of possible actions
        name: Human-readable name for the agent
    """
    
    def __init__(self, state_size: int, action_size: int, name: str = "BaseAgent"):
        """Initialize the base agent.
        
        Args:
            state_size: Dimension of the observation space
            action_size: Number of possible actions
            name: Human-readable name for the agent
            
        Raises:
            ValueError: If state_size or action_size are not positive integers
        """
        if state_size <= 0:
            raise ValueError(f"state_size must be positive, got {state_size}")
        if action_size <= 0:
            raise ValueError(f"action_size must be positive, got {action_size}")
            
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
    
    @abstractmethod
    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select an action given the current state.
        
        Args:
            state: Current observation from the environment
            epsilon: Exploration rate for epsilon-greedy policies
            
        Returns:
            Selected action index
        """
        pass
    
    @abstractmethod
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """Update the agent's policy based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether the episode terminated
            
        Returns:
            Dictionary of learning metrics (e.g., loss, Q-values)
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the agent's parameters to disk.
        
        Args:
            path: Directory path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the agent's parameters from disk.
        
        Args:
            path: Path to the saved model
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        pass
    
    def reset(self) -> None:
        """Reset agent state between episodes (optional)."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration for logging/reproducibility.
        
        Returns:
            Dictionary containing agent hyperparameters and settings
        """
        return {
            "name": self.name,
            "state_size": self.state_size,
            "action_size": self.action_size,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(state_size={self.state_size}, action_size={self.action_size})"
