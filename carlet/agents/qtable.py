"""Q-Learning agent with table-based value function."""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
import logging

from carlet.agents.base import BaseAgent


logger = logging.getLogger(__name__)


class QTableAgent(BaseAgent):
    """Q-Learning agent using discretized state space and tabular Q-values.
    
    This agent maintains a Q-table mapping (state, action) pairs to expected
    returns. States are discretized into bins for tractability. The agent
    learns using the standard Q-learning update rule.
    
    Attributes:
        learning_rate: Learning rate (alpha) for Q-value updates
        discount: Discount factor (gamma) for future rewards
        num_bins: Number of bins for state discretization
        q_table: Dictionary mapping state-action tuples to Q-values
        bins: List of bin edges for each state dimension
    
    Example:
        >>> agent = QTableAgent(state_size=10, action_size=5)
        >>> state = np.random.randn(10)
        >>> action = agent.choose_action(state, epsilon=0.1)
        >>> metrics = agent.learn(state, action, reward=1.0, next_state, done=False)
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        num_bins: int = 10,
        name: str = "QTableAgent"
    ):
        """Initialize the Q-Table agent.
        
        Args:
            state_size: Dimension of the observation space
            action_size: Number of possible actions
            learning_rate: Learning rate for Q-value updates (0 < alpha <= 1)
            discount: Discount factor for future rewards (0 <= gamma < 1)
            num_bins: Number of bins for discretizing each state dimension
            name: Human-readable name for the agent
            
        Raises:
            ValueError: If hyperparameters are out of valid range
        """
        super().__init__(state_size, action_size, name)
        
        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if not 0 <= discount < 1:
            raise ValueError(f"discount must be in [0, 1), got {discount}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        
        self.learning_rate = learning_rate
        self.discount = discount
        self.num_bins = num_bins
        
        # Initialize discretization bins (assuming normalized states in [-1, 1])
        self.bins = np.linspace(-1, 1, num_bins)
        
        # Initialize empty Q-table
        self.q_table: Dict[Tuple, float] = {}
        
        logger.info(f"Initialized {self.name} with lr={learning_rate}, gamma={discount}, bins={num_bins}")
    
    def _discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Convert continuous state to discrete state tuple.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Tuple of bin indices representing discretized state
        """
        if len(state) != self.state_size:
            raise ValueError(f"Expected state of size {self.state_size}, got {len(state)}")
        
        discrete_state = []
        for value in state:
            # Clip to valid range to handle edge cases
            clipped_value = np.clip(value, -1, 1)
            bin_idx = np.digitize(clipped_value, self.bins) - 1
            # Ensure index is within valid range
            bin_idx = np.clip(bin_idx, 0, self.num_bins - 1)
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def _get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q-value for state-action pair, initializing if needed.
        
        Args:
            state: Current state
            action: Action index
            
        Returns:
            Q-value for the state-action pair
        """
        discrete_state = self._discretize_state(state)
        key = discrete_state + (action,)
        
        # Lazy initialization with small random values
        if key not in self.q_table:
            self.q_table[key] = np.random.uniform(low=-2, high=0)
        
        return self.q_table[key]
    
    def _set_q_value(self, state: np.ndarray, action: int, value: float) -> None:
        """Set Q-value for state-action pair.
        
        Args:
            state: Current state
            action: Action index
            value: New Q-value
        """
        discrete_state = self._discretize_state(state)
        key = discrete_state + (action,)
        self.q_table[key] = value
    
    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current observation
            epsilon: Exploration rate (0 = fully greedy, 1 = fully random)
            
        Returns:
            Selected action index
        """
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_size)
        
        # Greedy action selection
        q_values = [self._get_q_value(state, a) for a in range(self.action_size)]
        return int(np.argmax(q_values))
    
    def choose_action_excluding(
        self,
        state: np.ndarray,
        excluded_actions: List[int],
        maximize: bool = True
    ) -> int:
        """Choose action excluding certain actions (useful for planning).
        
        Args:
            state: Current state
            excluded_actions: List of action indices to exclude
            maximize: If True, select max Q-value; if False, select min
            
        Returns:
            Selected action index
            
        Raises:
            ValueError: If all actions are excluded
        """
        valid_actions = [a for a in range(self.action_size) if a not in excluded_actions]
        
        if not valid_actions:
            raise ValueError("No valid actions available after exclusion")
        
        if maximize:
            return max(valid_actions, key=lambda a: self._get_q_value(state, a))
        else:
            return min(valid_actions, key=lambda a: self._get_q_value(state, a))
    
    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """Update Q-values using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode terminated
            
        Returns:
            Dictionary with learning metrics
        """
        if action < 0 or action >= self.action_size:
            raise ValueError(f"Invalid action {action}, must be in [0, {self.action_size})")
        
        # Get current Q-value
        current_q = self._get_q_value(state, action)
        
        # Compute target
        if done:
            target = reward
        else:
            # Max Q-value over next state actions
            next_q_values = [self._get_q_value(next_state, a) for a in range(self.action_size)]
            max_next_q = max(next_q_values)
            target = reward + self.discount * max_next_q
        
        # Q-learning update
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * target
        self._set_q_value(state, action, new_q)
        
        # Return metrics for monitoring
        td_error = target - current_q
        return {
            "q_value": new_q,
            "td_error": td_error,
            "target": target,
        }
    
    def save(self, path: Path) -> None:
        """Save Q-table to disk.
        
        Args:
            path: Directory path to save the model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Q-table as JSON (convert tuples to strings for JSON compatibility)
        q_table_serializable = {str(k): v for k, v in self.q_table.items()}
        
        model_file = path / "q_table.json"
        config_file = path / "config.json"
        
        # Save Q-table
        with open(model_file, 'w') as f:
            json.dump(q_table_serializable, f, indent=2)
        
        # Save configuration
        config = self.get_config()
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved {self.name} to {path} ({len(self.q_table)} entries)")
    
    def load(self, path: Path) -> None:
        """Load Q-table from disk.
        
        Args:
            path: Path to the saved model directory
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        path = Path(path)
        model_file = path / "q_table.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Q-table file not found: {model_file}")
        
        # Load Q-table
        with open(model_file, 'r') as f:
            q_table_serializable = json.load(f)
        
        # Convert string keys back to tuples
        self.q_table = {eval(k): v for k, v in q_table_serializable.items()}
        
        logger.info(f"Loaded {self.name} from {path} ({len(self.q_table)} entries)")
    
    def load_legacy(self, filepath: str) -> None:
        """Load Q-table from legacy text format.
        
        Args:
            filepath: Path to legacy .txt Q-table file
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Legacy Q-table file not found: {filepath}")
        
        self.q_table = {}
        with open(filepath, 'r') as f:
            for line in f:
                if ': ' not in line:
                    continue
                key_str, value_str = line.strip().split(': ', 1)
                key = eval(key_str)
                value = eval(value_str)
                self.q_table[key] = value
        
        logger.info(f"Loaded legacy Q-table from {filepath} ({len(self.q_table)} entries)")
    
    def get_config(self) -> Dict:
        """Get agent configuration.
        
        Returns:
            Dictionary of hyperparameters and settings
        """
        config = super().get_config()
        config.update({
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "num_bins": self.num_bins,
            "q_table_size": len(self.q_table),
        })
        return config
