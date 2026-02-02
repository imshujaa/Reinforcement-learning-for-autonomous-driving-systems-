"""Unit tests for QTableAgent."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from carlet.agents.qtable import QTableAgent


class TestQTableAgent:
    """Test suite for QTableAgent."""
    
    def test_initialization(self):
        """Test agent initialization with valid parameters."""
        agent = QTableAgent(state_size=10, action_size=5)
        assert agent.state_size == 10
        assert agent.action_size == 5
        assert agent.learning_rate == 0.1
        assert agent.discount == 0.95
        assert len(agent.q_table) == 0
    
    def test_invalid_initialization(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            QTableAgent(state_size=0, action_size=5)
        
        with pytest.raises(ValueError):
            QTableAgent(state_size=10, action_size=5, learning_rate=1.5)
        
        with pytest.raises(ValueError):
            QTableAgent(state_size=10, action_size=5, discount=1.5)
    
    def test_discretize_state(self):
        """Test state discretization."""
        agent = QTableAgent(state_size=3, action_size=2, num_bins=10)
        state = np.array([0.5, -0.5, 0.0])
        discrete = agent._discretize_state(state)
        
        assert isinstance(discrete, tuple)
        assert len(discrete) == 3
        assert all(isinstance(x, (int, np.integer)) for x in discrete)
    
    def test_choose_action_greedy(self):
        """Test greedy action selection."""
        agent = QTableAgent(state_size=2, action_size=3)
        state = np.array([0.1, 0.2])
        
        action = agent.choose_action(state, epsilon=0.0)
        assert 0 <= action < 3
    
    def test_choose_action_random(self):
        """Test random action selection with epsilon=1."""
        agent = QTableAgent(state_size=2, action_size=3)
        state = np.array([0.1, 0.2])
        
        actions = [agent.choose_action(state, epsilon=1.0) for _ in range(100)]
        # With high probability, we should see multiple different actions
        assert len(set(actions)) > 1
    
    def test_learn(self):
        """Test learning updates Q-values."""
        agent = QTable(state_size=2, action_size=3, learning_rate=0.5, discount=0.9)
        state = np.array([0.1, 0.2])
        next_state = np.array([0.3, 0.4])
        
        metrics = agent.learn(state, action=1, reward=1.0, next_state=next_state, done=False)
        
        assert 'q_value' in metrics
        assert 'td_error' in metrics
        assert len(agent.q_table) > 0
    
    def test_save_load(self):
        """Test saving and loading agent."""
        agent = QTableAgent(state_size=2, action_size=3)
        state = np.array([0.1, 0.2])
        
        # Train a bit to populate Q-table
        for _ in range(10):
            action = agent.choose_action(state)
            agent.learn(state, action, 1.0, state, False)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_agent"
            agent.save(save_path)
            
            # Load into new agent
            new_agent = QTableAgent(state_size=2, action_size=3)
            new_agent.load(save_path)
            
            # Verify Q-tables match
            assert len(new_agent.q_table) == len(agent.q_table)
            for key in agent.q_table:
                assert key in new_agent.q_table
                assert abs(new_agent.q_table[key] - agent.q_table[key]) < 1e-6
    
    def test_choose_action_excluding(self):
        """Test action selection with exclusions."""
        agent = QTableAgent(state_size=2, action_size=5)
        state = np.array([0.1, 0.2])
        
        # Exclude all but one action
        excluded = [0, 1, 2, 3]
        action = agent.choose_action_excluding(state, excluded)
        assert action == 4
        
        # Test error when all actions excluded
        with pytest.raises(ValueError):
            agent.choose_action_excluding(state, list(range(5)))
