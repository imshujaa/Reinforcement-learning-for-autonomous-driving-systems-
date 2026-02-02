"""Unit tests for configuration management."""

import pytest
from pathlib import Path
import tempfile
import yaml

from carlet.utils.config import (
    Config,
    AgentConfig,
    EnvironmentConfig,
    load_config,
    get_default_config
)


class TestConfig:
    """Test suite for configuration management."""
    
    def test_default_config(self):
        """Test loading default configuration."""
        config = get_default_config()
        assert config.environment.name == "highwayMA-v0"
        assert config.agent.type == "qtable"
        assert config.training.episodes == 500
    
    def test_environment_config(self):
        """Test environment configuration validation."""
        # Valid config
        env_config = EnvironmentConfig(name="highwayMA-v0")
        assert env_config.label == "highway"
        assert env_config.features == 5
        assert env_config.num_actions == 5
        
        # Invalid environment
        with pytest.raises(ValueError):
            EnvironmentConfig(name="invalid-env")
    
    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        # Valid config
        agent_config = AgentConfig(type="qtable", learning_rate=0.1, discount=0.95)
        assert agent_config.learning_rate == 0.1
        
        # Invalid learning rate
        with pytest.raises(ValueError):
            AgentConfig(type="qtable", learning_rate=1.5, discount=0.95)
        
        # Invalid discount
        with pytest.raises(ValueError):
            AgentConfig(type="qtable", learning_rate=0.1, discount=1.5)
    
    def test_config_to_yaml(self):
        """Test saving configuration to YAML."""
        config = get_default_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            config.to_yaml(yaml_path)
            
            assert yaml_path.exists()
            
            # Verify can be loaded back
            loaded_config = Config.from_yaml(yaml_path)
            assert loaded_config.environment.name == config.environment.name
    
    def test_config_from_yaml(self):
        """Test loading configuration from YAML."""
        yaml_content = """
experiment_name: "test_experiment"
seed: 42

environment:
  name: "highwayMA-v0"
  render: false

agent:
  type: "qtable"
  learning_rate: 0.2
  discount: 0.9

causal:
  enabled: true
  rollout_steps: 2

training:
  episodes: 100
  repetitions: 1

logging:
  level: "DEBUG"
"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            yaml_path.write_text(yaml_content)
            
            config = Config.from_yaml(yaml_path)
            assert config.experiment_name == "test_experiment"
            assert config.seed == 42
            assert config.agent.learning_rate == 0.2
            assert config.training.episodes == 100
    
    def test_load_config_with_overrides(self):
        """Test config loading with overrides."""
        config = load_config(training__episodes=1000, agent__learning_rate=0.05)
        
        assert config.training.episodes == 1000
        assert config.agent.learning_rate == 0.05
    
    def test_create_directories(self):
        """Test directory creation."""
        config = get_default_config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.training.output_dir = tmpdir
            config.create_directories()
            
            paths = config.get_paths()
            for path in paths.values():
                assert path.exists()
