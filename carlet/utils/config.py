"""Configuration management for CaRLeT using Pydantic for validation."""

from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import yaml
import os


class AgentConfig(BaseModel):
    """Configuration for RL agents."""
    
    type: str = Field(..., description="Agent type: qtable, dqn, ppo, sac")
    learning_rate: float = Field(0.1, gt=0, le=1)
    discount: float = Field(0.95, ge=0, lt=1)
    
    # Q-Table specific
    num_bins: Optional[int] = Field(10, ge=2)
    
    # DQN specific
    batch_size: Optional[int] = Field(32, gt=0)
    buffer_size: Optional[int] = Field(2000, gt=0)
    gradient_steps: Optional[int] = Field(1, gt=0)
    target_update_freq: Optional[int] = Field(10, gt=0)


class CausalConfig(BaseModel):
    """Configuration for causal reasoning."""
    
    enabled: bool = True
    discovery_method: str = Field("prior_knowledge", description="Method: prior_knowledge or causal_discovery")
    rollout_steps: int = Field(1, ge=1, description="Planning depth")
    simulations: int = Field(20, ge=1, description="Number of counterfactual simulations")
    num_clusters: int = Field(100, ge=1, description="K-means clusters for transition selection")
    plan_after: int = Field(1, ge=1, description="Plan after N real episodes")
    actions_to_plan: int = Field(1, ge=1, description="Planning breadth")
    start_planning_at: int = Field(50, ge=0, description="Episode to start planning")
    update_model_every: int = Field(100, ge=1, description="Causal model update frequency")


class EnvironmentConfig(BaseModel):
    """Configuration for the environment."""
    
    name: str = Field(..., description="Environment ID: highwayMA-v0, intersectionMA-v0")
    render: bool = False
    
    @field_validator('name')
    @classmethod
    def validate_env_name(cls, v):
        valid = ['highwayMA-v0', 'intersectionMA-v0']
        if v not in valid:
            raise ValueError(f"Environment must be one of {valid}, got {v}")
        return v
    
    @property
    def label(self) -> str:
        """Get short environment label."""
        return "highway" if "highway" in self.name.lower() else "intersection"
    
    @property
    def features(self) -> int:
        """Get number of features per vehicle."""
        return 5 if self.label == "highway" else 7
    
    @property
    def num_vehicles(self) -> int:
        """Get number of vehicles."""
        return 2
    
    @property
    def num_actions(self) -> int:
        """Get number of actions."""
        return 5 if self.label == "highway" else 3
    
    @property
    def state_size(self) -> int:
        """Get total state size."""
        return self.num_vehicles * self.features


class TrainingConfig(BaseModel):
    """Configuration for training."""
    
    episodes: int = Field(500, gt=0, description="Number of training episodes")
    repetitions: int = Field(10, gt=0, description="Number of experimental repetitions")
    show_every: int = Field(1, gt=0, description="Render frequency")
    update_every: int = Field(10, gt=0, description="Logging frequency")
    
    # Exploration
    start_epsilon: float = Field(1.0, ge=0, le=1)
    end_epsilon: float = Field(0.1, ge=0, le=1)
    epsilon_decay_episodes: Optional[int] = None  # If None, use episodes // 3
    stop_with_epsilon: bool = True  # Stop planning when epsilon reaches minimum
    
    # Smoothing
    smooth_window: int = Field(50, gt=0, description="Window for reward smoothing")
    
    # Paths
    ego_model_path: Optional[str] = None
    output_dir: str = "output"
    models_dir: str = "models"
    
    @field_validator('end_epsilon')
    @classmethod
    def validate_epsilon(cls, v, info):
        if 'start_epsilon' in info.data and v > info.data['start_epsilon']:
            raise ValueError("end_epsilon must be <= start_epsilon")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging and monitoring."""
    
    level: str = Field("INFO", description="Logging level")
    log_dir: str = "output/logs"
    tensorboard: bool = True
    mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        valid = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid:
            raise ValueError(f"Log level must be one of {valid}")
        return v.upper()


class Config(BaseModel):
    """Main configuration for CaRLeT experiments."""
    
    experiment_name: str = "carlet_experiment"
    seed: Optional[int] = None
    
    environment: EnvironmentConfig
    agent: AgentConfig
    causal: CausalConfig
    training: TrainingConfig
    logging: LoggingConfig
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Expand environment variables
        data = cls._expand_env_vars(data)
        
        return cls(**data)
    
    @staticmethod
    def _expand_env_vars(data: Any) -> Any:
        """Recursively expand environment variables in config."""
        if isinstance(data, dict):
            return {k: Config._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Config._expand_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith('$'):
            env_var = data[1:]
            return os.getenv(env_var, data)
        return data
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    def get_paths(self) -> Dict[str, Path]:
        """Get all relevant paths for the experiment.
        
        Returns:
            Dictionary of path names to Path objects
        """
        base = Path(self.training.output_dir)
        env_label = self.environment.label
        
        return {
            "output": base,
            "data": base / "data" / env_label,
            "logs": Path(self.logging.log_dir) / env_label,
            "models": Path(self.training.models_dir),
            "checkpoints": base / "checkpoints" / env_label,
            "tensorboard": base / "tensorboard" / env_label,
            "causal_graphs": base / "causal_graphs" / env_label,
        }
    
    def create_directories(self) -> None:
        """Create all necessary directories for the experiment."""
        paths = self.get_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str] = None, **overrides) -> Config:
    """Load configuration with optional overrides.
    
    Args:
        config_path: Path to YAML config file (optional)
        **overrides: Keyword arguments to override config values
        
    Returns:
        Config object
        
    Example:
        >>> config = load_config('configs/highway.yaml', training__episodes=1000)
    """
    if config_path:
        config = Config.from_yaml(Path(config_path))
    else:
        # Load default configuration
        config = get_default_config()
    
    # Apply overrides (supports nested keys with __)
    if overrides:
        config_dict = config.model_dump()
        for key, value in overrides.items():
            keys = key.split('__')
            d = config_dict
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
        config = Config(**config_dict)
    
    return config


def get_default_config() -> Config:
    """Get default configuration.
    
    Returns:
        Default Config object
    """
    return Config(
        experiment_name="carlet_default",
        environment=EnvironmentConfig(name="highwayMA-v0"),
        agent=AgentConfig(type="qtable"),
        causal=CausalConfig(),
        training=TrainingConfig(),
        logging=LoggingConfig(),
    )
