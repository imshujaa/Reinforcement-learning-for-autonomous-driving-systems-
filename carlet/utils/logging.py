"""Enhanced logging utilities for CaRLeT."""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """Setup a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to add console handler
        file: Whether to add file handler
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """Logger for experiment metrics and progress."""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        """Initialize experiment logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics_{timestamp}.csv"
        
        # Initialize metrics file
        with open(self.metrics_file, 'w') as f:
            f.write("episode,avg_reward,max_reward,min_reward,fail_rate,avg_derivative,epsilon\n")
    
    def log_episode(
        self,
        episode: int,
        avg_reward: float,
        max_reward: float,
        min_reward: float,
        fail_rate: float,
        avg_derivative: float = 0.0,
        epsilon: float = 0.0
    ) -> None:
        """Log episode metrics.
        
        Args:
            episode: Episode number
            avg_reward: Average reward
            max_reward: Maximum reward
            min_reward: Minimum reward
            fail_rate: Failure rate
            avg_derivative: Average reward derivative
            epsilon: Current epsilon value
        """
        with open(self.metrics_file, 'a') as f:
            f.write(f"{episode},{avg_reward:.4f},{max_reward:.4f},{min_reward:.4f},"
                   f"{fail_rate:.4f},{avg_derivative:.6f},{epsilon:.4f}\n")
