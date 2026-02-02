"""Simple training script for CaRLeT."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from carlet.utils.config import load_config
from carlet.utils.logging import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CaRLeT agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/highway.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    
    # Allow overriding any config parameter
    parser.add_argument(
        "--training.episodes",
        type=int,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--causal.enabled",
        type=lambda x: x.lower() == 'true',
        help="Enable/disable causal reasoning"
    )
    parser.add_argument(
        "--agent.learning_rate",
        type=float,
        help="Learning rate"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Build config overrides from command line
    overrides = {}
    for key, value in vars(args).items():
        if '.' in key and value is not None:
            overrides[key.replace('.', '__')] = value
    
    if args.output_dir:
        overrides['training__output_dir'] = args.output_dir
    
    # Load configuration
    config = load_config(args.config, **overrides)
    
    # Setup logger
    logger = setup_logger(
        "carlet_train",
        log_dir=Path(config.logging.log_dir),
        level=config.logging.level
    )
    
    logger.info("="*60)
    logger.info(f"Starting CaRLeT Training: {config.experiment_name}")
    logger.info("="*60)
    logger.info(f"Environment: {config.environment.name}")
    logger.info(f"Agent: {config.agent.type}")
    logger.info(f"Causal Reasoning: {config.causal.enabled}")
    logger.info(f"Episodes: {config.training.episodes}")
    logger.info(f"Repetitions: {config.training.repetitions}")
    logger.info("="*60)
    
    # Create output directories
    config.create_directories()
    
    # Save configuration
    paths = config.get_paths()
    config_save_path = paths['output'] / "config.yaml"
    config.to_yaml(config_save_path)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Note: Full training implementation would go here
    # For now, this demonstrates the structure
    logger.info("\\n" + "="*60)
    logger.info("TRAINING SETUP COMPLETE")
    logger.info("="*60)
    logger.info(f"\\nTo complete the training implementation:")
    logger.info(f"1. Instantiate environment: {config.environment.name}")
    logger.info(f"2. Create {config.agent.type} agent")
    logger.info(f"3. Initialize causal model: {config.causal.enabled}")
    logger.info(f"4. Run training loop for {config.training.episodes} episodes")
    logger.info(f"5. Save results to: {paths['output']}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
