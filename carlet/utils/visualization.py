"""Visualization utilities for CaRLeT."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd


def plot_training_curves(
    metrics_file: Path,
    save_path: Optional[Path] = None,
    window: int = 50
) -> None:
    """Plot training curves from metrics file.
    
    Args:
        metrics_file: Path to CSV metrics file
        save_path: Path to save the plot (optional)
        window: Window size for smoothing
    """
    # Load data
    df = pd.read_csv(metrics_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    # Plot 1: Rewards
    ax = axes[0, 0]
    ax.plot(df['episode'], df['avg_reward'], alpha=0.3, label='Raw')
    if len(df) >= window:
        smoothed = df['avg_reward'].rolling(window=window, center=True).mean()
        ax.plot(df['episode'], smoothed, linewidth=2, label=f'Smoothed ({window})')
    ax.fill_between(df['episode'], df['min_reward'], df['max_reward'], alpha=0.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Rewards over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Failure Rate
    ax = axes[0, 1]
    ax.plot(df['episode'], df['fail_rate'], alpha=0.3, label='Raw')
    if len(df) >= window:
        smoothed = df['fail_rate'].rolling(window=window, center=True).mean()
        ax.plot(df['episode'], smoothed, linewidth=2, label=f'Smoothed ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Failure Rate')
    ax.set_title('Failure Rate over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Epsilon
    ax = axes[1, 0]
    ax.plot(df['episode'], df['epsilon'], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Derivative
    ax = axes[1, 1]
    ax.plot(df['episode'], df['avg_derivative'], alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward Derivative')
    ax.set_title('Learning Progress (Derivative)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(
    metrics_files: List[Tuple[str, Path]],
    save_path: Optional[Path] = None,
    window: int = 50
) -> None:
    """Plot comparison of multiple experiments.
    
    Args:
        metrics_files: List of (label, path) tuples
        save_path: Path to save the plot
        window: Smoothing window
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Algorithm Comparison', fontsize=16)
    
    for label, path in metrics_files:
        df = pd.read_csv(path)
        
        # Rewards
        ax = axes[0]
        if len(df) >= window:
            smoothed = df['avg_reward'].rolling(window=window, center=True).mean()
            ax.plot(df['episode'], smoothed, linewidth=2, label=label)
        else:
            ax.plot(df['episode'], df['avg_reward'], linewidth=2, label=label)
        
        # Failure rate
        ax = axes[1]
        if len(df) >= window:
            smoothed = df['fail_rate'].rolling(window=window, center=True).mean()
            ax.plot(df['episode'], smoothed, linewidth=2, label=label)
        else:
            ax.plot(df['episode'], df['fail_rate'], linewidth=2, label=label)
    
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Rewards Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Failure Rate')
    axes[1].set_title('Failure Rate Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_table(metrics_files: List[Tuple[str, Path]]) -> pd.DataFrame:
    """Create summary statistics table.
    
    Args:
        metrics_files: List of (label, path) tuples
        
    Returns:
        DataFrame with summary statistics
    """
    summaries = []
    
    for label, path in metrics_files:
        df = pd.read_csv(path)
        
        # Use last 100 episodes for final performance
        final_df = df.tail(100)
        
        summary = {
            'Algorithm': label,
            'Final Avg Reward': final_df['avg_reward'].mean(),
            'Final Avg Reward Std': final_df['avg_reward'].std(),
            'Best Reward': df['max_reward'].max(),
            'Final Failure Rate': final_df['fail_rate'].mean(),
            'Final Failure Rate Std': final_df['fail_rate'].std(),
        }
        summaries.append(summary)
    
    return pd.DataFrame(summaries)
