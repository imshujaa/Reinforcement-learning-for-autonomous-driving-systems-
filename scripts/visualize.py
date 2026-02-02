"""Visualization script for CaRLeT results."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from carlet.utils.visualization import (
    plot_training_curves,
    plot_comparison,
    create_summary_table
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize CaRLeT results")
    
    parser.add_argument(
        "--metrics-file",
        type=str,
        help="Path to metrics CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.png",
        help="Output plot path"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Smoothing window size"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple experiments"
    )
    parser.add_argument(
        "--metrics",
        action="append",
        help="Metrics files for comparison (format: LABEL=path)"
    )
    
    return parser.parse_args()


def main():
    """Main visualization function."""
    args = parse_args()
    
    if args.compare:
        # Comparison mode
        if not args.metrics:
            print("Error: --metrics required for comparison mode")
            return 1
        
        metrics_files = []
        for item in args.metrics:
            label, path = item.split('=')
            metrics_files.append((label, Path(path)))
        
        print(f"Comparing {len(metrics_files)} experiments...")
        plot_comparison(metrics_files, save_path=Path(args.output), window=args.window)
        
        # Create summary table
        summary = create_summary_table(metrics_files)
        print("\\nSummary Statistics:")
        print(summary.to_string(index=False))
        
        # Save summary
        summary_path = Path(args.output).parent / "summary_statistics.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\\nSummary saved to: {summary_path}")
        
    else:
        # Single experiment mode
        if not args.metrics_file:
            print("Error: --metrics-file required")
            return 1
        
        metrics_path = Path(args.metrics_file)
        if not metrics_path.exists():
            print(f"Error: File not found: {metrics_path}")
            return 1
        
        print(f"Plotting training curves from: {metrics_path}")
        plot_training_curves(
            metrics_path,
            save_path=Path(args.output),
            window=args.window
        )
    
    print(f"\\nVisualization saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
