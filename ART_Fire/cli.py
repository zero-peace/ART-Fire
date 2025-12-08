"""
CLI module for Art-Fire package.

This module provides command-line interface functionality for the
Art-Fire anomaly detection system.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional

from .core.temporal_anomaly_detector import TemporalAnomalyDetector
from .config import DEFAULT_WINDOW_SIZE, DEFAULT_THRESHOLD, LOG_LEVEL


def detect_anomalies(input_file: str, output_file: Optional[str] = None, 
                    method: str = "spot", window_size: int = DEFAULT_WINDOW_SIZE,
                    threshold: float = DEFAULT_THRESHOLD) -> None:
    """
    Detect anomalies in time series data.
    
    Args:
        input_file: Path to input CSV file containing time series data
        output_file: Path to output CSV file for anomaly results
        method: Detection method ('spot', '3sigma', 'percentile', 'fixed_window')
        window_size: Size of sliding window for detection
        threshold: Detection threshold parameter
    """
    try:
        # Load data
        print(f"Loading data from {input_file}")
        data = pd.read_csv(input_file)
        
        # Assume the last column contains the time series values
        # This can be made more flexible in the future
        if len(data.columns) == 1:
            values = data.iloc[:, 0].values
        else:
            # Use the last numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in input data")
            values = data[numeric_cols[-1]].values
        
        # Initialize detector
        detector = TemporalAnomalyDetector(
            method=method,
            window_size=window_size,
            threshold=threshold
        )
        
        # Perform detection
        print(f"Performing anomaly detection using {method} method...")
        anomalies = detector.detect(values)
        
        # Prepare output
        result_df = pd.DataFrame({
            'index': range(len(values)),
            'value': values,
            'is_anomaly': anomalies
        })
        
        # Save results
        if output_file:
            result_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        else:
            # Print summary
            anomaly_count = np.sum(anomalies)
            print(f"Detection complete!")
            print(f"Total data points: {len(values)}")
            print(f"Anomalies detected: {anomaly_count}")
            print(f"Anomaly rate: {anomaly_count/len(values)*100:.2f}%")
            
            # Show first few anomalies
            anomaly_indices = np.where(anomalies == 1)[0]
            if len(anomaly_indices) > 0:
                print(f"First 10 anomalies at indices: {anomaly_indices[:10]}")
            
    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Art-Fire: Advanced Temporal Anomaly Detection for Wildfire Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic anomaly detection
  art-fire detect data.csv
  
  # Use 3-sigma method with custom parameters
  art-fire detect data.csv --method 3sigma --window-size 50 --output results.csv
  
  # Use percentile method with custom threshold
  art-fire detect data.csv --method percentile --threshold 0.01
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect anomalies in time series data')
    detect_parser.add_argument('input', help='Input CSV file containing time series data')
    detect_parser.add_argument('--output', '-o', help='Output CSV file for results')
    detect_parser.add_argument('--method', '-m', 
                             choices=['spot', '3sigma', 'percentile', 'fixed_window'],
                             default='spot',
                             help='Anomaly detection method (default: spot)')
    detect_parser.add_argument('--window-size', '-w', 
                             type=int, 
                             default=DEFAULT_WINDOW_SIZE,
                             help=f'Sliding window size (default: {DEFAULT_WINDOW_SIZE})')
    detect_parser.add_argument('--threshold', '-t', 
                             type=float, 
                             default=DEFAULT_THRESHOLD,
                             help=f'Detection threshold (default: {DEFAULT_THRESHOLD})')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'detect':
        detect_anomalies(
            input_file=args.input,
            output_file=args.output,
            method=args.method,
            window_size=args.window_size,
            threshold=args.threshold
        )
    elif args.command == 'version':
        from . import __version__
        print(f"Art-Fire version {__version__}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()