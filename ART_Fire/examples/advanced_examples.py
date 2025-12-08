"""
Example usage of Art-Fire anomaly detection system.

This module demonstrates how to use the Art-Fire package for detecting
anomalies in time series data, with examples for different detection methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Import Art-Fire modules
try:
    from art_fire.core.temporal_anomaly_detector import TemporalAnomalyDetector
    from art_fire.core.spatial_anomaly_detector import SpatialAnomalyDetector
    ART_FIRE_AVAILABLE = True
except ImportError:
    print("Art-Fire not available. Please install the package first.")
    ART_FIRE_AVAILABLE = False
    exit(1)


def generate_synthetic_data(n_points: int = 1000, anomaly_rate: float = 0.05) -> tuple:
    """
    Generate synthetic time series data with anomalies.
    
    Args:
        n_points: Number of data points to generate
        anomaly_rate: Proportion of anomalous points
        
    Returns:
        tuple: (timestamps, values, true_anomalies)
    """
    np.random.seed(42)
    
    # Generate timestamps
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]
    
    # Generate normal data with trend and seasonality
    trend = np.linspace(0, 10, n_points)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_points) / 24)  # Daily cycle
    noise = np.random.normal(0, 1, n_points)
    values = trend + seasonal + noise
    
    # Add anomalies
    n_anomalies = int(n_points * anomaly_rate)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    true_anomalies = np.zeros(n_points, dtype=int)
    
    for idx in anomaly_indices:
        # Randomly choose anomaly type
        if np.random.random() < 0.5:
            values[idx] += np.random.uniform(5, 15)  # Spike anomaly
        else:
            values[idx] -= np.random.uniform(5, 15)  # Drop anomaly
        true_anomalies[idx] = 1
    
    return timestamps, values, true_anomalies


def example_temporal_detection():
    """Example of temporal anomaly detection."""
    print("=== Temporal Anomaly Detection Example ===")
    
    # Generate synthetic data
    timestamps, values, true_anomalies = generate_synthetic_data(n_points=500)
    
    # Test different detection methods
    methods = ['spot', '3sigma', 'percentile', 'fixed_window']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, method in enumerate(methods):
        print(f"\nTesting method: {method}")
        
        # Initialize detector
        detector = TemporalAnomalyDetector(
            method=method,
            window_size=50,
            threshold=0.05
        )
        
        # Perform detection
        detected_anomalies = detector.detect(values)
        
        # Calculate metrics
        true_positives = np.sum(true_anomalies & detected_anomalies)
        false_positives = np.sum((1 - true_anomalies) & detected_anomalies)
        false_negatives = np.sum(true_anomalies & (1 - detected_anomalies))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Detected {np.sum(detected_anomalies)} anomalies")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1_score:.3f}")
        
        # Plot results
        ax = axes[i]
        ax.plot(timestamps, values, 'b-', label='Data', alpha=0.7)
        
        # Mark true anomalies
        anomaly_times = [timestamps[j] for j in range(len(values)) if true_anomalies[j]]
        anomaly_values = [values[j] for j in range(len(values)) if true_anomalies[j]]
        ax.scatter(anomaly_times, anomaly_values, c='green', marker='o', s=50, 
                  label='True Anomalies', alpha=0.8)
        
        # Mark detected anomalies
        detected_times = [timestamps[j] for j in range(len(values)) if detected_anomalies[j]]
        detected_values = [values[j] for j in range(len(values)) if detected_anomalies[j]]
        ax.scatter(detected_times, detected_values, c='red', marker='x', s=50, 
                  label='Detected Anomalies', alpha=0.8)
        
        ax.set_title(f'Method: {method} (F1: {f1_score:.3f})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig('temporal_detection_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as 'temporal_detection_comparison.png'")


def example_parameter_tuning():
    """Example of parameter tuning for optimal detection."""
    print("\n=== Parameter Tuning Example ===")
    
    # Generate data
    timestamps, values, true_anomalies = generate_synthetic_data(n_points=1000)
    
    # Test different window sizes
    window_sizes = [20, 50, 100, 200]
    best_f1 = 0
    best_params = {}
    
    for window_size in window_sizes:
        detector = TemporalAnomalyDetector(
            method='3sigma',
            window_size=window_size,
            threshold=0.05
        )
        
        detected_anomalies = detector.detect(values)
        
        # Calculate F1 score
        true_positives = np.sum(true_anomalies & detected_anomalies)
        false_positives = np.sum((1 - true_anomalies) & detected_anomalies)
        false_negatives = np.sum(true_anomalies & (1 - detected_anomalies))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Window size {window_size}: F1 = {f1_score:.3f}")
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_params = {'window_size': window_size}
    
    print(f"\nBest parameters: {best_params} (F1 = {best_f1:.3f})")


def example_real_world_usage():
    """Example of real-world usage with environmental data."""
    print("\n=== Real-World Usage Example ===")
    
    # This example assumes you have environmental data
    # For demonstration, we'll use synthetic data that simulates
    # temperature readings from a wildfire monitoring station
    
    # Generate temperature data with fire event
    np.random.seed(123)
    n_hours = 24 * 7  # One week of hourly data
    
    # Normal temperature pattern
    base_temp = 20  # Base temperature in Celsius
    daily_cycle = 10 * np.sin(2 * np.pi * np.arange(n_hours) / 24)
    noise = np.random.normal(0, 2, n_hours)
    
    temperatures = base_temp + daily_cycle + noise
    
    # Simulate fire event (sudden temperature spike)
    fire_start = 24 * 3 + 12  # Day 3, noon
    fire_duration = 8  # 8 hours
    fire_temp_increase = 30  # 30°C increase
    
    for i in range(fire_duration):
        if fire_start + i < n_hours:
            temperatures[fire_start + i] += fire_temp_increase * np.exp(-i / 3)
    
    # Create detector for wildfire monitoring
    detector = TemporalAnomalyDetector(
        method='spot',
        window_size=48,  # 2-day window
        threshold=0.01   # Strict threshold for early warning
    )
    
    # Perform detection
    anomalies = detector.detect(temperatures)
    
    # Analyze results
    anomaly_indices = np.where(anomalies == 1)[0]
    
    if len(anomaly_indices) > 0:
        print(f"Fire event detected!")
        print(f"Anomalies found at hours: {anomaly_indices[:10]}")  # First 10
        print(f"Peak temperature: {np.max(temperatures):.1f}°C")
        print(f"Anomaly rate: {len(anomaly_indices)/len(temperatures)*100:.1f}%")
        
        # Simple fire severity assessment
        max_temp = np.max(temperatures)
        if max_temp > 60:
            severity = "HIGH"
        elif max_temp > 45:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        print(f"Estimated fire severity: {severity}")
    else:
        print("No fire events detected")


def main():
    """Run all examples."""
    print("Art-Fire Anomaly Detection Examples")
    print("=" * 40)
    
    if not ART_FIRE_AVAILABLE:
        print("Error: Art-Fire package not available.")
        return
    
    # Run examples
    example_temporal_detection()
    example_parameter_tuning()
    example_real_world_usage()
    
    print("\n" + "=" * 40)
    print("All examples completed successfully!")
    print("Check the generated plots and results.")


if __name__ == "__main__":
    main()