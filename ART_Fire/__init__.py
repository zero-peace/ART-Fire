"""
Art-Fire: Advanced Temporal Anomaly Detection for Wildfire Monitoring

A comprehensive Python package for detecting temporal anomalies in environmental data,
specifically designed for wildfire monitoring and early warning systems.

This package provides multiple anomaly detection algorithms including:
- Temporal Anomaly Detection (SPOT, 3-sigma, percentile-based)
- Spatial Anomaly Detection
- Time-Range Spatial Anomaly Detection

Author: Art-Fire Team
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Art-Fire Team"
__email__ = "contact@art-fire.org"
__license__ = "MIT"

# Import main classes for easy access
try:
    from .core.temporal_anomaly_detector import TemporalAnomalyDetector
    from .core.spatial_anomaly_detector import SpatialAnomalyDetector
    from .core.time_range_spatial_anomaly_detector import TimeRangeSpatialAnomalyDetector
except ImportError as e:
    # Handle cases where dependencies might not be available during installation
    import warnings
    warnings.warn(f"Some dependencies not available during import: {e}")
    TemporalAnomalyDetector = None
    SpatialAnomalyDetector = None
    TimeRangeSpatialAnomalyDetector = None

__all__ = [
    "TemporalAnomalyDetector",
    "SpatialAnomalyDetector", 
    "TimeRangeSpatialAnomalyDetector",
    "__version__",
    "__author__",
    "__email__",
    "__license__"
]