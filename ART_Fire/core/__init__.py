"""
Art_Fire Core Anomaly Detection Module

This module provides the core anomaly detection algorithms for wildfire detection,
including temporal, spatial, and combined time-range spatial anomaly detection.
"""

from .temporal_anomaly_detector import TemporalAnomalyDetector
from .spatial_anomaly_detector import SpatialAnomalyDetector
from .time_range_spatial_anomaly_detector import TimeRangeSpatialAnomalyDetector

__all__ = ['TemporalAnomalyDetector', 'SpatialAnomalyDetector', 'TimeRangeSpatialAnomalyDetector']

__version__ = '1.0.0'
__author__ = 'Art_Fire Development Team'
__email__ = 'contact@artfire.org'
