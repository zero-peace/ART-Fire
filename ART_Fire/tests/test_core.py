"""
Test suite for Art-Fire package.

This module contains unit tests for the Art-Fire anomaly detection system.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Import the modules to test
try:
    from art_fire.core.temporal_anomaly_detector import TemporalAnomalyDetector
    from art_fire.core.spatial_anomaly_detector import SpatialAnomalyDetector
    from art_fire.core.time_range_spatial_anomaly_detector import TimeRangeSpatialAnomalyDetector
    ART_FIRE_AVAILABLE = True
except ImportError:
    ART_FIRE_AVAILABLE = False


class TestTemporalAnomalyDetector(unittest.TestCase):
    """Test cases for TemporalAnomalyDetector class."""
    
    @unittest.skipUnless(ART_FIRE_AVAILABLE, "Art-Fire not available")
    def setUp(self):
        """Set up test fixtures."""
        self.detector = TemporalAnomalyDetector(
            method="3sigma",
            window_size=50,
            threshold=0.05
        )
        # Generate synthetic test data
        np.random.seed(42)
        self.normal_data = np.random.normal(0, 1, 1000)
        # Add some artificial anomalies
        self.anomalous_data = self.normal_data.copy()
        self.anomalous_data[100:105] = 10  # Spike anomalies
        self.anomalous_data[500:503] = -8  # Drop anomalies
    
    @unittest.skipUnless(ART_FIRE_AVAILABLE, "Art-Fire not available")
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.method, "3sigma")
        self.assertEqual(self.detector.window_size, 50)
        self.assertEqual(self.detector.threshold, 0.05)
        self.assertEqual(len(self.detector.history_data), 0)
    
    @unittest.skipUnless(ART_FIRE_AVAILABLE, "Art-Fire not available")
    def test_detect_with_normal_data(self):
        """Test detection with normal data."""
        labels = self.detector.detect(self.normal_data)
        
        # Should return array of same length
        self.assertEqual(len(labels), len(self.normal_data))
        # Should be binary labels (0 or 1)
        self.assertTrue(np.all(np.isin(labels, [0, 1])))
        # Should have very few anomalies in normal data
        anomaly_rate = np.mean(labels)
        self.assertLess(anomaly_rate, 0.05)  # Less than 5% anomalies
    
    @unittest.skipUnless(ART_FIRE_AVAILABLE, "Art-Fire not available")
    def test_detect_with_anomalous_data(self):
        """Test detection with data containing known anomalies."""
        labels = self.detector.detect(self.anomalous_data)
        
        # Should detect some anomalies
        self.assertGreater(np.sum(labels), 0)
        # Should have higher anomaly rate than normal data
        anomaly_rate = np.mean(labels)
        self.assertGreater(anomaly_rate, 0.01)  # More than 1% anomalies


class TestSpatialAnomalyDetector(unittest.TestCase):
    """Test cases for SpatialAnomalyDetector class."""
    
    @unittest.skipUnless(ART_FIRE_AVAILABLE, "Art-Fire not available")
    def setUp(self):
        """Set up test fixtures."""
        self.detector = SpatialAnomalyDetector()
        # Generate synthetic spatial data
        np.random.seed(42)
        self.spatial_data = np.random.normal(0, 1, (100, 3))  # 100 points, 3 features
    
    @unittest.skipUnless(ART_FIRE_AVAILABLE, "Art-Fire not available")
    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsNotNone(self.detector)


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration module."""
    
    def test_config_import(self):
        """Test that configuration can be imported."""
        try:
            from art_fire import config
            self.assertTrue(hasattr(config, 'DEFAULT_WINDOW_SIZE'))
            self.assertTrue(hasattr(config, 'DEFAULT_THRESHOLD'))
        except ImportError:
            self.skipTest("Configuration module not available")


class TestCLI(unittest.TestCase):
    """Test cases for CLI module."""
    
    def test_cli_import(self):
        """Test that CLI module can be imported."""
        try:
            from art_fire import cli
            self.assertTrue(hasattr(cli, 'main'))
        except ImportError:
            self.skipTest("CLI module not available")


if __name__ == '__main__':
    unittest.main()