"""
Configuration file for Art-Fire package.

This module contains default configuration values and settings for the
Art-Fire anomaly detection system.
"""

import os
from pathlib import Path

# Package information
PACKAGE_NAME = "art-fire"
VERSION = "0.1.0"
AUTHOR = "Art-Fire Team"
EMAIL = "contact@art-fire.org"

# Default paths
DEFAULT_DATA_DIR = Path.home() / ".art-fire" / "data"
DEFAULT_LOG_DIR = Path.home() / ".art-fire" / "logs"
DEFAULT_CONFIG_DIR = Path.home() / ".art-fire" / "config"

# Create directories if they don't exist
for dir_path in [DEFAULT_DATA_DIR, DEFAULT_LOG_DIR, DEFAULT_CONFIG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Anomaly detection settings
DEFAULT_WINDOW_SIZE = 100
DEFAULT_THRESHOLD = 0.05
DEFAULT_CONFIDENCE_LEVEL = 0.95

# SPOT algorithm parameters
DEFAULT_SPOT_Q = 1e-4  # Risk parameter
DEFAULT_SPOT_D = 10    # Depth parameter

# 3-sigma method parameters
DEFAULT_SIGMA_MULTIPLIER = 3.0

# Percentile method parameters
DEFAULT_LOWER_PERCENTILE = 1
DEFAULT_UPPER_PERCENTILE = 99

# Logging configuration
LOG_LEVEL = os.getenv("ART_FIRE_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

# Data processing settings
MAX_DATA_POINTS = 100000
MIN_DATA_POINTS = 10
DEFAULT_BATCH_SIZE = 1000

# Performance settings
ENABLE_PARALLEL_PROCESSING = True
MAX_WORKERS = os.cpu_count() or 4
MEMORY_LIMIT_MB = 2048  # 2GB default memory limit

# Visualization settings
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 300
DEFAULT_COLOR_PALETTE = "viridis"

# Export settings
DEFAULT_EXPORT_FORMAT = "csv"
SUPPORTED_EXPORT_FORMATS = ["csv", "json", "parquet", "hdf5"]

# API settings (if using as a service)
API_HOST = os.getenv("ART_FIRE_API_HOST", "localhost")
API_PORT = int(os.getenv("ART_FIRE_API_PORT", "8000"))
API_DEBUG = os.getenv("ART_FIRE_API_DEBUG", "False").lower() == "true"