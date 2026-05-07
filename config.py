"""
Configuration file for the forecasting system
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data configuration
DATA_FILE = "Forecasting Case- Study.xlsx"
DATE_COLUMN = "Date"
TARGET_COLUMN = "Total"
STATE_COLUMN = "State"

# Forecasting configuration
FORECAST_HORIZON = 8  # weeks
VALIDATION_WEEKS = 8
TEST_WEEKS = 8

# Feature engineering
LAG_FEATURES = [1, 7, 14, 30]
ROLLING_WINDOWS = [7, 14, 30]

# Model configuration
MODELS = {
    "SARIMA": {"enabled": True},
    "Prophet": {"enabled": True},
    "XGBoost": {"enabled": True},
    "LSTM": {"enabled": False}  # Disabled temporarily - TensorFlow still installing
}

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Sales Forecasting API"
API_VERSION = "1.0.0"

# Random seed for reproducibility
RANDOM_SEED = 42
