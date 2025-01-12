"""
Module: Configuration
Author: Satej
Description: This script manages all configuration settings for the project,
             including database connection strings, file paths, and other
             constants.
"""

import os

# Database configurations
DB_USER = "satej"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_NAME = "healthcare_db"
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# File paths
RAW_DATA_FILE = "data/raw_patient_data.csv"
PREPROCESSED_DATA_FILE = "data/satej_preprocessed_data.csv"
FEATURE_ENGINEERED_DATA_FILE = "data/satej_feature_engineered_data.csv"
MODEL_FILE = "models/satej_best_model.pkl"
LOG_FILE = "satej_project.log"

# Flask app configurations
FLASK_HOST = "0.0.0.0"
FLASK_PORT_API = 5000
FLASK_PORT_DASHBOARD = 5001

# Model training configurations
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# Plot configurations
EDA_PLOTS_DIR = "eda_plots/"
CORRELATION_HEATMAP_FILE = os.path.join(EDA_PLOTS_DIR, "correlation_heatmap.png")

# Logging
LOGGING_LEVEL = "INFO"

# Utility constants
DEFAULT_ENCODING = "utf-8"
JSON_SAVE_PATH = "output/satej_config.json"
