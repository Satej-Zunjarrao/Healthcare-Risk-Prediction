"""
Module: Utilities
Author: Satej
Description: This script contains helper functions used across different
             modules, including logging, data validation, and reusable utilities.
"""

import logging
import os
import json
import pandas as pd

# Setup logging
LOG_FILE = "satej_project.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_message(message: str, level: str = "info"):
    """
    Log a message to the project log file.

    Args:
        message (str): Message to log.
        level (str): Log level ('info', 'warning', 'error').
    """
    levels = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
    if level.lower() in levels:
        logging.log(levels[level.lower()], message)
    else:
        logging.info(message)

def validate_data_file(file_path: str) -> bool:
    """
    Validate the existence and readability of a data file.

    Args:
        file_path (str): Path to the data file.

    Returns:
        bool: True if the file exists and is readable, False otherwise.
    """
    if not os.path.exists(file_path):
        log_message(f"File not found: {file_path}", "error")
        return False
    if not os.access(file_path, os.R_OK):
        log_message(f"File is not readable: {file_path}", "error")
        return False
    log_message(f"File validated: {file_path}")
    return True

def save_json(data: dict, file_path: str):
    """
    Save a dictionary as a JSON file.

    Args:
        data (dict): Data to save.
        file_path (str): Path to save the JSON file.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        log_message(f"Data successfully saved to {file_path}")
    except Exception as e:
        log_message(f"Error saving data to {file_path}: {str(e)}", "error")

def load_json(file_path: str) -> dict:
    """
    Load a dictionary from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Loaded data.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        log_message(f"Data successfully loaded from {file_path}")
        return data
    except Exception as e:
        log_message(f"Error loading data from {file_path}: {str(e)}", "error")
        return {}

def print_dataframe_info(df: pd.DataFrame):
    """
    Print and log basic information about a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
    """
    log_message(f"DataFrame Info:\n{df.info()}")
    print(df.info())
    print("\nFirst 5 rows:\n", df.head())

