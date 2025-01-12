"""
Module: Feature Engineering
Author: Satej
Description: This script handles feature engineering tasks, including creating
             new features, normalizing numerical data, and encoding
             categorical variables for better model performance.
"""

import pandas as pd
import numpy as np

def create_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features representing risk scores based on clinical thresholds.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new risk score features added.
    """
    # Example: Create a binary risk score for high blood pressure
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        df['high_bp_risk'] = ((df['systolic_bp'] > 140) | (df['diastolic_bp'] > 90)).astype(int)

    # Example: Risk score based on high glucose levels
    if 'glucose_level' in df.columns:
        df['high_glucose_risk'] = (df['glucose_level'] > 126).astype(int)

    return df

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numerical features to improve model performance.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with normalized numerical features.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features using one-hot encoding.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df

if __name__ == "__main__":
    # Load preprocessed data
    data_file = "satej_preprocessed_data.csv"
    data = pd.read_csv(data_file)
    print(f"Data loaded for feature engineering: {data.shape}")
    
    # Step 1: Create risk score features
    data = create_risk_scores(data)
    print("Risk scores created.")

    # Step 2: Normalize numerical features
    data = normalize_features(data)
    print("Numerical features normalized.")

    # Step 3: Encode categorical features
    data = encode_categorical_features(data)
    print("Categorical features encoded.")

    # Save the engineered data
    data.to_csv("satej_feature_engineered_data.csv", index=False)
    print("Feature-engineered data saved to 'satej_feature_engineered_data.csv'")
