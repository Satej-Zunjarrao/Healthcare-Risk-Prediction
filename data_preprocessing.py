"""
Module: Data Preprocessing
Author: Satej
Description: This script handles data collection, cleaning, and preprocessing
             steps required to prepare healthcare datasets for analysis
             and model training.
"""

import pandas as pd
import numpy as np
import sqlalchemy

# Database configuration
DATABASE_URL = "mysql+pymysql://satej:password@localhost/healthcare_db"

def fetch_data_from_database(query: str, db_url: str = DATABASE_URL) -> pd.DataFrame:
    """
    Fetch data from a SQL database using the provided query.
    
    Args:
        query (str): SQL query string to fetch data.
        db_url (str): Database connection URL.
    
    Returns:
        pd.DataFrame: DataFrame containing the queried data.
    """
    engine = sqlalchemy.create_engine(db_url)
    with engine.connect() as connection:
        return pd.read_sql(query, connection)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning tasks such as handling missing values,
    removing duplicates, and resolving inconsistencies.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Handle missing values
    df.fillna(method='ffill', inplace=True)  # Forward fill for sequential data
    df.dropna(inplace=True)  # Drop rows with any remaining NaNs
    
    # Remove duplicate records
    df.drop_duplicates(inplace=True)
    
    # Example: Standardize medical codes (if applicable)
    if 'medical_code' in df.columns:
        df['medical_code'] = df['medical_code'].str.upper()
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data by normalizing numerical features and
    encoding categorical variables.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Normalize numerical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    
    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

if __name__ == "__main__":
    # SQL query to fetch patient data
    sql_query = "SELECT * FROM patient_data"
    
    # Step 1: Fetch data
    raw_data = fetch_data_from_database(sql_query)
    print(f"Raw data fetched: {raw_data.shape}")
    
    # Step 2: Clean data
    cleaned_data = clean_data(raw_data)
    print(f"Data after cleaning: {cleaned_data.shape}")
    
    # Step 3: Preprocess data
    preprocessed_data = preprocess_data(cleaned_data)
    print(f"Data after preprocessing: {preprocessed_data.shape}")
    
    # Save the preprocessed data for further use
    preprocessed_data.to_csv("satej_preprocessed_data.csv", index=False)
    print("Preprocessed data saved to 'satej_preprocessed_data.csv'")
