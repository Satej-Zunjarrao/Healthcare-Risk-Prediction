"""
Module: EDA and Visualization
Author: Satej
Description: This script performs exploratory data analysis (EDA) on the 
             healthcare dataset, uncovering insights and visualizing 
             relationships between features and target variables.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load preprocessed data from a CSV file.
    
    Args:
        file_path (str): Path to the preprocessed data file.
    
    Returns:
        pd.DataFrame: DataFrame containing the preprocessed data.
    """
    return pd.read_csv(file_path)

def plot_feature_distributions(df: pd.DataFrame, output_dir: str = "eda_plots/"):
    """
    Plot distributions of numerical features.
    
    Args:
        df (pd.DataFrame): DataFrame containing preprocessed data.
        output_dir (str): Directory to save the plots.
    """
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"{output_dir}{col}_distribution.png")
        plt.close()

def correlation_heatmap(df: pd.DataFrame, output_path: str = "eda_plots/correlation_heatmap.png"):
    """
    Generate and save a correlation heatmap of the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing preprocessed data.
        output_path (str): Path to save the heatmap image.
    """
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    # Load preprocessed data
    data_file = "satej_preprocessed_data.csv"
    data = load_data(data_file)
    print(f"Data loaded: {data.shape}")
    
    # Create output directory for plots
    import os
    os.makedirs("eda_plots", exist_ok=True)
    
    # Step 1: Plot feature distributions
    plot_feature_distributions(data)
    print("Feature distribution plots saved in 'eda_plots/'")
    
    # Step 2: Generate correlation heatmap
    correlation_heatmap(data)
    print("Correlation heatmap saved as 'eda_plots/correlation_heatmap.png'")
