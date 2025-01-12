# Healthcare Risk Prediction System

## Overview
The **Healthcare Risk Prediction System** is a Python-based solution designed to predict the likelihood of patients developing chronic health conditions such as diabetes and heart disease. The system leverages machine learning models to analyze patient data, providing healthcare providers with actionable insights to support early diagnosis and personalized treatment plans.

This project includes a modular and scalable pipeline for data collection, preprocessing, exploratory analysis, feature engineering, model training, deployment, and visualization.

---

## Key Features
- **Data Collection**: Extracts patient records from SQL databases and CSV files.
- **Data Preprocessing**: Cleans and standardizes data, handling missing values and outliers.
- **Exploratory Data Analysis (EDA)**: Visualizes correlations between risk factors and disease occurrence.
- **Feature Engineering**: Creates new features (e.g., risk scores) and encodes categorical variables.
- **Model Training**: Trains and fine-tunes multiple ML models, including Logistic Regression and Gradient Boosting.
- **Model Deployment**: Deploys the best-performing model as a Flask API for real-time predictions.
- **Visualization**: Creates dashboards to display patient risk categories and key predictors.
- **Automation**: Supports reusable and modular components for scalable workflows.

---

## Directory Structure
```plaintext
project/
│
├── data_preprocessing.py        # Handles data collection and cleaning
├── eda_visualization.py         # Performs exploratory data analysis and generates visualizations
├── feature_engineering.py       # Engineers new features and preprocesses data
├── model_training.py            # Trains, validates, and fine-tunes ML models
├── model_deployment.py          # Deploys the trained model as a Flask API
├── dashboard_visualization.py   # Generates dashboards for healthcare insights
├── config.py                    # Stores reusable configurations and constants
├── utils.py                     # Provides helper functions for logging, validation, etc.
├── README.md                    # Project documentation
```

## Modules

### 1. data_preprocessing.py
- Fetches patient records from SQL databases and CSV files.
- Cleans data by handling missing values, outliers, and duplicates.
- Saves preprocessed data for analysis.

### 2. eda_visualization.py
- Generates visualizations of patient demographics and health metrics.
- Creates correlation heatmaps and feature distribution plots.
- Provides insights into key risk factors.

### 3. feature_engineering.py
- Creates new features (e.g., high blood pressure and glucose risk scores).
- Normalizes numerical features and encodes categorical variables.
- Outputs feature-engineered data for model training.

### 4. model_training.py
- Trains and fine-tunes multiple models, including Logistic Regression, Random Forest, and Gradient Boosting.
- Evaluates models using metrics like ROC-AUC, accuracy, and F1 score.
- Selects and saves the best-performing model.

### 5. model_deployment.py
- Deploys the best-performing model as a RESTful Flask API.
- Provides real-time predictions for integration with healthcare systems.

### 6. dashboard_visualization.py
- Creates dashboards to visualize patient risk categories and feature importance.
- Displays interactive plots for stakeholder insights.

### 7. config.py
- Centralized configuration file for database, file paths, and logging.
- Stores reusable constants for model training and deployment.

### 8. utils.py
- Provides utility functions for logging, data validation, and file handling.
- Includes reusable methods for consistent operations across modules.

---

## Contact

For queries or collaboration, feel free to reach out:

- **Name**: Satej Zunjarrao  
- **Email**: zsatej1028@gmail.com

