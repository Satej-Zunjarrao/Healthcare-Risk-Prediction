"""
Module: Model Training
Author: Satej
Description: This script trains, validates, and fine-tunes multiple machine learning
             models for healthcare prediction. It includes evaluation metrics
             to select the best-performing model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        LogisticRegression: Trained model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest Classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    """
    Train a Gradient Boosting Classifier.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        GradientBoostingClassifier: Trained model.
    """
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model using accuracy, F1 score, and ROC-AUC.

    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target variable.

    Returns:
        dict: Evaluation metrics (accuracy, F1, and ROC-AUC).
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba)
    }
    return metrics

if __name__ == "__main__":
    # Load feature-engineered data
    data_file = "satej_feature_engineered_data.csv"
    data = pd.read_csv(data_file)
    print(f"Data loaded for model training: {data.shape}")
    
    # Split data into features and target
    X = data.drop(columns=["target"])  # Replace "target" with your actual target column name
    y = data["target"]  # Replace with your actual target column name
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 1: Train Logistic Regression
    logistic_model = train_logistic_regression(X_train, y_train)
    logistic_metrics = evaluate_model(logistic_model, X_test, y_test)
    print(f"Logistic Regression Metrics: {logistic_metrics}")

    # Step 2: Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest Metrics: {rf_metrics}")

    # Step 3: Train Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_metrics = evaluate_model(gb_model, X_test, y_test)
    print(f"Gradient Boosting Metrics: {gb_metrics}")
    
    # Select the best-performing model
    best_model, best_metrics = max(
        [(logistic_model, logistic_metrics), 
         (rf_model, rf_metrics), 
         (gb_model, gb_metrics)],
        key=lambda x: x[1]["ROC-AUC"]
    )
    print(f"Best Model: {type(best_model).__name__} with Metrics: {best_metrics}")
    
    # Save the best model (using joblib)
    import joblib
    joblib.dump(best_model, "satej_best_model.pkl")
    print("Best model saved as 'satej_best_model.pkl'")
