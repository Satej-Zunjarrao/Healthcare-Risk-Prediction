"""
Module: Dashboard Visualization
Author: Satej
Description: This script generates interactive dashboards to visualize healthcare
             predictions, patient risk categories, and trends using Plotly and Flask.
"""

import pandas as pd
from flask import Flask, render_template
import plotly.express as px
import plotly.io as pio

# Initialize Flask app
app = Flask(__name__)

# Load prediction data
DATA_PATH = "satej_feature_engineered_data.csv"
data = pd.read_csv(DATA_PATH)

@app.route("/")
def dashboard():
    """
    Render the main dashboard page.
    """
    # Generate visualizations
    risk_distribution_plot = generate_risk_distribution_plot(data)
    feature_importance_plot = generate_feature_importance_plot()
    
    # Render the dashboard HTML
    return render_template(
        "dashboard.html",
        risk_distribution_plot=risk_distribution_plot,
        feature_importance_plot=feature_importance_plot
    )

def generate_risk_distribution_plot(df: pd.DataFrame):
    """
    Generate a Plotly chart for patient risk category distribution.

    Args:
        df (pd.DataFrame): Dataset containing risk category.

    Returns:
        str: HTML representation of the Plotly chart.
    """
    fig = px.histogram(df, x="target", color="target", title="Risk Category Distribution")
    return pio.to_html(fig, full_html=False)

def generate_feature_importance_plot():
    """
    Generate a Plotly chart for feature importance using model metadata.

    Returns:
        str: HTML representation of the Plotly chart.
    """
    # Dummy feature importance (replace with actual values if available)
    feature_importance = {
        "Age": 0.25,
        "BMI": 0.20,
        "Cholesterol": 0.15,
        "Blood Pressure": 0.10,
        "Glucose Level": 0.30
    }
    
    df_importance = pd.DataFrame(
        list(feature_importance.items()), columns=["Feature", "Importance"]
    )
    
    fig = px.bar(
        df_importance,
        x="Feature",
        y="Importance",
        title="Feature Importance",
        labels={"Importance": "Importance Score"}
    )
    return pio.to_html(fig, full_html=False)

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5001, debug=True)
