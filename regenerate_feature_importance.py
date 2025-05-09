#!/usr/bin/env python3
"""
Regenerate Feature Importance Chart for YouTube Analytics Statistical Report
------------------------------------------------------------------------
This script regenerates the feature importance visualization using the actual CSV data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set style for plots
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)
sns.set_style("whitegrid")

# Create output directory for visualizations
os.makedirs('statistical_analysis/figures', exist_ok=True)

# Load the data
def load_data():
    print("Loading data from youtube_analytics_master.csv...")
    # Load the original data
    df = pd.read_csv('youtube_analytics_master.csv')
    print(f"Loaded {len(df)} rows of data")
    return df

# Clean data function
def clean_data(df):
    print("Cleaning data...")
    # Make a copy to avoid warnings
    df = df.copy()

    # Convert duration to numeric (it's in seconds)
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

    # Convert percentage columns to numeric
    percentage_cols = ['Stayed to watch (%)', 'Average percentage viewed (%)', 'Impressions click-through rate (%)']
    for col in percentage_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert engagement metrics to numeric
    engagement_cols = ['Comments added', 'Likes', 'Engaged views', 'Watch time (hours)', 'Subscribers', 'Impressions']
    for col in engagement_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Data cleaning complete")
    return df

def perform_regression_analysis(df):
    print("Performing regression analysis...")
    
    # Select features for regression
    features = [
        'Likes', 'Subscribers', 'Watch time (hours)', 'Comments added',
        'Average percentage viewed (%)', 'Impressions click-through rate (%)',
        'Duration', 'Stayed to watch (%)'
    ]
    
    # Target variable
    target = 'Engaged views'
    
    # Drop rows with missing values
    regression_df = df[features + [target]].dropna()
    print(f"Using {len(regression_df)} rows for regression after dropping NAs")
    
    # Split data into features and target
    X = regression_df[features]
    y = regression_df[target]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Get feature importance (standardized coefficients)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.coef_
    })
    
    # Sort by absolute importance
    feature_importance['Abs_Importance'] = np.abs(feature_importance['Importance'])
    feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)
    
    print("Feature importance:")
    print(feature_importance[['Feature', 'Importance']])
    
    # Create feature importance visualization
    plt.figure(figsize=(12, 8))
    colors = ['green' if x > 0 else 'red' for x in feature_importance['Importance']]
    
    # Plot horizontal bars
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
    
    # Add value labels
    for i, value in enumerate(feature_importance['Importance']):
        plt.text(
            value + (0.01 * np.sign(value) * max(abs(feature_importance['Importance']))),
            i,
            f'{value:.2f}',
            va='center',
            fontweight='bold',
            fontsize=10,
            color='black'
        )
    
    plt.title('Feature Importance for Predicting Engaged Views', fontsize=16)
    plt.xlabel('Standardized Coefficient (Impact on Engaged Views)', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('statistical_analysis/figures/feature_importance_new.png', dpi=300)
    plt.close()
    
    print("Feature importance visualization saved")
    return feature_importance

# Main function
def main():
    print("Starting feature importance regeneration process...")
    df = load_data()
    df_clean = clean_data(df)
    
    print("Generating feature importance visualization using actual CSV data...")
    feature_importance = perform_regression_analysis(df_clean)
    
    print("Feature importance visualization regenerated successfully using actual CSV data!")

if __name__ == "__main__":
    main()
