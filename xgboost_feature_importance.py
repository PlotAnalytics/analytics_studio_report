#!/usr/bin/env python3
"""
XGBoost Feature Importance Analysis for YouTube Analytics
--------------------------------------------------------
This script uses XGBoost to analyze feature importance
for predicting engaged views in YouTube analytics data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

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

def perform_xgboost_analysis(df):
    print("Performing XGBoost analysis...")

    # Select features for analysis
    features = [
        'Likes', 'Subscribers', 'Watch time (hours)', 'Comments added',
        'Average percentage viewed (%)', 'Impressions click-through rate (%)',
        'Duration', 'Stayed to watch (%)'
    ]

    # Target variable
    target = 'Engaged views'

    # Drop rows with missing values
    analysis_df = df[features + [target]].dropna()
    print(f"Using {len(analysis_df)} rows for analysis after dropping NAs")

    # Split data into features and target
    X = analysis_df[features]
    y = analysis_df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"XGBoost R-squared: {r2:.4f}")
    print(f"XGBoost RMSE: {rmse:.2f}")

    # Get built-in feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)

    print("XGBoost built-in feature importance:")
    print(importance_df)

    # Create built-in feature importance visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('XGBoost Feature Importance for Predicting Engaged Views', fontsize=16)
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)

    # Add value labels
    for i, v in enumerate(importance_df['Importance']):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('statistical_analysis/figures/xgboost_importance.png', dpi=300)
    plt.close()

    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    perm_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': perm_importance.importances_mean
    })
    perm_importance_df = perm_importance_df.sort_values('Importance', ascending=False)

    print("Permutation feature importance:")
    print(perm_importance_df)

    # Create permutation importance visualization
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=perm_importance_df, palette='viridis')
    plt.title('Permutation Feature Importance for Predicting Engaged Views', fontsize=16)
    plt.xlabel('Increase in MSE when Feature is Permuted', fontsize=14)
    plt.ylabel('Feature', fontsize=14)

    # Add value labels
    for i, v in enumerate(perm_importance_df['Importance']):
        plt.text(v + 0.01, i, f'{v:.2e}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('statistical_analysis/figures/permutation_importance.png', dpi=300)
    plt.close()

    return model, r2, rmse, importance_df, perm_importance_df

# Main function
def main():
    print("Starting XGBoost feature importance analysis...")
    df = load_data()
    df_clean = clean_data(df)

    print("Generating feature importance visualizations using XGBoost...")
    _, r2, rmse, xgb_importance, perm_importance = perform_xgboost_analysis(df_clean)

    print("XGBoost feature importance analysis complete!")
    print(f"Model R-squared: {r2:.4f}")
    print(f"Model RMSE: {rmse:.2f}")
    print("\nTop 5 features by XGBoost importance:")
    print(xgb_importance.head(5))
    print("\nTop 5 features by permutation importance:")
    print(perm_importance.head(5))

if __name__ == "__main__":
    main()
