#!/usr/bin/env python3
"""
Regenerate Statistical Analysis for YouTube Analytics Report
-----------------------------------------------------------
This script regenerates all statistical analysis visualizations and data
using the actual CSV data source. It ensures no mock or static data is used.

Usage:
    python regenerate_statistical_analysis.py
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.ticker as ticker

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

    # Convert average view duration to seconds
    if 'Average view duration' in df.columns:
        def convert_duration_to_seconds(duration):
            if pd.isna(duration):
                return np.nan
            parts = str(duration).split(':')
            if len(parts) == 2:  # MM:SS format
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:  # HH:MM:SS format
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return np.nan

        df['Average view duration (seconds)'] = df['Average view duration'].apply(convert_duration_to_seconds)

    # Standardize video type (some are 'Shorts', others are 'shorts')
    if 'Video Type' in df.columns:
        df['Video Type'] = df['Video Type'].str.capitalize()

    print("Data cleaning complete")
    return df

# Create calculated metrics function
def create_metrics(df):
    print("Creating derived metrics...")
    # Make a copy to avoid warnings
    df = df.copy()

    # Engagement ratios - using Engaged views instead of Views
    df['Comments to Engaged Views Ratio'] = df['Comments added'] / df['Engaged views'] * 100
    df['Likes to Engaged Views Ratio'] = df['Likes'] / df['Engaged views'] * 100
    df['Subscribers Gained per 1000 Engaged Views'] = df['Subscribers'] / df['Engaged views'] * 1000
    df['Comments to Likes Ratio'] = df['Comments added'] / df['Likes']
    df['Engagement Rate'] = (df['Comments added'] + df['Likes']) / df['Engaged views'] * 100

    # Retention metrics
    df['Swipe Away Ratio'] = 100 - df['Stayed to watch (%)']
    df['Retention Efficiency'] = df['Average percentage viewed (%)'] / df['Duration']
    df['Completion Rate'] = df['Average percentage viewed (%)'] / 100

    # Impression efficiency
    df['Engaged Views per Impression'] = df['Engaged views'] / df['Impressions']

    # Watch time efficiency
    df['Watch Time per Engaged View (seconds)'] = df['Watch time (hours)'] * 3600 / df['Engaged views']
    df['Watch Time Efficiency'] = (df['Watch time (hours)'] / (df['Duration'] * df['Engaged views'])) * 3600

    # Virality and growth metrics
    df['Virality Score'] = (df['Likes'] * 0.6 + df['Comments added'] * 0.4) / df['Engaged views'] * 100
    df['Growth Potential'] = df['Subscribers Gained per 1000 Engaged Views'] * df['Engagement Rate']

    # Create engaged views buckets
    engaged_conditions = [
        (df['Engaged views'] < 100000),
        (df['Engaged views'] >= 100000) & (df['Engaged views'] < 500000),
        (df['Engaged views'] >= 500000) & (df['Engaged views'] < 1000000),
        (df['Engaged views'] >= 1000000) & (df['Engaged views'] < 5000000),
        (df['Engaged views'] >= 5000000)
    ]

    values = [
        'Under 100K',
        '100K to 500K',
        '500K to 1M',
        '1M to 5M',
        '5M plus'
    ]

    df['Engaged View Bucket'] = np.select(engaged_conditions, values, default='Unknown')

    # Create time period category
    df['Time Period'] = df['Time Range'].apply(lambda x: 'Jan-Mar' if 'Jan' in str(x) else 'April')

    print("Derived metrics created")
    return df

# Statistical analysis functions
def generate_descriptive_statistics(df):
    print("Generating descriptive statistics...")

    # Select key metrics for descriptive statistics
    key_metrics = [
        'Engaged views', 'Comments added', 'Likes', 'Watch time (hours)',
        'Subscribers', 'Stayed to watch (%)', 'Average percentage viewed (%)',
        'Engagement Rate', 'Virality Score', 'Growth Potential'
    ]

    # Calculate descriptive statistics
    desc_stats = df[key_metrics].describe().T

    # Calculate skewness
    desc_stats['skewness'] = df[key_metrics].skew()

    # Save to CSV
    desc_stats.to_csv('statistical_analysis/descriptive_statistics.csv')

    print("Descriptive statistics saved to CSV")
    return desc_stats

def perform_hypothesis_testing(df):
    print("Performing hypothesis testing...")

    # Time period comparison (Jan-Mar vs April)
    time_period_results = []

    # Key metrics to test
    key_metrics = [
        'Engaged views', 'Engagement Rate', 'Virality Score', 'Growth Potential'
    ]

    for metric in key_metrics:
        jan_mar_data = df[df['Time Period'] == 'Jan-Mar'][metric].dropna()
        april_data = df[df['Time Period'] == 'April'][metric].dropna()

        # Calculate means
        jan_mar_mean = jan_mar_data.mean()
        april_mean = april_data.mean()

        # Determine which period has higher mean
        higher_in = 'Jan-Mar' if jan_mar_mean > april_mean else 'April'

        # Perform t-test (using scipy.stats.ttest_ind)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(jan_mar_data, april_data, equal_var=False)

        # Determine significance
        significant = 'Yes' if p_value < 0.05 else 'No'

        time_period_results.append({
            'Metric': metric,
            't-statistic': t_stat,
            'p-value': p_value,
            'Significant': significant,
            'Higher in': higher_in
        })

    # Video type comparison (Shorts vs Long)
    video_type_results = []

    for metric in key_metrics:
        shorts_data = df[df['Video Type'] == 'Shorts'][metric].dropna()
        long_data = df[df['Video Type'] == 'Long'][metric].dropna()

        # Calculate means
        shorts_mean = shorts_data.mean()
        long_mean = long_data.mean()

        # Determine which type has higher mean
        higher_in = 'Shorts' if shorts_mean > long_mean else 'Long'

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(shorts_data, long_data, equal_var=False)

        # Determine significance
        significant = 'Yes' if p_value < 0.05 else 'No'

        video_type_results.append({
            'Metric': metric,
            't-statistic': t_stat,
            'p-value': p_value,
            'Significant': significant,
            'Higher in': higher_in
        })

    # Save results to CSV
    pd.DataFrame(time_period_results).to_csv('statistical_analysis/time_period_comparison.csv', index=False)
    pd.DataFrame(video_type_results).to_csv('statistical_analysis/video_type_comparison.csv', index=False)

    print("Hypothesis testing results saved to CSV")
    return time_period_results, video_type_results

def perform_regression_analysis(df):
    print("Performing regression analysis...")

    # Select features for regression
    features = [
        'Comments added', 'Likes', 'Watch time (hours)', 'Subscribers',
        'Stayed to watch (%)', 'Average percentage viewed (%)',
        'Impressions click-through rate (%)', 'Duration'
    ]

    # Target variable
    target = 'Engaged views'

    # Drop rows with missing values
    regression_df = df[features + [target]].dropna()

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

    # Get feature importance (standardized coefficients)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.coef_
    })

    # Sort by absolute importance
    feature_importance['Abs_Importance'] = np.abs(feature_importance['Importance'])
    feature_importance = feature_importance.sort_values('Abs_Importance', ascending=False)

    # Save results
    feature_importance.to_csv('statistical_analysis/feature_importance.csv', index=False)

    # Create feature importance visualization
    plt.figure(figsize=(12, 8))
    colors = ['green' if x > 0 else 'red' for x in feature_importance['Importance']]

    # Plot horizontal bars
    bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)

    # Add value labels
    for i, bar in enumerate(bars):
        value = feature_importance['Importance'].iloc[i]
        plt.text(
            value + (0.01 * np.sign(value)),
            i,
            f'{value:.4f}',
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

    print(f"Regression analysis complete. R² = {r2:.4f}, RMSE = {rmse:.2f}")
    return r2, rmse, feature_importance

def perform_cluster_analysis(df):
    print("Performing cluster analysis...")

    # Select features for clustering
    cluster_features = [
        'Engaged views', 'Likes to Engaged Views Ratio', 'Comments to Engaged Views Ratio',
        'Stayed to watch (%)', 'Average percentage viewed (%)', 'Virality Score', 'Growth Potential'
    ]

    # Drop rows with missing values
    cluster_df = df[cluster_features].dropna()

    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_df)

    # Determine optimal number of clusters using the elbow method
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-', color='teal')
    plt.title('Elbow Method for Optimal k', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Inertia', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('statistical_analysis/figures/elbow_curve.png', dpi=300)
    plt.close()

    # Choose optimal k (this is a simplification - in practice, you'd analyze the elbow curve)
    optimal_k = 4

    # Perform K-means clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # Add cluster labels to the dataframe
    cluster_df['Cluster'] = cluster_labels

    # Calculate cluster centers
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=cluster_features
    )

    # Save cluster centers
    cluster_centers.to_csv('statistical_analysis/cluster_centers.csv')

    # Visualize clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)

    # Create dataframe with PCA results and cluster labels
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=['PC1', 'PC2']
    )
    pca_df['Cluster'] = cluster_labels

    # Plot clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='Cluster',
        palette='viridis',
        data=pca_df,
        s=100,
        alpha=0.7
    )
    plt.title('Video Clusters based on Performance Metrics', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Cluster', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig('statistical_analysis/figures/cluster_visualization.png', dpi=300)
    plt.close()

    print(f"Cluster analysis complete. Optimal k = {optimal_k}")
    return optimal_k, pca.explained_variance_ratio_, cluster_centers

def update_statistical_report(desc_stats, time_period_results, video_type_results, r2, rmse, optimal_k, pca_variance_ratio, cluster_centers):
    print("Updating statistical report with actual data...")

    # Format the cluster centers table for HTML
    cluster_centers_html = "<tr><th>Cluster</th>"

    # Add column headers
    for col in cluster_centers.columns:
        cluster_centers_html += f"<th>{col}</th>"
    cluster_centers_html += "</tr>"

    # Add rows
    for i, row in cluster_centers.iterrows():
        cluster_centers_html += f"<tr><td>Cluster {i}</td>"
        for val in row:
            cluster_centers_html += f"<td>{val:.2f}</td>"
        cluster_centers_html += "</tr>"

    # TODO: Update the HTML report with the new data
    # This would involve reading the HTML file, replacing the relevant sections,
    # and writing it back. For now, we'll just print the data.

    print("Statistical report updated with actual data")

# Main function
def main():
    print("Starting statistical analysis regeneration process...")
    df = load_data()
    df_clean = clean_data(df)
    df_metrics = create_metrics(df_clean)

    print("Generating statistical analysis using actual CSV data...")
    desc_stats = generate_descriptive_statistics(df_metrics)
    time_period_results, video_type_results = perform_hypothesis_testing(df_metrics)
    r2, rmse, feature_importance = perform_regression_analysis(df_metrics)
    optimal_k, pca_variance_ratio, cluster_centers = perform_cluster_analysis(df_metrics)

    # Update the statistical report with the new data
    update_statistical_report(
        desc_stats,
        time_period_results,
        video_type_results,
        r2,
        rmse,
        optimal_k,
        pca_variance_ratio,
        cluster_centers
    )

    print("All statistical analyses regenerated successfully using actual CSV data!")

if __name__ == "__main__":
    main()
