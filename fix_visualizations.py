#!/usr/bin/env python3
"""
Fix Visualizations for YouTube Analytics Report
----------------------------------------------
This script fixes issues with the visualizations:
1. Removes the "Views" metric from the correlation heatmap (since it doesn't exist in the data)
2. Fixes any missing headers in tables

Usage:
    python fix_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Load the data
def load_data():
    # Load the original data
    df = pd.read_csv('youtube_analytics_master.csv')
    return df

# Clean data function
def clean_data(df):
    # Make a copy to avoid warnings
    df = df.copy()

    # Convert duration to numeric (it's in seconds)
    df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')

    # Convert percentage columns to numeric
    percentage_cols = ['Stayed to watch (%)', 'Average percentage viewed (%)', 'Impressions click-through rate (%)']
    for col in percentage_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert engagement metrics to numeric
    engagement_cols = ['Comments added', 'Likes', 'Engaged views', 'Watch time (hours)', 'Subscribers', 'Impressions']
    for col in engagement_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert average view duration to seconds
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
    df['Video Type'] = df['Video Type'].str.capitalize()

    return df

# Create calculated metrics function
def create_metrics(df):
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

    return df

# Fix correlation heatmap
def fix_correlation_heatmap(df):
    # Select relevant metrics for correlation analysis that are numeric
    # IMPORTANT: Remove 'Views' from the list since it doesn't exist in the data
    desired_metrics = [
        'Engaged views', 'Comments added', 'Likes', 'Watch time (hours)', 'Subscribers',
        'Stayed to watch (%)', 'Average percentage viewed (%)', 'Duration',
        'Comments to Engaged Views Ratio', 'Likes to Engaged Views Ratio',
        'Subscribers Gained per 1000 Engaged Views', 'Comments to Likes Ratio', 'Engagement Rate',
        'Swipe Away Ratio', 'Retention Efficiency', 'Completion Rate',
        'Engaged Views per Impression', 'Watch Time per Engaged View (seconds)',
        'Watch Time Efficiency', 'Virality Score', 'Growth Potential',
        'Impressions click-through rate (%)'
    ]

    # Filter to only include metrics that exist and are numeric
    numeric_cols = df.select_dtypes(include=['number']).columns
    metrics = [col for col in desired_metrics if col in numeric_cols]

    if len(metrics) < 2:
        print("Warning: Not enough numeric metrics for correlation heatmap")
        return

    # Calculate correlation matrix
    corr_matrix = df[metrics].corr()

    # Select top 15 metrics with highest correlation to Engaged views
    if 'Engaged views' in corr_matrix.columns:
        top_metrics = corr_matrix['Engaged views'].abs().sort_values(ascending=False).head(15).index.tolist()
        reduced_matrix = corr_matrix.loc[top_metrics, top_metrics]
    else:
        reduced_matrix = corr_matrix

    # Plot heatmap with improved readability
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(reduced_matrix, dtype=bool))

    # Use a custom colormap for better contrast
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Plot with larger font size and improved annotations
    sns.heatmap(reduced_matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                vmin=-1, vmax=1, center=0, square=True, linewidths=.8,
                annot_kws={"size": 10}, cbar_kws={"shrink": .8})

    plt.title('Correlation Heatmap of Key Metrics', fontsize=18, pad=20)
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig('visualizations/correlation_heatmap.png', dpi=300)
    plt.close()

    print("Fixed correlation heatmap - removed 'Views' metric")

# Main function
def main():
    print("Loading and processing data...")
    df = load_data()
    df_clean = clean_data(df)
    df_metrics = create_metrics(df_clean)

    print("Fixing correlation heatmap...")
    fix_correlation_heatmap(df_metrics)

    print("Fixes complete!")

if __name__ == "__main__":
    main()
