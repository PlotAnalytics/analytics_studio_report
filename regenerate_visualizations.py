#!/usr/bin/env python3
"""
Regenerate Visualizations for YouTube Analytics Report
-----------------------------------------------------
This script regenerates all visualizations for the YouTube analytics report
using the actual CSV data source. It ensures no mock or static data is used.

Usage:
    python regenerate_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

# Set style for plots
plt.style.use('ggplot')
sns.set_theme(font_scale=1.2)  # Using set_theme instead of deprecated set
sns.set_style("whitegrid")

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

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

# Visualization functions
def plot_view_bucket_distribution(df):
    print("Generating view bucket distribution chart...")
    plt.figure(figsize=(12, 6))

    # Count videos in each bucket
    bucket_counts = df['Engaged View Bucket'].value_counts().sort_index()

    # Create custom order for view buckets
    bucket_order = ['Under 100K', '100K to 500K', '500K to 1M', '1M to 5M', '5M plus']
    bucket_counts = bucket_counts.reindex(bucket_order)

    # Define colors for better visual distinction
    colors = sns.color_palette("viridis", len(bucket_order))

    # Plot with custom colors
    ax = sns.barplot(x=bucket_counts.index, y=bucket_counts.values, palette=colors)

    # Add count labels on top of bars
    for i, count in enumerate(bucket_counts.values):
        if pd.notna(count):  # Check if count is not NaN
            ax.text(i, count + (count * 0.03), str(int(count)),
                   ha='center', fontweight='bold', fontsize=12)

    # Add count labels inside bars
    for i, bar in enumerate(ax.patches):
        count = bucket_counts.values[i]
        if pd.notna(count) and count > 0:  # Check if count is not NaN and greater than 0
            # Position text in the middle of the bar
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = bar.get_height() / 2

            # Add text with white outline for better visibility
            ax.text(text_x, text_y, str(int(count)),
                  ha='center', va='center',
                  fontweight='bold', color='white',
                  bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    plt.title('Distribution of Videos by Engaged View Count Bucket', fontsize=16, pad=15)
    plt.xlabel('Engaged View Count Bucket', fontsize=14)
    plt.ylabel('Number of Videos', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure
    plt.savefig('visualizations/view_bucket_distribution.png', dpi=300)
    plt.close()

def plot_metrics_by_view_bucket(df):
    print("Generating metrics by view bucket chart...")
    # Define key metrics to plot
    metrics = [
        'Likes to Engaged Views Ratio',
        'Comments to Engaged Views Ratio',
        'Subscribers Gained per 1000 Engaged Views',
        'Stayed to watch (%)',
        'Average percentage viewed (%)',
        'Impressions click-through rate (%)'
    ]

    # Create custom order for view buckets
    bucket_order = ['Under 100K', '100K to 500K', '500K to 1M', '1M to 5M', '5M plus']

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()

    # Define colors for better visual distinction
    colors = sns.color_palette("viridis", len(bucket_order))

    for i, metric in enumerate(metrics):
        # Group by engaged view bucket and calculate mean of the metric
        metric_by_bucket = df.groupby('Engaged View Bucket')[metric].mean().reindex(bucket_order)

        # Plot with custom colors
        bars = sns.barplot(x=metric_by_bucket.index, y=metric_by_bucket.values, ax=axes[i], palette=colors)

        # Add value annotations on each bar
        for j, bar in enumerate(bars.patches):
            value = metric_by_bucket.values[j]
            if pd.notna(value):  # Check if value is not NaN
                # Format value based on metric type
                if '%' in metric:
                    value_text = f'{value:.1f}%'
                elif value > 1000:
                    value_text = f'{value:.1f}'
                else:
                    value_text = f'{value:.2f}'

                # Position text in the middle of the bar
                text_x = bar.get_x() + bar.get_width() / 2
                text_y = bar.get_height() / 2

                # Add text with white outline for better visibility
                axes[i].text(text_x, text_y, value_text,
                          ha='center', va='center',
                          fontweight='bold', color='white',
                          bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

        axes[i].set_title(f'{metric} by Engaged View Bucket', fontsize=14, pad=10)
        axes[i].set_xlabel('Engaged View Count Bucket', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)

        # Add grid lines for better readability
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        # Format y-axis as percentage
        if '%' in metric:
            axes[i].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))

    plt.tight_layout()

    # Save the figure
    plt.savefig('visualizations/metrics_by_engaged_view_bucket.png', dpi=300)
    plt.close()

def plot_impact_scale(df):
    """Create a visual scale showing all variables and their impact levels on virality"""
    print("Generating impact scale visualization...")
    # Get correlation with engaged views
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if 'Engaged views' not in numeric_cols:
        print("Warning: 'Engaged views' column is not numeric. Impact scale cannot be created.")
        return

    # Calculate correlation with engaged views
    correlation_values = df[numeric_cols].corr()['Engaged views'].sort_values(ascending=False)

    # Remove Engaged views itself from the correlation values
    if 'Engaged views' in correlation_values.index:
        correlation_values = correlation_values.drop('Engaged views')

    # Filter out metrics with too many NaN values
    correlation_values = correlation_values.dropna()

    # Take top 20 metrics by absolute correlation value
    top_metrics = correlation_values.abs().sort_values(ascending=False).head(20).index
    correlation_values = correlation_values[top_metrics]

    # Create a horizontal bar chart with color gradient
    plt.figure(figsize=(14, 12))

    # Define color map based on correlation values
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(correlation_values)))

    # Sort by correlation value
    correlation_values = correlation_values.sort_values()

    # Create horizontal bar chart
    bars = plt.barh(correlation_values.index, correlation_values.values, color=colors)

    # Add correlation values as text labels
    for i, (metric, value) in enumerate(correlation_values.items()):
        plt.text(value + (0.05 if value >= 0 else -0.05),
                 i,
                 f'{value:.2f}',
                 va='center',
                 ha='left' if value >= 0 else 'right',
                 fontweight='bold',
                 fontsize=10)

    # Add vertical lines to indicate impact levels
    plt.axvline(x=0.7, color='green', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=0.3, color='yellow', linestyle='--', alpha=0.7, linewidth=2)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    plt.axvline(x=-0.3, color='orange', linestyle='--', alpha=0.7, linewidth=2)

    # Add impact level labels
    plt.text(0.85, len(correlation_values) - 1, 'HIGH IMPACT', fontsize=12, color='green', fontweight='bold')
    plt.text(0.45, len(correlation_values) - 2, 'MODERATE IMPACT', fontsize=12, color='olive', fontweight='bold')
    plt.text(0.15, len(correlation_values) - 3, 'LOW IMPACT', fontsize=12, color='orange', fontweight='bold')
    plt.text(-0.45, len(correlation_values) - 4, 'NEGATIVE IMPACT', fontsize=12, color='red', fontweight='bold')

    # Customize the plot
    plt.title('Impact of Metrics on Engaged Views (Virality)', fontsize=18, pad=20)
    plt.xlabel('Correlation with Engaged Views', fontsize=14)
    plt.ylabel('Metrics', fontsize=14)
    plt.xlim(-1, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add a legend explaining the correlation values
    plt.figtext(0.5, 0.01,
                'Correlation values range from -1 to 1. Values closer to 1 indicate stronger positive correlation with virality.\n'
                'Values closer to -1 indicate stronger negative correlation. Values near 0 indicate little to no correlation.',
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save the figure
    plt.savefig('visualizations/impact_scale.png', dpi=300)
    plt.close()

def plot_correlation_heatmap(df):
    print("Generating correlation heatmap...")
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

# Main function
def main():
    print("Starting visualization regeneration process...")
    df = load_data()
    df_clean = clean_data(df)
    df_metrics = create_metrics(df_clean)

    print("Generating visualizations using actual CSV data...")
    plot_view_bucket_distribution(df_metrics)
    plot_metrics_by_view_bucket(df_metrics)
    plot_impact_scale(df_metrics)
    plot_correlation_heatmap(df_metrics)

    print("All visualizations regenerated successfully using actual CSV data!")

if __name__ == "__main__":
    main()
