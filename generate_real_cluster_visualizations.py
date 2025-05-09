#!/usr/bin/env python3
"""
Generate Visualizations for Cluster Analysis Deep Dive Report Using Real Data
----------------------------------------------------------------------------
This script creates visualizations for the cluster analysis deep dive report
using the actual YouTube analytics data, including cluster distribution charts,
radar charts for cluster profiles, and metrics comparisons.

Usage:
    python generate_real_cluster_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Create output directory
os.makedirs('cluster_analysis/figures', exist_ok=True)

# Load the data
df = pd.read_csv('youtube_analytics_master.csv')

# Create derived metrics
df['Likes to Views Ratio'] = df['Likes'] / df['Engaged views'] * 100
df['Comments to Views Ratio'] = df['Comments added'] / df['Engaged views'] * 100

# Select features for clustering
cluster_features = [
    'Engaged views', 'Likes to Views Ratio', 'Comments to Views Ratio',
    'Average percentage viewed (%)'
]

# Drop rows with missing values
cluster_df = df[cluster_features].dropna()

# Perform K-means clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_df)

# Use 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_features)

# Add cluster labels to the cluster_df
cluster_df['Cluster'] = cluster_labels

# Store the indices of the rows we're using for clustering
cluster_indices = cluster_df.index

# Create a new column in the original df for clusters, initialized with NaN
df['Cluster'] = np.nan

# Assign cluster labels only to the rows that were used in clustering
df.loc[cluster_indices, 'Cluster'] = cluster_labels

# Calculate cluster centers
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=cluster_features
)

# Define cluster names and colors based on our analysis
cluster_names = {
    0: "Solid Performers",
    1: "Viral Performers",
    2: "Niche Engagers",
    3: "Anomalous Content"
}

cluster_colors = {
    0: "#1a73e8",  # Blue
    1: "#34a853",  # Green
    2: "#fbbc05",  # Yellow
    3: "#9c27b0"   # Purple
}

# Get actual cluster distribution
cluster_counts = df['Cluster'].dropna().astype(int).value_counts().sort_index()
cluster_distribution = {}
for cluster in range(4):
    if cluster in cluster_counts:
        cluster_distribution[cluster] = round(cluster_counts[cluster] / len(cluster_df) * 100, 1)
    else:
        cluster_distribution[cluster] = 0.0

# 1. Create Cluster Distribution Pie Chart
def create_cluster_distribution_chart():
    plt.figure(figsize=(10, 8))
    
    # Data
    labels = [f"{cluster_names[i]}\n({cluster_distribution[i]}%)" for i in range(4)]
    sizes = list(cluster_distribution.values())
    colors = list(cluster_colors.values())
    explode = (0, 0.1, 0, 0.2)  # Explode the viral and anomalous slices
    
    # Plot
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 14})
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Distribution of Videos Across Clusters', fontsize=18, pad=20)
    
    # Save
    plt.tight_layout()
    plt.savefig('cluster_analysis/figures/cluster_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Create Cluster Metrics Comparison Bar Chart
def create_cluster_metrics_comparison():
    # Select metrics to compare
    metrics = ['Engaged views', 'Likes to Views Ratio', 'Comments to Views Ratio', 
               'Average percentage viewed (%)']
    
    # Normalize data for comparison (exclude anomalous cluster)
    df_norm = cluster_centers.loc[[0, 1, 2]].copy()
    
    # Log transform engaged views for better visualization
    df_norm['Engaged views'] = np.log10(df_norm['Engaged views'])
    
    # Normalize each column to 0-1 scale
    for col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    # Prepare data for plotting
    df_plot = df_norm.T
    df_plot.columns = [cluster_names[i] for i in df_plot.columns]
    
    # Plot
    plt.figure(figsize=(14, 10))
    ax = df_plot.plot(kind='bar', width=0.8, figsize=(14, 10), 
                      color=[cluster_colors[0], cluster_colors[1], cluster_colors[2]])
    
    # Customize plot
    plt.title('Normalized Metrics Comparison Across Clusters', fontsize=18, pad=20)
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Normalized Value (higher is better)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value annotations
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    # Save
    plt.tight_layout()
    plt.savefig('cluster_analysis/figures/cluster_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Create Radar Chart for Cluster Profiles
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes."""
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Rotate plot so that first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon returns a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    # Register projection
    register_projection(RadarAxes)
    return theta

def create_cluster_radar_charts():
    # Select metrics for radar chart
    metrics = ['Likes to Views Ratio', 'Comments to Views Ratio', 
               'Average percentage viewed (%)']
    
    # Normalize data for radar chart (exclude anomalous cluster)
    df_radar = cluster_centers.loc[[0, 1, 2], metrics].copy()
    
    # Normalize each column to 0-1 scale
    for col in df_radar.columns:
        min_val = df_radar[col].min()
        max_val = df_radar[col].max()
        if max_val > min_val:
            df_radar[col] = (df_radar[col] - min_val) / (max_val - min_val)
    
    # Set up radar chart
    N = len(metrics)
    theta = radar_factory(N, frame='polygon')
    
    # Create figure
    fig, axs = plt.subplots(figsize=(12, 10), nrows=1, ncols=1,
                           subplot_kw=dict(projection='radar'))
    
    # Plot each cluster
    for i, (idx, row) in enumerate(df_radar.iterrows()):
        color = cluster_colors[idx]
        axs.plot(theta, row.values, color=color, linewidth=2.5, label=cluster_names[idx])
        axs.fill(theta, row.values, facecolor=color, alpha=0.25)
    
    # Customize plot
    axs.set_varlabels(metrics)
    axs.set_ylim(0, 1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    plt.title('Cluster Performance Profile Comparison', fontsize=18, y=1.05)
    
    # Save
    plt.tight_layout()
    plt.savefig('cluster_analysis/figures/cluster_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run all visualization functions
def main():
    print("Generating cluster distribution chart...")
    create_cluster_distribution_chart()
    
    print("Generating cluster metrics comparison chart...")
    create_cluster_metrics_comparison()
    
    print("Generating cluster radar charts...")
    create_cluster_radar_charts()
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    main()
