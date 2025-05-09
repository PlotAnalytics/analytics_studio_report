#!/usr/bin/env python3
"""
Generate Visualizations for Cluster Analysis Deep Dive Report
------------------------------------------------------------
This script creates visualizations for the cluster analysis deep dive report,
including cluster distribution charts, radar charts for cluster profiles,
and a mock video list for each cluster.

Usage:
    python generate_cluster_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Create output directory
os.makedirs('cluster_analysis/figures', exist_ok=True)

# Load cluster centers data
cluster_centers = pd.read_csv('statistical_analysis/cluster_centers.csv', index_col=0)

# Define cluster names and colors
cluster_names = {
    0: "Solid Performers",
    1: "Niche Engagers",
    2: "Viral Performers",
    3: "Anomalous Content"
}

cluster_colors = {
    0: "#1a73e8",  # Blue
    1: "#fbbc05",  # Yellow
    2: "#34a853",  # Green
    3: "#9c27b0"   # Purple
}

# Define cluster distribution (approximate percentages)
cluster_distribution = {
    0: 35,  # Solid Performers
    1: 45,  # Niche Engagers
    2: 15,  # Viral Performers
    3: 5    # Anomalous Content
}

# 1. Create Cluster Distribution Pie Chart
def create_cluster_distribution_chart():
    plt.figure(figsize=(10, 8))
    
    # Data
    labels = [f"{cluster_names[i]}\n({cluster_distribution[i]}%)" for i in range(4)]
    sizes = list(cluster_distribution.values())
    colors = list(cluster_colors.values())
    explode = (0, 0, 0.1, 0.2)  # Explode the viral and anomalous slices
    
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
               'Stayed to watch (%)', 'Average percentage viewed (%)', 
               'Virality Score', 'Growth Potential']
    
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

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

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
               'Stayed to watch (%)', 'Average percentage viewed (%)', 
               'Virality Score', 'Growth Potential']
    
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

# 4. Generate mock video list for each cluster
def generate_mock_video_lists():
    # Create a mock dataset of videos for each cluster
    video_data = []
    
    # Video title templates for each cluster
    title_templates = {
        0: [  # Solid Performers
            "How to {action} Your {object} in {timeframe}",
            "Top 10 {adjective} {objects} for {activity}",
            "{number} Ways to Improve Your {skill}",
            "The Ultimate Guide to {topic}",
            "Why Your {object} Isn't {action}ing Properly"
        ],
        1: [  # Niche Engagers
            "Advanced {topic} Techniques Only Experts Know",
            "The Truth About {controversial_topic} Nobody Talks About",
            "How I {achievement} in Just {timeframe} (Detailed Breakdown)",
            "Responding to Your Questions About {niche_topic}",
            "Deep Dive: Understanding {complex_topic} From First Principles"
        ],
        2: [  # Viral Performers
            "I Tried {celebrity}'s {routine} for {timeframe} - Here's What Happened",
            "We {extreme_action} for 24 Hours Straight!",
            "{shocking_statement} (Not Clickbait)",
            "This {object} Changed My Life Forever",
            "You Won't Believe What Happened When I {action}..."
        ],
        3: [  # Anomalous Content
            "Test Video - Please Ignore",
            "Video Removed Due to Copyright Claim",
            "Private: {topic} Discussion (Draft)",
            "Unlisted: {event} Footage Raw",
            "Deleted Scene from {content}"
        ]
    }
    
    # Fill-in variables
    variables = {
        'action': ['Optimize', 'Transform', 'Upgrade', 'Fix', 'Improve', 'Master'],
        'object': ['Workflow', 'Setup', 'Strategy', 'Routine', 'System', 'Process'],
        'timeframe': ['7 Days', '24 Hours', 'One Month', '5 Minutes', 'Two Weeks'],
        'adjective': ['Essential', 'Game-Changing', 'Underrated', 'Powerful', 'Innovative'],
        'objects': ['Tools', 'Techniques', 'Methods', 'Resources', 'Strategies', 'Hacks'],
        'activity': ['Productivity', 'Learning', 'Growth', 'Success', 'Performance'],
        'number': ['5', '7', '10', '3', '12', '15'],
        'skill': ['Productivity', 'Focus', 'Creativity', 'Learning', 'Memory'],
        'topic': ['Time Management', 'Note-Taking', 'Reading', 'Public Speaking', 'Networking'],
        'controversial_topic': ['Productivity Systems', 'Speed Reading', 'Multitasking', 'AI Tools'],
        'achievement': ['Doubled My Output', 'Mastered a New Skill', 'Built a Second Income'],
        'niche_topic': ['Zettelkasten Method', 'Spaced Repetition', 'Deep Work Protocol'],
        'complex_topic': ['Knowledge Management', 'Cognitive Biases', 'Flow State Triggers'],
        'celebrity': ['Elon Musk', 'Bill Gates', 'Warren Buffett', 'Tim Ferriss'],
        'routine': ['Morning Routine', 'Productivity System', 'Workout Regimen', 'Diet'],
        'extreme_action': ['Lived in the Woods', 'Worked Without Sleep', 'Ate Only One Food'],
        'shocking_statement': ['I Quit My Job to Do This', 'This Changed Everything', 'I Was Wrong All Along'],
        'event': ['Conference', 'Meeting', 'Interview', 'Live Event']
    }
    
    # Generate random video data
    np.random.seed(42)  # For reproducibility
    
    for cluster_id in range(4):
        # Number of videos in this cluster (based on distribution)
        n_videos = 20  # Fixed number for the example
        
        for i in range(n_videos):
            # Select random title template
            template = random.choice(title_templates[cluster_id])
            
            # Fill in template with random variables
            title = template
            for var in variables:
                if '{' + var + '}' in title:
                    title = title.replace('{' + var + '}', random.choice(variables[var]))
            
            # Generate random metrics appropriate for the cluster
            if cluster_id == 0:  # Solid Performers
                views = np.random.randint(100000, 300000)
                likes = int(views * np.random.uniform(0.05, 0.08))
                comments = int(views * np.random.uniform(0.0003, 0.0007))
            elif cluster_id == 1:  # Niche Engagers
                views = np.random.randint(5000, 30000)
                likes = int(views * np.random.uniform(0.03, 0.05))
                comments = int(views * np.random.uniform(0.001, 0.003))
            elif cluster_id == 2:  # Viral Performers
                views = np.random.randint(1000000, 3000000)
                likes = int(views * np.random.uniform(0.07, 0.1))
                comments = int(views * np.random.uniform(0.0002, 0.0006))
            else:  # Anomalous Content
                views = np.random.randint(0, 10)
                likes = -np.random.randint(0, 100)
                comments = 0
            
            # Add to dataset
            video_data.append({
                'Cluster': cluster_id,
                'Cluster Name': cluster_names[cluster_id],
                'Video Title': title,
                'Views': views,
                'Likes': likes,
                'Comments': comments,
                'Video ID': f"vid_{cluster_id}_{i:03d}"
            })
    
    # Convert to DataFrame
    videos_df = pd.DataFrame(video_data)
    
    # Save to CSV
    videos_df.to_csv('cluster_analysis/cluster_videos.csv', index=False)
    
    return videos_df

# Run all visualization functions
def main():
    print("Generating cluster distribution chart...")
    create_cluster_distribution_chart()
    
    print("Generating cluster metrics comparison chart...")
    create_cluster_metrics_comparison()
    
    print("Generating cluster radar charts...")
    create_cluster_radar_charts()
    
    print("Generating mock video lists...")
    generate_mock_video_lists()
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    main()
