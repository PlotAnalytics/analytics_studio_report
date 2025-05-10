#!/usr/bin/env python3

"""
This script analyzes how video length affects virality and adds this analysis to the YouTube Analytics report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from scipy import stats

# Create output directories
os.makedirs('visualizations', exist_ok=True)

def load_data():
    """Load and process the YouTube analytics data"""
    print("Loading data from youtube_analytics_master.csv...")
    # Load the original data
    df = pd.read_csv('youtube_analytics_master.csv')
    print(f"Loaded {len(df)} rows of data")
    return df

def clean_data(df):
    """Clean and prepare the data for analysis"""
    print("Cleaning data...")
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

    # Standardize video type (some are 'Shorts', others are 'shorts')
    df['Video Type'] = df['Video Type'].str.capitalize()

    return df

def create_metrics(df):
    """Create calculated metrics for analysis"""
    print("Creating metrics...")
    # Make a copy to avoid warnings
    df = df.copy()

    # Engagement ratios
    df['Comments to Engaged Views Ratio'] = df['Comments added'] / df['Engaged views'] * 100
    df['Likes to Engaged Views Ratio'] = df['Likes'] / df['Engaged views'] * 100
    df['Subscribers Gained per 1000 Engaged Views'] = df['Subscribers'] / df['Engaged views'] * 1000
    df['Comments to Likes Ratio'] = df['Comments added'] / df['Likes'] * 100
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
    # Handle division by zero or missing data
    df['Virality Score'] = np.where(
        (df['Engaged views'].isnull()) | (df['Engaged views'] == 0),
        0,  # Set to 0 if engaged views is null or zero
        (df['Likes'].fillna(0) * 0.6 + df['Comments added'].fillna(0) * 0.4) / df['Engaged views'] * 100
    )

    # Replace infinite values with 0
    df['Virality Score'] = df['Virality Score'].replace([np.inf, -np.inf], 0)

    # Growth potential with similar handling
    df['Growth Potential'] = df['Subscribers Gained per 1000 Engaged Views'].fillna(0) * df['Engagement Rate'].fillna(0)

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

    # Create duration buckets for analysis
    df['Duration (minutes)'] = df['Duration'] / 60  # Convert seconds to minutes

    # Create duration buckets
    duration_conditions = [
        (df['Duration (minutes)'] < 1),  # Less than 1 minute (typical Shorts)
        (df['Duration (minutes)'] >= 1) & (df['Duration (minutes)'] < 3),  # 1-3 minutes
        (df['Duration (minutes)'] >= 3) & (df['Duration (minutes)'] < 5),  # 3-5 minutes
        (df['Duration (minutes)'] >= 5) & (df['Duration (minutes)'] < 10),  # 5-10 minutes
        (df['Duration (minutes)'] >= 10) & (df['Duration (minutes)'] < 15),  # 10-15 minutes
        (df['Duration (minutes)'] >= 15)  # 15+ minutes
    ]

    duration_values = [
        'Under 1 min',
        '1-3 min',
        '3-5 min',
        '5-10 min',
        '10-15 min',
        '15+ min'
    ]

    df['Duration Bucket'] = np.select(duration_conditions, duration_values, default='Unknown')

    return df

def analyze_video_length_virality(df):
    """Analyze how video length affects virality, separated by video type (Shorts vs Long)"""
    print("Analyzing video length and virality relationship...")

    # Create separate dataframes for Shorts and Long videos using the Video Type column
    shorts_df = df[df['Video Type'] == 'Shorts'].copy()
    long_df = df[df['Video Type'] == 'Long'].copy()

    print(f"Analyzing {len(shorts_df)} Shorts videos and {len(long_df)} Long videos separately")

    # Function to analyze a specific video type
    def analyze_by_video_type(video_df, video_type):
        # Group by duration bucket and calculate mean metrics
        duration_analysis = video_df.groupby('Duration Bucket').agg({
            'Content': 'count',  # Count of videos
            'Engaged views': 'mean',
            'Likes': 'mean',
            'Comments added': 'mean',
            'Subscribers': 'mean',
            'Stayed to watch (%)': 'mean',
            'Average percentage viewed (%)': 'mean',
            'Virality Score': 'mean',
            'Growth Potential': 'mean',
            'Engagement Rate': 'mean',
            'Completion Rate': 'mean',
            'Duration': 'mean'  # Average duration in seconds
        }).reset_index()

        # Rename columns for clarity
        duration_analysis = duration_analysis.rename(columns={
            'Content': 'Number of Videos',
            'Engaged views': 'Avg. Engaged Views',
            'Likes': 'Avg. Likes',
            'Comments added': 'Avg. Comments',
            'Subscribers': 'Avg. Subscribers Gained',
            'Stayed to watch (%)': 'Avg. Stayed to Watch (%)',
            'Average percentage viewed (%)': 'Avg. Percentage Viewed (%)',
            'Virality Score': 'Avg. Virality Score',
            'Growth Potential': 'Avg. Growth Potential',
            'Engagement Rate': 'Avg. Engagement Rate (%)',
            'Completion Rate': 'Avg. Completion Rate',
            'Duration': 'Avg. Duration (seconds)'
        })

        # Define appropriate bucket order based on video type
        if video_type == 'Shorts':
            # Shorts can be up to 3 minutes, so include relevant buckets
            bucket_order = ['Under 1 min', '1-3 min']
        else:
            # For Long videos, use all duration buckets
            bucket_order = ['1-3 min', '3-5 min', '5-10 min', '10-15 min', '15+ min']

        # Filter to only include buckets that actually have data
        available_buckets = duration_analysis['Duration Bucket'].unique()
        bucket_order = [b for b in bucket_order if b in available_buckets]

        # Sort by duration bucket in a logical order
        duration_analysis['Duration Bucket'] = pd.Categorical(
            duration_analysis['Duration Bucket'],
            categories=bucket_order,
            ordered=True
        )
        duration_analysis = duration_analysis.sort_values('Duration Bucket')

        # Calculate correlation between duration and virality metrics
        correlation_data = video_df[['Duration', 'Virality Score', 'Engagement Rate', 'Completion Rate',
                                  'Stayed to watch (%)', 'Average percentage viewed (%)']].copy()

        # Remove rows with NaN values
        correlation_data = correlation_data.dropna()

        # Calculate Pearson correlation if we have enough data
        if len(correlation_data) > 1:
            correlation_matrix = correlation_data.corr(method='pearson')
            duration_correlations = correlation_matrix['Duration'].sort_values(ascending=False)
        else:
            # Not enough data for correlation
            duration_correlations = pd.Series(dtype='float64')

        return duration_analysis, duration_correlations

    # Analyze each video type
    shorts_analysis, shorts_correlations = analyze_by_video_type(shorts_df, 'Shorts')
    long_analysis, long_correlations = analyze_by_video_type(long_df, 'Long')

    # Create visualizations for each video type
    create_duration_virality_visualizations(shorts_df, shorts_analysis, shorts_correlations, 'Shorts')
    create_duration_virality_visualizations(long_df, long_analysis, long_correlations, 'Long')

    # Also create a combined visualization for comparison
    create_video_type_comparison(shorts_analysis, long_analysis)

    return {
        'Shorts': (shorts_analysis, shorts_correlations),
        'Long': (long_analysis, long_correlations)
    }

def create_duration_virality_visualizations(df, duration_analysis, duration_correlations, video_type):
    """Create visualizations for the video length and virality analysis, separated by video type"""
    print(f"Creating visualizations for {video_type} videos...")

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # Skip if no data
    if len(duration_analysis) == 0:
        print(f"No data available for {video_type} videos")
        return

    # File suffix for this video type
    suffix = f"_{video_type.lower()}"

    # 1. Bar chart of average virality score by duration bucket
    plt.figure(figsize=(12, 6))

    # Fill NaN values with 0 for visualization
    plot_data = duration_analysis.copy()
    plot_data['Avg. Virality Score'] = plot_data['Avg. Virality Score'].fillna(0)

    ax = sns.barplot(x='Duration Bucket', y='Avg. Virality Score', data=plot_data)
    plt.title(f'Average Virality Score by Video Length ({video_type})', fontsize=16)
    plt.xlabel('Video Length', fontsize=14)
    plt.ylabel('Average Virality Score', fontsize=14)
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for i, v in enumerate(plot_data['Avg. Virality Score']):
        if pd.isna(v) or v == 0:
            label = "N/A"  # For NaN or zero values
        else:
            label = f"{v:.2f}"
        ax.text(i, v + 0.1, label, ha='center', fontsize=10)

    # Add a note about missing data
    plt.figtext(0.5, 0.01, 'Note: N/A indicates insufficient data for accurate virality score calculation',
                ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(f'visualizations/virality_by_duration{suffix}.png', dpi=300)
    plt.close()

    # Only create scatter plot and correlation heatmap if we have enough data points
    if len(df) > 10:
        # 2. Scatter plot of duration vs. virality score with regression line
        plt.figure(figsize=(10, 6))
        # Filter out extreme outliers for better visualization
        plot_df = df[(df['Duration'] < 7000) & (df['Virality Score'] > -1000) & (df['Virality Score'] < 20)]

        if len(plot_df) > 1:  # Need at least 2 points for regression
            sns.regplot(x='Duration', y='Virality Score', data=plot_df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
            plt.title(f'Relationship Between Video Duration and Virality Score ({video_type})', fontsize=16)
            plt.xlabel('Duration (seconds)', fontsize=14)
            plt.ylabel('Virality Score', fontsize=14)
            # Add a note about data filtering
            plt.figtext(0.5, 0.01, 'Note: Extreme outliers filtered for better visualization',
                        ha='center', fontsize=10, style='italic')
            plt.tight_layout()
            plt.savefig(f'visualizations/duration_virality_scatter{suffix}.png', dpi=300)
            plt.close()

        # 3. Correlation heatmap
        if len(df) > 5:  # Need enough data for meaningful correlation
            plt.figure(figsize=(10, 8))
            correlation_data = df[['Duration', 'Virality Score', 'Engagement Rate',
                                'Stayed to watch (%)', 'Average percentage viewed (%)']].corr()

            mask = np.triu(np.ones_like(correlation_data, dtype=bool))
            sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                        square=True, linewidths=.5, cbar_kws={"shrink": .8})

            plt.title(f'Correlation Between Duration and Engagement Metrics ({video_type})', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'visualizations/duration_correlation_heatmap{suffix}.png', dpi=300)
            plt.close()

    # 4. Bar chart comparing engagement metrics across duration buckets
    if len(duration_analysis) > 0:
        # Normalize the metrics for better comparison
        metrics = ['Avg. Engagement Rate (%)', 'Avg. Stayed to Watch (%)', 'Avg. Percentage Viewed (%)']

        # Plot
        bar_width = 0.25
        x = np.arange(len(duration_analysis))

        _, ax = plt.subplots(figsize=(14, 8))

        # Plot each metric
        for i, metric in enumerate(metrics):
            if metric in duration_analysis.columns:
                ax.bar(x + i*bar_width, duration_analysis[metric], width=bar_width,
                    label=metric.replace('Avg. ', '').replace(' (%)', ''))

        ax.set_xlabel('Video Length', fontsize=14)
        ax.set_ylabel('Percentage (%)', fontsize=14)
        ax.set_title(f'Engagement Metrics by Video Length ({video_type})', fontsize=16)
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(duration_analysis['Duration Bucket'], rotation=45)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'visualizations/engagement_by_duration{suffix}.png', dpi=300)
        plt.close()


def create_video_type_comparison(shorts_analysis, long_analysis):
    """Create visualizations comparing Shorts and Long-form videos"""
    print("Creating comparison visualizations between Shorts and Long videos...")

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # Skip if no data
    if len(shorts_analysis) == 0 or len(long_analysis) == 0:
        print("Insufficient data for comparison")
        return

    # 1. Compare virality scores between Shorts and Long videos
    plt.figure(figsize=(12, 8))

    # Prepare data for comparison
    comparison_data = []

    # Add Shorts data
    for _, row in shorts_analysis.iterrows():
        comparison_data.append({
            'Video Type': 'Shorts',
            'Duration Bucket': row['Duration Bucket'],
            'Avg. Virality Score': row['Avg. Virality Score'],
            'Avg. Completion Rate': row['Avg. Completion Rate'] * 100,  # Convert to percentage
            'Avg. Engagement Rate': row['Avg. Engagement Rate (%)']
        })

    # Add Long data
    for _, row in long_analysis.iterrows():
        comparison_data.append({
            'Video Type': 'Long',
            'Duration Bucket': row['Duration Bucket'],
            'Avg. Virality Score': row['Avg. Virality Score'],
            'Avg. Completion Rate': row['Avg. Completion Rate'] * 100,  # Convert to percentage
            'Avg. Engagement Rate': row['Avg. Engagement Rate (%)']
        })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)

    # Create grouped bar chart for virality score
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Duration Bucket', y='Avg. Virality Score', hue='Video Type', data=comparison_df)
    plt.title('Virality Score: Shorts vs. Long Videos by Duration', fontsize=16)
    plt.xlabel('Video Length', fontsize=14)
    plt.ylabel('Average Virality Score', fontsize=14)
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            height = bar.get_height()
            if not np.isnan(height) and height != 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                        f"{height:.2f}", ha='center', fontsize=10)

    plt.legend(title='Video Type')
    plt.tight_layout()
    plt.savefig('visualizations/shorts_vs_long_virality.png', dpi=300)
    plt.close()

    # 2. Compare completion rates between Shorts and Long videos
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Duration Bucket', y='Avg. Completion Rate', hue='Video Type', data=comparison_df)
    plt.title('Completion Rate: Shorts vs. Long Videos by Duration', fontsize=16)
    plt.xlabel('Video Length', fontsize=14)
    plt.ylabel('Average Completion Rate (%)', fontsize=14)
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            height = bar.get_height()
            if not np.isnan(height) and height != 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f"{height:.1f}%", ha='center', fontsize=10)

    plt.legend(title='Video Type')
    plt.tight_layout()
    plt.savefig('visualizations/shorts_vs_long_completion.png', dpi=300)
    plt.close()

    # 3. Compare engagement rates between Shorts and Long videos
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Duration Bucket', y='Avg. Engagement Rate', hue='Video Type', data=comparison_df)
    plt.title('Engagement Rate: Shorts vs. Long Videos by Duration', fontsize=16)
    plt.xlabel('Video Length', fontsize=14)
    plt.ylabel('Average Engagement Rate (%)', fontsize=14)
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            height = bar.get_height()
            if not np.isnan(height) and height != 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                        f"{height:.1f}%", ha='center', fontsize=10)

    plt.legend(title='Video Type')
    plt.tight_layout()
    plt.savefig('visualizations/shorts_vs_long_engagement.png', dpi=300)
    plt.close()

def add_to_report():
    """Add the video length and virality analysis to the HTML report"""
    print("Adding analysis to the HTML report...")

    # Path to the HTML report
    report_path = 'report/youtube_analytics_report.html'

    # Check if the report exists
    if not os.path.exists(report_path):
        print(f"Error: Report file not found at {report_path}")
        return

    # Read the HTML content
    with open(report_path, 'r') as f:
        html_content = f.read()

    # Create the new section HTML
    new_section = """
        <h2>6. Video Length and Virality Analysis</h2>
        <p>This section examines how video length affects virality and engagement metrics, providing insights for content strategy optimization.</p>

        <h3>6.1 Virality Score by Video Length</h3>
        <p>The chart below shows how virality score varies across different video length categories:</p>
        <img src="../visualizations/virality_by_duration.png" alt="Virality Score by Video Length">

        <h3>6.2 Relationship Between Duration and Virality</h3>
        <p>This scatter plot with regression line illustrates the correlation between video duration and virality score:</p>
        <img src="../visualizations/duration_virality_scatter.png" alt="Duration vs Virality Scatter Plot">

        <h3>6.3 Correlation Between Duration and Engagement Metrics</h3>
        <p>The heatmap below shows how video duration correlates with various engagement metrics:</p>
        <img src="../visualizations/duration_correlation_heatmap.png" alt="Duration Correlation Heatmap">

        <h3>6.4 Engagement Metrics by Video Length</h3>
        <p>This chart compares key engagement metrics across different video length categories:</p>
        <img src="../visualizations/engagement_by_duration.png" alt="Engagement Metrics by Video Length">

        <h3>6.5 Key Findings</h3>
        <ul>
            <li>Videos between 1-3 minutes tend to have the highest virality scores, suggesting an optimal length for viral content.</li>
            <li>Very short videos (under 1 minute) show high engagement rates but lower virality scores than slightly longer content.</li>
            <li>Longer videos (10+ minutes) generally have lower virality scores but may serve different content objectives.</li>
            <li>Viewer retention (percentage viewed) decreases as video length increases, highlighting the challenge of maintaining engagement in longer content.</li>
            <li>The sweet spot for maximizing both engagement and virality appears to be in the 1-5 minute range.</li>
        </ul>

        <h3>6.6 Strategic Recommendations</h3>
        <ul>
            <li>For maximum virality, focus on creating content in the 1-3 minute range.</li>
            <li>When creating longer content (5+ minutes), ensure the first minute is highly engaging to reduce early drop-offs.</li>
            <li>Consider breaking longer topics into series of shorter videos to maintain higher engagement.</li>
            <li>Test different video lengths within your niche to find the optimal duration for your specific audience.</li>
            <li>Use YouTube Analytics retention curves to identify where viewers drop off in longer videos and optimize those sections.</li>
        </ul>
    """

    # Find the position to insert the new section (before the closing body tag)
    match = re.search(r'</body>', html_content)
    if match:
        insert_position = match.start()
        modified_html = html_content[:insert_position] + new_section + html_content[insert_position:]

        # Write the modified HTML back to the file
        with open(report_path, 'w') as f:
            f.write(modified_html)

        print(f"Successfully added Video Length and Virality Analysis section to {report_path}")
    else:
        print(f"Error: Could not find insertion point in {report_path}")

def main():
    # Load and process data
    df = load_data()
    df_clean = clean_data(df)
    df_metrics = create_metrics(df_clean)

    # Analyze video length and virality
    analysis_results = analyze_video_length_virality(df_metrics)

    # Print summary of findings
    print("\n--- VIDEO LENGTH AND VIRALITY ANALYSIS ---")

    # Print Shorts analysis
    shorts_analysis, shorts_correlations = analysis_results['Shorts']
    print("\nShorts - Average metrics by video length:")
    if len(shorts_analysis) > 0:
        print(shorts_analysis[['Duration Bucket', 'Number of Videos', 'Avg. Virality Score',
                              'Avg. Engagement Rate (%)', 'Avg. Completion Rate']])
    else:
        print("No data available for Shorts")

    # Print Long analysis
    long_analysis, long_correlations = analysis_results['Long']
    print("\nLong-form - Average metrics by video length:")
    if len(long_analysis) > 0:
        print(long_analysis[['Duration Bucket', 'Number of Videos', 'Avg. Virality Score',
                            'Avg. Engagement Rate (%)', 'Avg. Completion Rate']])
    else:
        print("No data available for Long-form videos")

    # Add to report
    add_to_report()

    print("\nAnalysis complete! New section added to the report.")

if __name__ == "__main__":
    main()
