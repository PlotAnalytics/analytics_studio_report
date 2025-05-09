import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

# Load the data
df = pd.read_csv('youtube_analytics_master.csv')

# Data cleaning and preparation
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
    engagement_cols = ['Comments added', 'Likes', 'Engaged views', 'Views', 'Watch time (hours)', 'Subscribers', 'Impressions']
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

# Create calculated metrics
def create_metrics(df):
    # Make a copy to avoid warnings
    df = df.copy()

    # Engagement ratios
    df['Comments to Views Ratio'] = df['Comments added'] / df['Views'] * 100  # Percentage of viewers who commented
    df['Likes to Views Ratio'] = df['Likes'] / df['Views'] * 100  # Percentage of viewers who liked
    df['Engaged Views Ratio'] = df['Engaged views'] / df['Views'] * 100  # Percentage of views that were engaged
    df['Subscribers Gained per 1000 Views'] = df['Subscribers'] / df['Views'] * 1000  # New subscribers per 1000 views
    df['Comments to Likes Ratio'] = df['Comments added'] / df['Likes'] * 100  # Comments as percentage of likes
    df['Engagement Rate'] = (df['Comments added'] + df['Likes']) / df['Views'] * 100  # Combined engagement rate

    # Retention metrics
    df['Swipe Away Ratio'] = 100 - df['Stayed to watch (%)']  # Percentage of viewers who swiped away
    df['Retention Efficiency'] = df['Average percentage viewed (%)'] / df['Duration']  # Retention relative to duration
    df['Completion Rate'] = df['Average percentage viewed (%)'] / 100  # Fraction of video watched on average

    # Impression efficiency
    df['Views per Impression'] = df['Views'] / df['Impressions'] * 100  # Views as percentage of impressions
    df['Engaged Views per Impression'] = df['Engaged views'] / df['Impressions'] * 100  # Engaged views as percentage of impressions

    # Watch time efficiency
    df['Watch Time per View (seconds)'] = df['Watch time (hours)'] * 3600 / df['Views']  # Average seconds watched per view
    df['Watch Time Efficiency'] = df['Watch time (hours)'] / df['Duration'] * 3600  # Watch time relative to total possible

    # Virality metrics
    df['Virality Score'] = (df['Engaged views'] * 0.5 + df['Likes'] * 0.3 + df['Comments added'] * 0.2) / df['Views'] * 100  # Weighted engagement score
    df['Growth Potential'] = df['Subscribers Gained per 1000 Views'] * df['Engagement Rate']  # Combined growth metric

    # Categorize videos into view buckets
    conditions = [
        (df['Views'] < 100000),
        (df['Views'] >= 100000) & (df['Views'] < 500000),
        (df['Views'] >= 500000) & (df['Views'] < 1000000),
        (df['Views'] >= 1000000) & (df['Views'] < 5000000),
        (df['Views'] >= 5000000)
    ]

    values = [
        'Under 100K',
        '100K to 500K',
        '500K to 1M',
        '1M to 5M',
        '5M plus'
    ]

    df['View Bucket'] = np.select(conditions, values, default='Unknown')

    # Create time period category
    df['Time Period'] = df['Time Range'].apply(lambda x: 'Jan-Mar' if 'Jan' in str(x) else 'April')

    # Create engaged views buckets
    engaged_conditions = [
        (df['Engaged views'] < 100000),
        (df['Engaged views'] >= 100000) & (df['Engaged views'] < 500000),
        (df['Engaged views'] >= 500000) & (df['Engaged views'] < 1000000),
        (df['Engaged views'] >= 1000000) & (df['Engaged views'] < 5000000),
        (df['Engaged views'] >= 5000000)
    ]

    df['Engaged View Bucket'] = np.select(engaged_conditions, values, default='Unknown')

    # Create a dictionary of metric definitions
    metric_definitions = {
        # Original metrics
        'Content': 'Video ID from YouTube. Unique identifier for each video.',
        'Account': 'YouTube channel name. The name of the channel that published the video.',
        'Video Type': 'Format of the video (Shorts or Long). Shorts are vertical, short-form videos (≤60 seconds), while Long videos are traditional horizontal format.',
        'Time Range': 'Time period when data was collected (Jan 1 - March 30 or April 1 - 30). Used to segment data for temporal analysis.',
        'Video title': 'Title of the YouTube video. The headline displayed above the video.',
        'Video publish time': 'Date when the video was published. The timestamp when the video was made public.',
        'Duration': 'Length of the video in seconds. Total runtime of the video content.',
        'Stayed to watch (%)': 'Percentage of viewers who did not immediately swipe away. Measures initial retention in the first few seconds.',
        'Comments added': 'Total number of comments on the video. Count of user-generated text responses.',
        'Likes': 'Total number of likes on the video. Count of positive reactions from viewers.',
        'Average percentage viewed (%)': 'Average percentage of the video that viewers watched. Measures overall retention throughout the video.',
        'Engaged views': 'Number of views with active engagement (likes, comments, shares). Key metric for measuring meaningful viewership.',
        'Views': 'Total number of views on the video. Raw count of video plays.',
        'Watch time (hours)': 'Total hours spent by all viewers watching the video. Cumulative viewing duration across all viewers.',
        'Subscribers': 'Net subscribers gained from the video. New subscribers minus lost subscribers attributed to this video.',
        'Average view duration': 'Average time viewers spent watching the video. Total watch time divided by number of views.',
        'Impressions': 'Number of times the video thumbnail was shown to potential viewers. Measures exposure in feeds, search, etc.',
        'Impressions click-through rate (%)': 'Percentage of impressions that turned into views. Formula: (Views / Impressions) × 100',

        # Calculated metrics
        'Comments to Views Ratio': 'Percentage of viewers who commented on the video. Formula: (Comments added / Views) × 100. Measures audience interaction propensity.',
        'Likes to Views Ratio': 'Percentage of viewers who liked the video. Formula: (Likes / Views) × 100. Measures positive sentiment and engagement rate.',
        'Engaged Views Ratio': 'Percentage of total views that were engaged (actively interacted). Formula: (Engaged views / Views) × 100. Measures quality of viewership.',
        'Subscribers Gained per 1000 Views': 'Number of new subscribers gained per 1000 views. Formula: (Subscribers / Views) × 1000. Measures conversion efficiency.',
        'Comments to Likes Ratio': 'Number of comments as a percentage of likes. Formula: (Comments added / Likes) × 100. Measures discussion generation relative to positive sentiment.',
        'Engagement Rate': 'Combined rate of likes and comments relative to views. Formula: ((Comments added + Likes) / Views) × 100. Comprehensive engagement metric.',
        'Swipe Away Ratio': 'Percentage of viewers who immediately swiped away. Formula: 100 - Stayed to watch (%). Measures initial rejection rate.',
        'Retention Efficiency': 'How well the video retains viewers relative to its duration. Formula: Average percentage viewed (%) / Duration. Normalizes retention for videos of different lengths.',
        'Completion Rate': 'Fraction of the video watched on average (0-1). Formula: Average percentage viewed (%) / 100. Simplified metric for retention analysis.',
        'Views per Impression': 'Percentage of impressions that converted to views. Formula: (Views / Impressions) × 100. Measures thumbnail and title effectiveness.',
        'Engaged Views per Impression': 'Percentage of impressions that converted to engaged views. Formula: (Engaged views / Impressions) × 100. Measures high-quality conversion rate.',
        'Watch Time per View (seconds)': 'Average number of seconds watched per view. Formula: (Watch time (hours) × 3600) / Views. Detailed retention metric.',
        'Watch Time Efficiency': 'Watch time relative to maximum possible watch time. Formula: (Watch time (hours) / Duration) × 3600. Measures how effectively content keeps viewers watching.',
        'Virality Score': 'Weighted score combining key engagement metrics. Formula: ((Engaged views × 0.5 + Likes × 0.3 + Comments added × 0.2) / Views) × 100. Custom metric to predict viral potential.',
        'Growth Potential': 'Combined metric of subscriber growth and engagement. Formula: Subscribers Gained per 1000 Views × Engagement Rate. Predicts channel growth impact.',
        'View Bucket': 'Category based on total view count. Videos grouped into: Under 100K, 100K to 500K, 500K to 1M, 1M to 5M, 5M plus.',
        'Engaged View Bucket': 'Category based on engaged view count. Videos grouped into: Under 100K, 100K to 500K, 500K to 1M, 1M to 5M, 5M plus.',
        'Time Period': 'Simplified time period (Jan-Mar or April). Derived from Time Range for easier temporal comparison.'
    }

    # Save metric definitions to a CSV file
    pd.DataFrame({
        'Metric': list(metric_definitions.keys()),
        'Definition': list(metric_definitions.values())
    }).to_csv('metric_definitions.csv', index=False)

    return df

# Analyze metrics by engaged view bucket
def analyze_by_view_bucket(df):
    # Group by engaged view bucket and calculate mean of key metrics
    metrics_by_bucket = df.groupby('Engaged View Bucket').agg({
        'Engaged views': 'mean',
        'Views': 'mean',
        'Comments added': 'mean',
        'Likes': 'mean',
        'Watch time (hours)': 'mean',
        'Subscribers': 'mean',
        'Stayed to watch (%)': 'mean',
        'Average percentage viewed (%)': 'mean',
        'Comments to Views Ratio': 'mean',
        'Likes to Views Ratio': 'mean',
        'Engaged Views Ratio': 'mean',
        'Subscribers Gained per 1000 Views': 'mean',
        'Comments to Likes Ratio': 'mean',
        'Engagement Rate': 'mean',
        'Swipe Away Ratio': 'mean',
        'Retention Efficiency': 'mean',
        'Completion Rate': 'mean',
        'Views per Impression': 'mean',
        'Engaged Views per Impression': 'mean',
        'Watch Time per View (seconds)': 'mean',
        'Watch Time Efficiency': 'mean',
        'Virality Score': 'mean',
        'Growth Potential': 'mean',
        'Impressions click-through rate (%)': 'mean'
    }).reset_index()

    # Sort by Engaged views in descending order
    metrics_by_bucket = metrics_by_bucket.sort_values('Engaged views', ascending=False).reset_index(drop=True)

    # Rename the column for consistency in output
    metrics_by_bucket = metrics_by_bucket.rename(columns={'Engaged View Bucket': 'View Bucket'})

    return metrics_by_bucket

# Compare short vs long videos
def compare_video_types(df):
    # Group by video type and calculate mean of key metrics
    metrics_by_type = df.groupby(['Video Type', 'Time Period']).agg({
        'Engaged views': 'mean',
        'Views': 'mean',
        'Comments added': 'mean',
        'Likes': 'mean',
        'Watch time (hours)': 'mean',
        'Subscribers': 'mean',
        'Stayed to watch (%)': 'mean',
        'Average percentage viewed (%)': 'mean',
        'Comments to Views Ratio': 'mean',
        'Likes to Views Ratio': 'mean',
        'Engaged Views Ratio': 'mean',
        'Subscribers Gained per 1000 Views': 'mean',
        'Comments to Likes Ratio': 'mean',
        'Engagement Rate': 'mean',
        'Swipe Away Ratio': 'mean',
        'Retention Efficiency': 'mean',
        'Completion Rate': 'mean',
        'Views per Impression': 'mean',
        'Engaged Views per Impression': 'mean',
        'Watch Time per View (seconds)': 'mean',
        'Watch Time Efficiency': 'mean',
        'Virality Score': 'mean',
        'Growth Potential': 'mean',
        'Impressions click-through rate (%)': 'mean'
    }).reset_index()

    # Sort by Video Type and then by Engaged views within each type
    metrics_by_type = metrics_by_type.sort_values(['Video Type', 'Engaged views'], ascending=[True, False]).reset_index(drop=True)

    return metrics_by_type

# Analyze Thumbsup Stories account specifically
def analyze_thumbsup_stories(df):
    # Filter for Thumbsup Stories account
    thumbsup_df = df[df['Account'] == 'Thumbsup Stories']

    # Group by video type and time period
    thumbsup_analysis = thumbsup_df.groupby(['Video Type', 'Time Period']).agg({
        'Content': 'count',  # Count of videos
        'Engaged views': ['mean', 'sum'],
        'Views': ['mean', 'sum'],
        'Comments added': ['mean', 'sum'],
        'Likes': ['mean', 'sum'],
        'Watch time (hours)': ['mean', 'sum'],
        'Subscribers': ['mean', 'sum'],
        'Stayed to watch (%)': 'mean',
        'Average percentage viewed (%)': 'mean',
        'Comments to Views Ratio': 'mean',
        'Likes to Views Ratio': 'mean',
        'Engaged Views Ratio': 'mean',
        'Subscribers Gained per 1000 Views': 'mean',
        'Comments to Likes Ratio': 'mean',
        'Engagement Rate': 'mean',
        'Swipe Away Ratio': 'mean',
        'Retention Efficiency': 'mean',
        'Completion Rate': 'mean',
        'Virality Score': 'mean',
        'Growth Potential': 'mean'
    }).reset_index()

    return thumbsup_analysis

# Determine metrics that correlate with virality
def analyze_virality_metrics(df):
    # Select only numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Make sure 'Engaged views' is in the numeric columns
    if 'Engaged views' not in numeric_cols:
        print("Warning: 'Engaged views' column is not numeric. Correlation analysis may not be accurate.")
        return {
            'high_impact': [],
            'soft_impact': [],
            'no_impact': [],
            'correlation_values': pd.Series()
        }

    # Calculate correlation with engaged views using only numeric columns
    correlation_with_engaged_views = df[numeric_cols].corr()['Engaged views'].sort_values(ascending=False)

    # Categorize metrics by impact level
    high_impact = correlation_with_engaged_views[correlation_with_engaged_views > 0.7].index.tolist()
    soft_impact = correlation_with_engaged_views[(correlation_with_engaged_views > 0.3) & (correlation_with_engaged_views <= 0.7)].index.tolist()
    no_impact = correlation_with_engaged_views[correlation_with_engaged_views <= 0.3].index.tolist()

    # Remove 'Engaged views' from high_impact as it's the target variable
    if 'Engaged views' in high_impact:
        high_impact.remove('Engaged views')

    return {
        'high_impact': high_impact,
        'soft_impact': soft_impact,
        'no_impact': no_impact,
        'correlation_values': correlation_with_engaged_views
    }

# Main analysis function
def main():
    print("Loading and cleaning data...")
    df_clean = clean_data(df)

    print("Creating calculated metrics...")
    df_metrics = create_metrics(df_clean)

    print("\n--- ANALYSIS BY VIEW BUCKET ---")
    metrics_by_bucket = analyze_by_view_bucket(df_metrics)
    print(metrics_by_bucket)

    print("\n--- SHORT VS LONG VIDEOS COMPARISON ---")
    video_type_comparison = compare_video_types(df_metrics)
    print(video_type_comparison)

    print("\n--- THUMBSUP STORIES ACCOUNT ANALYSIS ---")
    thumbsup_analysis = analyze_thumbsup_stories(df_metrics)
    print(thumbsup_analysis)

    print("\n--- VIRALITY METRICS ANALYSIS ---")
    virality_metrics = analyze_virality_metrics(df_metrics)

    print("High Impact Metrics (Strong correlation with virality):")
    for metric in virality_metrics['high_impact']:
        print(f"- {metric}: {virality_metrics['correlation_values'][metric]:.4f}")

    print("\nSoft Impact Metrics (Moderate correlation with virality):")
    for metric in virality_metrics['soft_impact']:
        print(f"- {metric}: {virality_metrics['correlation_values'][metric]:.4f}")

    print("\nNo Impact Metrics (Weak or no correlation with virality):")
    for metric in virality_metrics['no_impact'][:10]:  # Show only first 10 to keep output manageable
        print(f"- {metric}: {virality_metrics['correlation_values'][metric]:.4f}")

    # Save results to CSV files
    metrics_by_bucket.to_csv('results_by_view_bucket.csv', index=False)
    video_type_comparison.to_csv('results_short_vs_long.csv', index=False)
    thumbsup_analysis.to_csv('results_thumbsup_stories.csv', index=False)

    # Save correlation values
    pd.Series(virality_metrics['correlation_values']).to_csv('results_virality_correlations.csv')

    print("\nAnalysis complete. Results saved to CSV files.")

if __name__ == "__main__":
    main()
