import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import matplotlib.ticker as ticker

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# Load the data and processed metrics
def load_processed_data():
    # Load the original data
    df = pd.read_csv('youtube_analytics_master.csv')

    # Clean and process the data (using functions from the analysis script)
    df_clean = clean_data(df)
    df_metrics = create_metrics(df_clean)

    # Create engaged views buckets
    engaged_conditions = [
        (df_metrics['Engaged views'] < 100000),
        (df_metrics['Engaged views'] >= 100000) & (df_metrics['Engaged views'] < 500000),
        (df_metrics['Engaged views'] >= 500000) & (df_metrics['Engaged views'] < 1000000),
        (df_metrics['Engaged views'] >= 1000000) & (df_metrics['Engaged views'] < 5000000),
        (df_metrics['Engaged views'] >= 5000000)
    ]

    values = [
        'Under 100K',
        '100K to 500K',
        '500K to 1M',
        '1M to 5M',
        '5M plus'
    ]

    df_metrics['Engaged View Bucket'] = np.select(engaged_conditions, values, default='Unknown')

    return df_metrics

# Data cleaning function (same as in analysis script)
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

# Create calculated metrics function (same as in analysis script)
def create_metrics(df):
    # Make a copy to avoid warnings
    df = df.copy()

    # Engagement ratios
    df['Comments to Views Ratio'] = df['Comments added'] / df['Views'] * 100
    df['Likes to Views Ratio'] = df['Likes'] / df['Views'] * 100
    df['Engaged Views Ratio'] = df['Engaged views'] / df['Views'] * 100
    df['Subscribers Gained per 1000 Views'] = df['Subscribers'] / df['Views'] * 1000

    # Retention metrics
    df['Swipe Away Ratio'] = 100 - df['Stayed to watch (%)']
    df['Retention Efficiency'] = df['Average percentage viewed (%)'] / df['Duration']

    # Impression efficiency
    df['Views per Impression'] = df['Views'] / df['Impressions'] * 100

    # Watch time efficiency
    df['Watch Time per View (seconds)'] = df['Watch time (hours)'] * 3600 / df['Views']

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

    return df

# Visualization functions
def plot_view_bucket_distribution(df):
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
    # Define key metrics to plot
    metrics = [
        'Likes to Views Ratio',
        'Comments to Views Ratio',
        'Subscribers Gained per 1000 Views',
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

def plot_short_vs_long_comparison(df):
    # Define key metrics to plot
    metrics = [
        'Engaged views',  # Changed from Views to Engaged views
        'Likes to Views Ratio',
        'Comments to Views Ratio',
        'Subscribers Gained per 1000 Views',
        'Stayed to watch (%)',
        'Average percentage viewed (%)'
    ]

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()

    # Define colors for better visual distinction
    colors = sns.color_palette("Set2", 2)  # One color for each time period

    for i, metric in enumerate(metrics):
        # Group by video type and time period, calculate mean of the metric
        metric_by_type = df.groupby(['Video Type', 'Time Period'])[metric].mean().reset_index()

        # Plot with custom colors
        bars = sns.barplot(x='Video Type', y=metric, hue='Time Period', data=metric_by_type, ax=axes[i], palette=colors)

        # Add value annotations on each bar
        # Get the bars from the plot
        for j, bar in enumerate(bars.patches):
            # Calculate the value for this bar
            value = bar.get_height()
            if pd.notna(value):  # Check if value is not NaN
                # Format value based on metric type
                if '%' in metric:
                    value_text = f'{value:.1f}%'
                elif value > 1000000:
                    value_text = f'{value/1000000:.1f}M'
                elif value > 1000:
                    value_text = f'{value/1000:.1f}K'
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

        axes[i].set_title(f'{metric} by Video Type and Time Period', fontsize=14, pad=10)
        axes[i].set_xlabel('Video Type', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=12)

        # Add grid lines for better readability
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        # Format y-axis as percentage for percentage metrics
        if '%' in metric:
            axes[i].yaxis.set_major_formatter(ticker.PercentFormatter(decimals=1))

        # Improve legend
        axes[i].legend(title='Time Period', fontsize=10, title_fontsize=12)

    plt.tight_layout()

    # Save the figure
    plt.savefig('visualizations/short_vs_long_comparison.png', dpi=300)
    plt.close()

def plot_thumbsup_stories_analysis(df):
    # Filter for Thumbsup Stories account
    thumbsup_df = df[df['Account'] == 'Thumbsup Stories']

    # Check if there are any long videos in Jan-Mar
    long_jan_mar = thumbsup_df[(thumbsup_df['Video Type'] == 'Long') &
                              (thumbsup_df['Time Period'] == 'Jan-Mar')]

    print(f"Number of Thumbsup Stories long videos in Jan-Mar: {len(long_jan_mar)}")

    # Plot video count by type and time period
    plt.figure(figsize=(12, 8))

    # Count videos by type and time period
    video_counts = thumbsup_df.groupby(['Video Type', 'Time Period']).size().reset_index(name='Count')

    # Plot with custom colors
    colors = sns.color_palette("Set2", 2)  # One color for each time period
    ax = sns.barplot(x='Video Type', y='Count', hue='Time Period', data=video_counts, palette=colors)

    # Add value annotations on each bar
    for i, bar in enumerate(ax.patches):
        # Calculate the value for this bar
        value = bar.get_height()
        if pd.notna(value):  # Check if value is not NaN
            # Format value
            value_text = f'{int(value)}'

            # Position text in the middle of the bar
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = bar.get_height() / 2

            # Add text with white outline for better visibility
            ax.text(text_x, text_y, value_text,
                  ha='center', va='center',
                  fontweight='bold', color='white',
                  bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    plt.title('Thumbsup Stories: Video Count by Type and Time Period', fontsize=16, pad=15)
    plt.xlabel('Video Type', fontsize=14)
    plt.ylabel('Number of Videos', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Time Period', fontsize=12, title_fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig('visualizations/thumbsup_video_count.png', dpi=300)
    plt.close()

    # Plot average engaged views by type and time period
    plt.figure(figsize=(12, 8))

    # Calculate average engaged views by type and time period
    avg_engaged_views = thumbsup_df.groupby(['Video Type', 'Time Period'])['Engaged views'].mean().reset_index()

    # Plot with custom colors
    ax = sns.barplot(x='Video Type', y='Engaged views', hue='Time Period', data=avg_engaged_views, palette=colors)

    # Add value annotations on each bar
    for i, bar in enumerate(ax.patches):
        # Calculate the value for this bar
        value = bar.get_height()
        if pd.notna(value):  # Check if value is not NaN
            # Format value
            if value > 1000000:
                value_text = f'{value/1000000:.1f}M'
            elif value > 1000:
                value_text = f'{value/1000:.1f}K'
            else:
                value_text = f'{value:.0f}'

            # Position text in the middle of the bar
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = bar.get_height() / 2

            # Add text with white outline for better visibility
            ax.text(text_x, text_y, value_text,
                  ha='center', va='center',
                  fontweight='bold', color='white',
                  bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

    plt.title('Thumbsup Stories: Average Engaged Views by Type and Time Period', fontsize=16, pad=15)
    plt.xlabel('Video Type', fontsize=14)
    plt.ylabel('Average Engaged Views', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Time Period', fontsize=12, title_fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig('visualizations/thumbsup_avg_views.png', dpi=300)
    plt.close()

    # Add a new plot for engagement metrics comparison
    plt.figure(figsize=(14, 10))

    # Select key engagement metrics that exist in the dataframe
    available_metrics = thumbsup_df.columns.tolist()
    potential_metrics = ['Engagement Rate', 'Likes to Views Ratio', 'Comments to Views Ratio',
                         'Subscribers Gained per 1000 Views', 'Virality Score']

    engagement_metrics = [metric for metric in potential_metrics if metric in available_metrics]

    if len(engagement_metrics) < 2:
        print("Warning: Not enough engagement metrics available for comparison plot")
        return

    # Let's simplify and just create a new plot for each metric
    for metric in engagement_metrics:
        plt.figure(figsize=(10, 6))

        # Calculate mean of the metric for each video type and time period
        metric_data = thumbsup_df.groupby(['Video Type', 'Time Period'])[metric].mean().reset_index()

        # Plot
        ax = sns.barplot(x='Video Type', y=metric, hue='Time Period', data=metric_data, palette='Set2')

        # Add value annotations on each bar
        for i, bar in enumerate(ax.patches):
            # Calculate the value for this bar
            value = bar.get_height()
            if pd.notna(value):  # Check if value is not NaN
                # Format value
                value_text = f'{value:.2f}'

                # Position text in the middle of the bar
                text_x = bar.get_x() + bar.get_width() / 2
                text_y = bar.get_height() / 2

                # Add text with white outline for better visibility
                ax.text(text_x, text_y, value_text,
                      ha='center', va='center',
                      fontweight='bold', color='white',
                      bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))

        plt.title(f'Thumbsup Stories: {metric} by Video Type', fontsize=16, pad=15)
        plt.xlabel('Video Type', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Time Period', fontsize=12, title_fontsize=14)
        plt.tight_layout()

        # Save the figure
        plt.savefig(f'visualizations/thumbsup_{metric.replace(" ", "_").replace("%", "pct").lower()}.png', dpi=300)
        plt.close()



def plot_correlation_heatmap(df):
    # Select relevant metrics for correlation analysis that are numeric
    desired_metrics = [
        'Engaged views', 'Views', 'Comments added', 'Likes', 'Watch time (hours)', 'Subscribers',
        'Stayed to watch (%)', 'Average percentage viewed (%)', 'Duration',
        'Comments to Views Ratio', 'Likes to Views Ratio', 'Engaged Views Ratio',
        'Subscribers Gained per 1000 Views', 'Comments to Likes Ratio', 'Engagement Rate',
        'Swipe Away Ratio', 'Retention Efficiency', 'Completion Rate',
        'Views per Impression', 'Engaged Views per Impression', 'Watch Time per View (seconds)',
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

def plot_impact_scale(df):
    """Create a visual scale showing all variables and their impact levels on virality"""
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

# Main function to run all visualizations
def main():
    print("Loading and processing data...")
    df = load_processed_data()

    print("Generating visualizations...")

    print("1. Plotting view bucket distribution...")
    plot_view_bucket_distribution(df)

    print("2. Plotting metrics by view bucket...")
    plot_metrics_by_view_bucket(df)

    print("3. Plotting short vs long comparison...")
    plot_short_vs_long_comparison(df)

    print("4. Plotting Thumbsup Stories analysis...")
    plot_thumbsup_stories_analysis(df)

    print("5. Plotting correlation heatmap...")
    plot_correlation_heatmap(df)

    print("6. Plotting impact scale visualization...")
    plot_impact_scale(df)

    print("Visualizations complete. Results saved to 'visualizations' directory.")

if __name__ == "__main__":
    main()
