import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

# Import functions from analysis script
from youtube_analytics_analysis import clean_data, create_metrics, analyze_by_view_bucket, compare_video_types, analyze_thumbsup_stories, analyze_virality_metrics

# Create output directory for report
os.makedirs('report', exist_ok=True)

def generate_html_report(df):
    """Generate a comprehensive HTML report with all analysis results"""

    # Process data
    df_clean = clean_data(df)
    df_metrics = create_metrics(df_clean)

    # Run analyses
    metrics_by_bucket = analyze_by_view_bucket(df_metrics)
    video_type_comparison = compare_video_types(df_metrics)
    thumbsup_analysis = analyze_thumbsup_stories(df_metrics)
    virality_metrics = analyze_virality_metrics(df_metrics)

    # Start building HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Analytics Deep Dive Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #1a73e8;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .metric-box {{
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
            }}
            .high-impact {{
                color: #0d652d;
                font-weight: bold;
            }}
            .soft-impact {{
                color: #f9a825;
                font-weight: bold;
            }}
            .no-impact {{
                color: #c62828;
            }}
            .summary-box {{
                background-color: #e8f0fe;
                border-left: 5px solid #1a73e8;
                padding: 15px;
                margin-bottom: 20px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <h1>YouTube Analytics Deep Dive Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>This report provides a comprehensive analysis of YouTube performance metrics across different accounts,
            video types, and time periods. Special attention is given to metrics that determine virality and the
            performance of Thumbsup Stories account.</p>
        </div>

        <h2>1. Overview of Data</h2>
        <p>The analysis covers {len(df)} videos across multiple YouTube accounts:</p>
        <ul>
    """

    # Add account statistics
    for account, count in df['Account'].value_counts().items():
        html += f"<li><strong>{account}</strong>: {count} videos</li>\n"

    html += f"""
        </ul>
        <p>Time periods analyzed:</p>
        <ul>
    """

    # Add time period statistics
    for period, count in df_metrics['Time Period'].value_counts().items():
        html += f"<li><strong>{period}</strong>: {count} videos</li>\n"

    html += f"""
        </ul>
        <p>Video types:</p>
        <ul>
    """

    # Add video type statistics
    for vtype, count in df_metrics['Video Type'].value_counts().items():
        html += f"<li><strong>{vtype}</strong>: {count} videos</li>\n"

    html += f"""
        </ul>

        <h2>2. Distribution by View Count Buckets</h2>
        <p>Videos are categorized into 5 view count buckets for analysis:</p>
    """

    # Add view bucket statistics
    bucket_counts = df_metrics['View Bucket'].value_counts().sort_index()
    html += "<table>\n<tr><th>View Bucket</th><th>Number of Videos</th><th>Percentage</th></tr>\n"

    bucket_order = ['Under 100K', '100K to 500K', '500K to 1M', '1M to 5M', '5M plus']
    for bucket in bucket_order:
        if bucket in bucket_counts:
            count = bucket_counts[bucket]
            percentage = count / len(df_metrics) * 100
            html += f"<tr><td>{bucket}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>\n"

    html += "</table>\n"

    # Add visualization if available
    html += """
        <img src="../visualizations/view_bucket_distribution.png" alt="View Bucket Distribution">

        <h2>3. Key Metrics by View Count Bucket</h2>
        <p>Analysis of how key performance metrics vary across different view count buckets:</p>
    """

    # Add metrics by bucket table
    html += "<table>\n<tr><th>View Bucket</th>"
    for col in metrics_by_bucket.columns[1:]:
        html += f"<th>{col}</th>"
    html += "</tr>\n"

    for _, row in metrics_by_bucket.iterrows():
        html += f"<tr><td>{row['View Bucket']}</td>"
        for col in metrics_by_bucket.columns[1:]:
            value = row[col]
            if isinstance(value, (int, float)):
                # Special case for Subscribers Gained per 1000 Engaged Views - not a percentage
                if col == 'Subscribers Gained per 1000 Engaged Views':
                    html += f"<td>{value:.2f}</td>"
                elif 'Ratio' in col or 'percentage' in col.lower() or '%' in col:
                    html += f"<td>{value:.2f}%</td>"
                elif value > 1000000:
                    html += f"<td>{value/1000000:.2f}M</td>"
                elif value > 1000:
                    html += f"<td>{value/1000:.2f}K</td>"
                else:
                    html += f"<td>{value:.2f}</td>"
            else:
                html += f"<td>{value}</td>"
        html += "</tr>
"

    html += "</table>\n"

    # Add visualization if available
    html += """
        <img src="../visualizations/metrics_by_view_bucket.png" alt="Metrics by View Bucket">

        <h2>4. Short vs. Long Video Comparison</h2>
        <p>Comparison of performance metrics between short-form and long-form content:</p>
    """

    # Add short vs long comparison table
    html += "<table>\n<tr><th>Video Type</th><th>Time Period</th>"
    for col in video_type_comparison.columns[2:]:
        html += f"<th>{col}</th>"
    html += "</tr>\n"

    for _, row in video_type_comparison.iterrows():
        html += f"<tr><td>{row['Video Type']}</td><td>{row['Time Period']}</td>"
        for col in video_type_comparison.columns[2:]:
            value = row[col]
            if isinstance(value, (int, float)):
                # Special case for Subscribers Gained per 1000 Engaged Views - not a percentage
                if col == 'Subscribers Gained per 1000 Engaged Views':
                    html += f"<td>{value:.2f}</td>"
                elif 'Ratio' in col or 'percentage' in col.lower() or '%' in col:
                    html += f"<td>{value:.2f}%</td>"
                elif value > 1000000:
                    html += f"<td>{value/1000000:.2f}M</td>"
                elif value > 1000:
                    html += f"<td>{value/1000:.2f}K</td>"
                else:
                    html += f"<td>{value:.2f}</td>"
            else:
                html += f"<td>{value}</td>"
        html += "</tr>
"

    html += "</table>\n"

    # Add visualization if available
    html += """
        <img src="../visualizations/short_vs_long_comparison.png" alt="Short vs Long Comparison">

        <h2>5. Thumbsup Stories Account Analysis</h2>
        <p>Special focus on Thumbsup Stories account performance:</p>
    """

    # Check if there are any long videos for Thumbsup Stories in Jan-Mar
    thumbsup_df = df_metrics[df_metrics['Account'] == 'Thumbsup Stories']
    long_jan_mar = thumbsup_df[(thumbsup_df['Video Type'] == 'Long') &
                              (thumbsup_df['Time Period'] == 'Jan-Mar')]

    html += f"""
        <div class="summary-box">
            <h3>Key Findings for Thumbsup Stories</h3>
            <p>Number of Thumbsup Stories long videos in Jan-Mar: {len(long_jan_mar)}</p>
            <p>As noted, there {'are no' if len(long_jan_mar) == 0 else 'are very few'} long-form videos for Thumbsup Stories in the Jan-Mar period,
            while April shows some long-form content but with relatively low performance.</p>
        </div>
    """

    # Add visualizations if available
    html += """
        <img src="../visualizations/thumbsup_video_count.png" alt="Thumbsup Stories Video Count">
        <img src="../visualizations/thumbsup_avg_views.png" alt="Thumbsup Stories Average Views">

        <h2>6. Metrics That Determine Virality</h2>
        <p>Based on correlation analysis with <strong>engaged views</strong> (a better indicator of virality than raw views),
        we've identified which metrics have the strongest relationship with video virality:</p>

        <img src="../visualizations/impact_scale.png" alt="Impact Scale of Metrics on Virality">

        <div class="metric-box">
            <h3>High Impact Metrics (Strong correlation with virality)</h3>
            <ul>
    """

    # Add high impact metrics
    for metric in virality_metrics['high_impact']:
        corr_value = virality_metrics['correlation_values'][metric]
        html += f"<li class='high-impact'>{metric}: {corr_value:.4f}</li>\n"

    html += """
            </ul>
        </div>

        <div class="metric-box">
            <h3>Moderate Impact Metrics (Medium correlation with virality)</h3>
            <ul>
    """

    # Add soft impact metrics
    for metric in virality_metrics['soft_impact']:
        corr_value = virality_metrics['correlation_values'][metric]
        html += f"<li class='soft-impact'>{metric}: {corr_value:.4f}</li>\n"

    html += """
            </ul>
        </div>

        <div class="metric-box">
            <h3>Low/No Impact Metrics (Weak or no correlation with virality)</h3>
            <ul>
    """

    # Add no impact metrics (limit to top 10)
    for metric in virality_metrics['no_impact'][:10]:
        corr_value = virality_metrics['correlation_values'][metric]
        html += f"<li class='no-impact'>{metric}: {corr_value:.4f}</li>\n"

    html += """
            </ul>
        </div>

        <img src="../visualizations/correlation_heatmap.png" alt="Correlation Heatmap">

        <h2>7. Metric Definitions</h2>
        <p>Below are definitions for all metrics used in this analysis, including both original YouTube metrics and calculated metrics:</p>
    """

    # Load metric definitions
    try:
        metric_defs = pd.read_csv('metric_definitions.csv')

        # Add metric definitions table
        html += "<table>\n<tr><th>Metric</th><th>Definition</th></tr>\n"

        for _, row in metric_defs.iterrows():
            html += f"<tr><td><strong>{row['Metric']}</strong></td><td>{row['Definition']}</td></tr>\n"

        html += "</table>\n"
    except:
        html += "<p>Metric definitions file not found.</p>\n"

    html += """

        <h2>8. Conclusions and Recommendations</h2>
        <div class="summary-box">
            <h3>Key Takeaways</h3>
            <ul>
                <li>Engaged views is a better indicator of virality than raw views, as it represents viewers who actively interacted with the content.</li>
                <li>The metrics most strongly correlated with engaged views are likes, watch time, and subscribers gained.</li>
                <li>Short-form content generally outperforms long-form content in terms of engagement metrics across all accounts.</li>
                <li>Thumbsup Stories account shows a complete gap in long-form content during Jan-Mar, with very low performance for long-form videos in April.</li>
                <li>Videos with higher retention rates (stayed to watch %) and completion rates tend to generate more engaged views.</li>
                <li>The newly calculated Virality Score (weighted combination of engagement metrics) shows strong correlation with overall performance.</li>
                <li>Subscriber gain is more efficient in higher engaged view buckets, suggesting truly viral videos are more effective for channel growth.</li>
            </ul>

            <h3>Recommendations</h3>
            <ul>
                <li>Focus on optimizing for engaged views rather than raw views, as engagement is a stronger indicator of content quality and virality.</li>
                <li>Prioritize improving high-impact metrics like likes and watch time, which show the strongest correlation with engaged views.</li>
                <li>For Thumbsup Stories, the data strongly suggests reconsidering the investment in long-form content given its extremely low performance.</li>
                <li>Develop content strategies that maximize the Virality Score by balancing likes, comments, and engaged views.</li>
                <li>Optimize video retention by analyzing what keeps viewers watching in your most successful videos - this directly impacts watch time.</li>
                <li>Consider the optimal video duration for your audience - shorter videos tend to have better retention rates but long-form content can drive higher subscriber conversion when it performs well.</li>
                <li>Monitor and improve the Growth Potential metric (combination of subscriber gain and engagement) to maximize channel growth.</li>
                <li>Use the Engaged View Bucket categorization for more accurate performance benchmarking than raw view counts.</li>
            </ul>
        </div>

    </body>
    </html>
    """

    # Write HTML to file
    with open('report/youtube_analytics_report.html', 'w') as f:
        f.write(html)

    print("HTML report generated: report/youtube_analytics_report.html")

def main():
    print("Loading data...")
    df = pd.read_csv('youtube_analytics_master.csv')

    print("Generating comprehensive report...")
    generate_html_report(df)

    print("Report generation complete.")

if __name__ == "__main__":
    main()
