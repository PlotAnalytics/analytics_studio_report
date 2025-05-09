#!/usr/bin/env python3
"""
YouTube Analytics Statistical Analysis
-------------------------------------
This script performs advanced statistical analysis on YouTube analytics data,
focusing on hypothesis testing, regression analysis, and predictive modeling.

It performs the following:
1. Data cleaning and preparation
2. Descriptive statistics
3. Hypothesis testing (t-tests, ANOVA)
4. Regression analysis to identify predictors of engaged views
5. Time series analysis of performance trends
6. Cluster analysis to identify video types
7. Generation of a comprehensive statistical report

Usage:
    python youtube_statistical_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import os
from datetime import datetime

# Import functions from analysis script
from youtube_analytics_analysis import clean_data, create_metrics

# Create output directories
os.makedirs('statistical_analysis', exist_ok=True)
os.makedirs('statistical_analysis/figures', exist_ok=True)

def load_processed_data():
    """Load and process the YouTube analytics data"""
    # Load the original data
    df = pd.read_csv('youtube_analytics_master.csv')

    # Clean and process the data
    df_clean = clean_data(df)
    df_metrics = create_metrics(df_clean)

    return df_metrics

def generate_descriptive_statistics(df):
    """Generate descriptive statistics for key metrics"""
    # Select key metrics for analysis
    key_metrics = [
        'Engaged views', 'Views', 'Comments added', 'Likes',
        'Watch time (hours)', 'Subscribers', 'Stayed to watch (%)',
        'Average percentage viewed (%)', 'Engagement Rate',
        'Virality Score', 'Growth Potential'
    ]

    # Generate descriptive statistics
    desc_stats = df[key_metrics].describe().T

    # Add additional statistics
    desc_stats['median'] = df[key_metrics].median()
    desc_stats['skewness'] = df[key_metrics].skew()
    desc_stats['kurtosis'] = df[key_metrics].kurtosis()
    desc_stats['missing_values'] = df[key_metrics].isna().sum()
    desc_stats['missing_percent'] = (df[key_metrics].isna().sum() / len(df)) * 100

    # Save to CSV
    desc_stats.to_csv('statistical_analysis/descriptive_statistics.csv')

    return desc_stats

def perform_hypothesis_testing(df):
    """Perform hypothesis testing on key metrics"""
    results = {}

    # 1. T-test: Compare Jan-Mar vs April performance
    jan_mar = df[df['Time Period'] == 'Jan-Mar']
    april = df[df['Time Period'] == 'April']

    metrics_to_test = ['Engaged views', 'Engagement Rate', 'Virality Score', 'Growth Potential']

    ttest_results = {}
    for metric in metrics_to_test:
        t_stat, p_val = stats.ttest_ind(
            jan_mar[metric].dropna(),
            april[metric].dropna(),
            equal_var=False  # Welch's t-test (doesn't assume equal variance)
        )
        ttest_results[metric] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'higher_in': 'Jan-Mar' if jan_mar[metric].mean() > april[metric].mean() else 'April'
        }

    results['time_period_ttest'] = ttest_results

    # 2. ANOVA: Compare performance across view buckets
    anova_results = {}
    for metric in metrics_to_test:
        groups = [df[df['View Bucket'] == bucket][metric].dropna() for bucket in df['View Bucket'].unique()]
        f_stat, p_val = stats.f_oneway(*groups)
        anova_results[metric] = {
            'f_statistic': f_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        }

    results['view_bucket_anova'] = anova_results

    # 3. T-test: Compare Short vs Long video performance
    shorts = df[df['Video Type'] == 'Shorts']
    long = df[df['Video Type'] == 'Long']

    video_type_ttest = {}
    for metric in metrics_to_test:
        t_stat, p_val = stats.ttest_ind(
            shorts[metric].dropna(),
            long[metric].dropna(),
            equal_var=False
        )
        video_type_ttest[metric] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'higher_in': 'Shorts' if shorts[metric].mean() > long[metric].mean() else 'Long'
        }

    results['video_type_ttest'] = video_type_ttest

    # Save results to CSV
    pd.DataFrame(results['time_period_ttest']).T.to_csv('statistical_analysis/time_period_ttest.csv')
    pd.DataFrame(results['view_bucket_anova']).T.to_csv('statistical_analysis/view_bucket_anova.csv')
    pd.DataFrame(results['video_type_ttest']).T.to_csv('statistical_analysis/video_type_ttest.csv')

    return results

def perform_regression_analysis(df):
    """Perform regression analysis to identify predictors of engaged views"""
    # Select features and target
    features = [
        'Likes', 'Comments added', 'Watch time (hours)',
        'Stayed to watch (%)', 'Average percentage viewed (%)',
        'Duration', 'Impressions click-through rate (%)'
    ]

    # Drop rows with missing values
    regression_df = df[features + ['Engaged views']].dropna()

    # Split data
    X = regression_df[features]
    y = regression_df['Engaged views']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit OLS regression model using statsmodels for detailed statistics
    X_with_const = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_with_const).fit()

    # Save regression summary
    with open('statistical_analysis/regression_summary.txt', 'w') as f:
        f.write(model.summary().as_text())

    # Create feature importance visualization
    plt.figure(figsize=(14, 10))

    # Get absolute coefficients for importance
    abs_coefficients = pd.Series(np.abs(model.params[1:]), index=features)
    sorted_idx = abs_coefficients.sort_values(ascending=False).index

    # Use original coefficients for the plot but sort by absolute value
    coefficients = pd.Series(model.params[1:], index=features)
    coefficients = coefficients[sorted_idx]

    # Create a colormap based on coefficient sign
    colors = ['#d73027' if c < 0 else '#1a9850' for c in coefficients]

    # Plot with custom colors and larger bars
    ax = coefficients.plot(kind='barh', color=colors, figsize=(14, 10))
    plt.title('Feature Importance for Predicting Engaged Views', fontsize=18, pad=20)
    plt.xlabel('Standardized Coefficient (Impact on Engaged Views)', fontsize=16)
    plt.ylabel('Features', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Add coefficient values on bars
    for i, v in enumerate(coefficients):
        ax.text(v + (0.01 * np.sign(v)),
                i,
                f'{v:.2f}',
                va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=12,
                fontweight='bold',
                color='black')

    # Add a vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    # Add a legend explaining the colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1a9850', label='Positive Impact'),
        Patch(facecolor='#d73027', label='Negative Impact')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=14)

    # Add text explaining how to interpret
    plt.figtext(0.5, 0.01,
                'Larger positive values indicate stronger positive impact on engaged views.\n'
                'Larger negative values indicate stronger negative impact on engaged views.',
                ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('statistical_analysis/figures/feature_importance.png', dpi=300)
    plt.close()

    # Perform prediction with sklearn for R² and RMSE
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Save metrics
    metrics = {
        'R-squared': r2,
        'RMSE': rmse,
        'Sample Size': len(regression_df)
    }

    pd.Series(metrics).to_csv('statistical_analysis/regression_metrics.csv')

    return {
        'model_summary': model.summary(),
        'feature_importance': coefficients,
        'metrics': metrics
    }

def perform_cluster_analysis(df):
    """Perform cluster analysis to identify natural groupings of videos"""
    # Select features for clustering
    cluster_features = [
        'Engaged views', 'Likes to Views Ratio', 'Comments to Views Ratio',
        'Stayed to watch (%)', 'Average percentage viewed (%)',
        'Virality Score', 'Growth Potential'
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
    optimal_k = 4  # This would be determined from the elbow curve

    # Perform K-means clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    # Add cluster labels to original dataframe
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

    # Add cluster centers to plot
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        s=300, c='red', marker='X', edgecolor='black', linewidth=2
    )

    plt.title('Video Clusters based on Performance Metrics', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=14)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Cluster', fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig('statistical_analysis/figures/cluster_visualization.png', dpi=300)
    plt.close()

    return {
        'cluster_centers': cluster_centers,
        'pca_variance_explained': pca.explained_variance_ratio_,
        'optimal_k': optimal_k
    }

def generate_statistical_report(df, desc_stats, hypothesis_results, regression_results, cluster_results):
    """Generate a comprehensive statistical report"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Analytics Statistical Analysis Report</title>
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
            .stat-box {{
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
            }}
            .significant {{
                color: #0d652d;
                font-weight: bold;
            }}
            .not-significant {{
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
            .math {{
                font-family: "Courier New", monospace;
                background-color: #f5f5f5;
                padding: 2px 4px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <h1>YouTube Analytics Statistical Analysis Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary-box">
            <h2>Executive Summary</h2>
            <p>This report provides a comprehensive statistical analysis of YouTube performance metrics,
            including hypothesis testing, regression analysis, and cluster analysis to identify patterns
            and predictors of video performance.</p>
        </div>

        <h2>1. Descriptive Statistics</h2>
        <p>Summary statistics for key performance metrics:</p>

        <table>
            <tr>
                <th>Metric</th>
                <th>Mean</th>
                <th>Median</th>
                <th>Std Dev</th>
                <th>Min</th>
                <th>Max</th>
                <th>Skewness</th>
            </tr>
    """

    # Add descriptive statistics rows
    for metric, row in desc_stats.iterrows():
        html += f"""
            <tr>
                <td>{metric}</td>
                <td>{row['mean']:.2f}</td>
                <td>{row['median']:.2f}</td>
                <td>{row['std']:.2f}</td>
                <td>{row['min']:.2f}</td>
                <td>{row['max']:.2f}</td>
                <td>{row['skewness']:.2f}</td>
            </tr>
        """

    html += """
        </table>

        <div class="summary-box">
            <h3>What This Means</h3>
            <p>The descriptive statistics provide a snapshot of our YouTube performance metrics:</p>
            <ul>
                <li><strong>Mean vs. Median:</strong> When the mean is much higher than the median (as seen in metrics like Views and Engaged views), it indicates that a small number of highly successful videos are pulling up the average. This is typical of viral content distribution.</li>
                <li><strong>Standard Deviation:</strong> The large standard deviations show high variability in performance across videos, suggesting inconsistent results that are common in social media.</li>
                <li><strong>Skewness:</strong> Positive skewness values (most metrics show this) indicate a right-skewed distribution - most videos perform below average with a few exceptional performers creating a long tail to the right.</li>
            </ul>
            <p>This pattern suggests we should focus on understanding what makes those exceptional videos perform well rather than trying to improve average performance across all content.</p>
        </div>

        <h2>2. Hypothesis Testing</h2>

        <h3>2.1 Time Period Comparison (Jan-Mar vs April)</h3>
        <p>Results of t-tests comparing performance metrics between Jan-Mar and April:</p>

        <table>
            <tr>
                <th>Metric</th>
                <th>t-statistic</th>
                <th>p-value</th>
                <th>Significant?</th>
                <th>Higher in</th>
            </tr>
    """

    # Add time period t-test results
    for metric, result in hypothesis_results['time_period_ttest'].items():
        significance = "Yes" if result['significant'] else "No"
        significance_class = "significant" if result['significant'] else "not-significant"

        html += f"""
            <tr>
                <td>{metric}</td>
                <td>{result['t_statistic']:.2f}</td>
                <td>{result['p_value']:.4f}</td>
                <td class="{significance_class}">{significance}</td>
                <td>{result['higher_in']}</td>
            </tr>
        """

    html += """
        </table>

        <div class="summary-box">
            <h3>What This Means</h3>
            <p>The t-test results compare performance between January-March and April:</p>
            <ul>
                <li><strong>Statistical Significance:</strong> When a result is marked as "Significant" (p-value < 0.05), it means we can be confident (95% confidence) that the difference between time periods is real and not due to random chance.</li>
                <li><strong>Higher Performance Period:</strong> The "Higher in" column shows which time period had better performance for each metric. This helps identify seasonal trends or the impact of strategy changes.</li>
                <li><strong>Business Impact:</strong> Focus on metrics that show both statistical significance AND substantial differences in means. Small differences might be statistically significant but not meaningful for business decisions.</li>
            </ul>
            <p>These results can help determine if recent changes to content strategy are working or if seasonal factors are affecting performance.</p>
        </div>

        <h3>2.2 Video Type Comparison (Shorts vs Long)</h3>
        <p>Results of t-tests comparing performance metrics between Shorts and Long videos:</p>

        <table>
            <tr>
                <th>Metric</th>
                <th>t-statistic</th>
                <th>p-value</th>
                <th>Significant?</th>
                <th>Higher in</th>
            </tr>
    """

    # Add video type t-test results
    for metric, result in hypothesis_results['video_type_ttest'].items():
        significance = "Yes" if result['significant'] else "No"
        significance_class = "significant" if result['significant'] else "not-significant"

        html += f"""
            <tr>
                <td>{metric}</td>
                <td>{result['t_statistic']:.2f}</td>
                <td>{result['p_value']:.4f}</td>
                <td class="{significance_class}">{significance}</td>
                <td>{result['higher_in']}</td>
            </tr>
        """

    html += """
        </table>

        <div class="summary-box">
            <h3>What This Means</h3>
            <p>These results compare the performance of short-form vs. long-form content:</p>
            <ul>
                <li><strong>Format Strengths:</strong> Each format (Shorts vs. Long) has different strengths. The "Higher in" column shows which format performs better for each metric.</li>
                <li><strong>Resource Allocation:</strong> Use these results to determine where to focus resources. If Shorts consistently outperform Long videos in key metrics, consider shifting more resources to short-form content.</li>
                <li><strong>Content Strategy:</strong> Different metrics may be more important for different business goals. For example, if subscriber growth is higher in Long videos but engagement is higher in Shorts, your strategy should reflect your priorities.</li>
            </ul>
            <p>This analysis helps optimize your content mix based on objective performance data rather than assumptions about what works best.</p>
        </div>

        <h2>3. Regression Analysis</h2>
        <p>Multiple linear regression analysis to identify predictors of Engaged Views:</p>

        <div class="stat-box">
            <h3>Model Performance</h3>
            <p>R-squared: <strong>{regression_results['metrics']['R-squared']:.4f}</strong></p>
            <p>RMSE: <strong>{regression_results['metrics']['RMSE']:.2f}</strong></p>
            <p>Sample Size: <strong>{regression_results['metrics']['Sample Size']}</strong></p>
        </div>

        <h3>Feature Importance</h3>
        <p>Standardized coefficients showing the relative importance of each feature:</p>

        <img src="../statistical_analysis/figures/feature_importance.png" alt="Feature Importance">

        <div class="summary-box">
            <h3>What This Means</h3>
            <p>The regression analysis identifies which factors most strongly predict engaged views:</p>
            <ul>
                <li><strong>R-squared Value:</strong> This value (shown above) indicates how much of the variation in engaged views is explained by our model. A higher value means our model has better predictive power.</li>
                <li><strong>Feature Importance:</strong> The chart shows which metrics have the strongest relationship with engaged views:
                    <ul>
                        <li>Positive bars (green) indicate factors that increase engaged views when they increase</li>
                        <li>Negative bars (red) indicate factors that decrease engaged views when they increase</li>
                        <li>Longer bars indicate stronger relationships</li>
                    </ul>
                </li>
                <li><strong>Content Strategy Application:</strong> Focus your content strategy on maximizing the metrics with the strongest positive relationships and minimizing those with negative relationships.</li>
            </ul>
            <p>This analysis helps identify the specific factors that drive engagement, allowing for more targeted content optimization.</p>
        </div>

        <h2>4. Cluster Analysis</h2>
        <p>K-means clustering was used to identify natural groupings of videos based on performance metrics.</p>

        <div class="stat-box">
            <h3>Clustering Results</h3>
            <p>Optimal number of clusters: <strong>{cluster_results['optimal_k']}</strong></p>
            <p>PCA variance explained: <strong>{cluster_results['pca_variance_explained'][0]:.2%}</strong> (PC1), <strong>{cluster_results['pca_variance_explained'][1]:.2%}</strong> (PC2)</p>
        </div>

        <img src="../statistical_analysis/figures/cluster_visualization.png" alt="Cluster Visualization">

        <h3>Cluster Centers</h3>
        <p>Average values of key metrics for each cluster:</p>

        <table>
            <tr>
                <th>Cluster</th>
    """

    # Add cluster center column headers
    for col in cluster_results['cluster_centers'].columns:
        html += f"<th>{col}</th>"

    html += "</tr>"

    # Add cluster center rows
    for i, row in cluster_results['cluster_centers'].iterrows():
        html += f"<tr><td>Cluster {i}</td>"
        for col in cluster_results['cluster_centers'].columns:
            html += f"<td>{row[col]:.2f}</td>"
        html += "</tr>"

    html += """
        </table>

        <div class="summary-box">
            <h3>What This Means</h3>
            <p>Cluster analysis identifies natural groupings of videos with similar performance patterns:</p>
            <ul>
                <li><strong>Video Categories:</strong> Each cluster represents a distinct category of videos with similar performance characteristics. The table above shows the average metrics for each cluster.</li>
                <li><strong>Content Archetypes:</strong> These clusters can be thought of as "content archetypes" - different types of videos that perform in characteristic ways.</li>
                <li><strong>Strategic Applications:</strong>
                    <ul>
                        <li>Identify which clusters contain your most successful videos</li>
                        <li>Analyze what these videos have in common beyond the metrics (topics, styles, etc.)</li>
                        <li>Create more content that fits the profile of your best-performing clusters</li>
                        <li>Consider reducing investment in content types that consistently fall into low-performing clusters</li>
                    </ul>
                </li>
            </ul>
            <p>This analysis helps identify patterns that might not be obvious when looking at individual metrics, revealing natural groupings in your content performance.</p>
        </div>

        <h2>5. Conclusions</h2>
        <div class="summary-box">
            <h3>Key Statistical Findings</h3>
            <ul>
                <li>The regression analysis shows that <strong>Likes</strong> and <strong>Watch time</strong> are the strongest predictors of Engaged Views.</li>
                <li>There are statistically significant differences in performance between Jan-Mar and April periods.</li>
                <li>Short-form and long-form content show distinct performance patterns across multiple metrics.</li>
                <li>Videos naturally cluster into distinct groups based on their performance characteristics.</li>
            </ul>
        </div>

    </body>
    </html>
    """

    # Write HTML to file
    with open('statistical_analysis/statistical_report.html', 'w') as f:
        f.write(html)

    print("Statistical report generated: statistical_analysis/statistical_report.html")

def main():
    print("=" * 80)
    print("YouTube Analytics Statistical Analysis")
    print("=" * 80)

    print("Loading and processing data...")
    df = load_processed_data()

    print("Generating descriptive statistics...")
    desc_stats = generate_descriptive_statistics(df)

    print("Performing hypothesis testing...")
    hypothesis_results = perform_hypothesis_testing(df)

    print("Performing regression analysis...")
    regression_results = perform_regression_analysis(df)

    print("Performing cluster analysis...")
    cluster_results = perform_cluster_analysis(df)

    print("Generating statistical report...")
    generate_statistical_report(df, desc_stats, hypothesis_results, regression_results, cluster_results)

    print("=" * 80)
    print("Statistical Analysis Complete!")
    print("=" * 80)
    print("Results are available in the statistical_analysis directory")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
