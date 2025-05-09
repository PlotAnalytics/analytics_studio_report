#!/usr/bin/env python3
"""
YouTube Analytics Deep Dive Analysis
-----------------------------------
This script runs a comprehensive analysis on YouTube analytics data,
focusing on metrics that determine virality across different view count buckets.

It performs the following:
1. Data cleaning and preparation
2. Calculation of derived metrics
3. Analysis by view count buckets
4. Comparison of short vs. long videos
5. Special analysis of Thumbsup Stories account
6. Determination of metrics that correlate with virality
7. Generation of visualizations
8. Creation of a comprehensive HTML report

Usage:
    python run_analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import analysis modules
from youtube_analytics_analysis import main as run_analysis
from youtube_analytics_visualizations import main as run_visualizations
from youtube_analytics_report import main as run_report

def main():
    print("=" * 80)
    print("YouTube Analytics Deep Dive Analysis")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if the CSV file exists
    if not os.path.exists('youtube_analytics_master.csv'):
        print("Error: youtube_analytics_master.csv not found!")
        print("Please ensure the CSV file is in the current directory.")
        return
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('report', exist_ok=True)
    
    # Step 1: Run the main analysis
    print("Step 1: Running main analysis...")
    run_analysis()
    print("Main analysis complete.")
    print()
    
    # Step 2: Generate visualizations
    print("Step 2: Generating visualizations...")
    run_visualizations()
    print("Visualizations complete.")
    print()
    
    # Step 3: Generate the comprehensive report
    print("Step 3: Generating comprehensive report...")
    run_report()
    print("Report generation complete.")
    print()
    
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("Results are available in the following directories:")
    print("- CSV results: ./results/")
    print("- Visualizations: ./visualizations/")
    print("- HTML Report: ./report/youtube_analytics_report.html")
    print()
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
