#!/usr/bin/env python3
"""
YouTube Analytics Complete Analysis Suite
----------------------------------------
This script runs both the standard analytics analysis and the advanced statistical analysis
on YouTube analytics data.

It performs the following:
1. Standard analytics analysis with visualizations
2. Advanced statistical analysis with hypothesis testing and modeling
3. Generation of both reports

Usage:
    python run_all_analyses.py
"""

import os
import subprocess
import time
from datetime import datetime

def main():
    print("=" * 80)
    print("YouTube Analytics Complete Analysis Suite")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if the CSV file exists
    if not os.path.exists('youtube_analytics_master.csv'):
        print("Error: youtube_analytics_master.csv not found!")
        print("Please ensure the CSV file is in the current directory.")
        return
    
    # Step 1: Run the main analytics analysis
    print("Step 1: Running main analytics analysis...")
    start_time = time.time()
    subprocess.run(['python3', 'run_analysis.py'], check=True)
    end_time = time.time()
    print(f"Main analytics analysis completed in {end_time - start_time:.2f} seconds.")
    print()
    
    # Step 2: Run the statistical analysis
    print("Step 2: Running advanced statistical analysis...")
    start_time = time.time()
    subprocess.run(['python3', 'youtube_statistical_analysis.py'], check=True)
    end_time = time.time()
    print(f"Statistical analysis completed in {end_time - start_time:.2f} seconds.")
    print()
    
    print("=" * 80)
    print("All Analyses Complete!")
    print("=" * 80)
    print("Results are available in the following locations:")
    print("- Main Analytics Report: ./report/youtube_analytics_report.html")
    print("- Statistical Analysis Report: ./statistical_analysis/statistical_report.html")
    print("- Visualizations: ./visualizations/")
    print("- Statistical Figures: ./statistical_analysis/figures/")
    print()
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
