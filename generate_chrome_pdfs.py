#!/usr/bin/env python3
"""
Generate High-Definition PDF Reports using Chrome
------------------------------------------------
This script uses Chrome's headless mode to convert HTML reports to high-quality PDF files.

Usage:
    python3 generate_chrome_pdfs.py
"""

import os
import subprocess
import sys
import time

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def convert_html_to_pdf_with_chrome(html_path, pdf_path):
    """Convert HTML file to PDF using Chrome's headless mode."""
    print(f"Converting {html_path} to PDF...")

    # Get absolute paths
    current_dir = os.getcwd()
    html_abs_path = os.path.join(current_dir, html_path)
    pdf_abs_path = os.path.join(current_dir, pdf_path)

    # Chrome command for high-quality PDF generation
    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_abs_path}",
        f"file://{html_abs_path}",
        "--no-margins",
        "--print-to-pdf-no-header",
    ]

    try:
        # Run Chrome in headless mode
        process = subprocess.run(
            chrome_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"Successfully created PDF: {pdf_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {html_path} to PDF: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    """Main function to generate PDF reports."""
    # Create output directory
    output_dir = "pdf_reports"
    create_directory_if_not_exists(output_dir)

    # Define reports to convert
    reports = [
        {
            "html_path": "report/youtube_analytics_report.html",
            "pdf_path": f"{output_dir}/YouTube_Analytics_Impact_Report.pdf",
            "title": "YouTube Analytics Impact Report"
        },
        {
            "html_path": "statistical_analysis/statistical_report.html",
            "pdf_path": f"{output_dir}/YouTube_Analytics_Statistical_Report.pdf",
            "title": "YouTube Analytics Statistical Report"
        },
        {
            "html_path": "cluster_analysis_deep_dive_report.html",
            "pdf_path": f"{output_dir}/YouTube_Analytics_Cluster_Analysis_Report.pdf",
            "title": "YouTube Analytics Cluster Analysis Deep Dive Report"
        }
    ]

    # Convert each report
    success_count = 0
    for report in reports:
        if convert_html_to_pdf_with_chrome(report["html_path"], report["pdf_path"]):
            success_count += 1

    # Print summary
    print(f"\nSummary: Successfully converted {success_count} of {len(reports)} reports to PDF.")
    print(f"PDF files are available in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
