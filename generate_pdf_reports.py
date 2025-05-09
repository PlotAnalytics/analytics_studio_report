#!/usr/bin/env python3
"""
Generate High-Definition PDF Reports
-----------------------------------
This script converts HTML reports to high-quality PDF files.

Requirements:
- weasyprint
- cssselect2
- tinycss2
- cairocffi

Usage:
    python3 generate_pdf_reports.py
"""

import os
from weasyprint import HTML, CSS
import sys
import time

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def convert_html_to_pdf(html_path, pdf_path, title):
    """Convert HTML file to PDF with high quality settings."""
    print(f"Converting {html_path} to PDF...")
    
    # Create custom CSS for better PDF rendering
    custom_css = CSS(string='''
        @page {
            size: letter;
            margin: 1cm;
            @top-center {
                content: string(title);
                font-family: Arial, sans-serif;
                font-size: 9pt;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-family: Arial, sans-serif;
                font-size: 9pt;
                color: #666;
            }
        }
        
        h1 {
            string-set: title content();
        }
        
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            page-break-inside: avoid;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
        }
        
        img {
            max-width: 100%;
            height: auto;
        }
        
        .chart-container {
            page-break-inside: avoid;
        }
        
        .video-list {
            max-height: none !important;
            overflow-y: visible !important;
        }
    ''')
    
    try:
        # Convert HTML to PDF
        HTML(filename=html_path).write_pdf(
            pdf_path,
            stylesheets=[custom_css],
            presentational_hints=True,
            optimize_size=('fonts', 'images'),
            zoom=1.0  # Adjust zoom factor for higher quality
        )
        print(f"Successfully created PDF: {pdf_path}")
        return True
    except Exception as e:
        print(f"Error converting {html_path} to PDF: {e}")
        return False

def main():
    """Main function to generate PDF reports."""
    # Create output directory
    output_dir = "pdf_reports"
    create_directory_if_not_exists(output_dir)
    
    # Define reports to convert
    reports = [
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
        if convert_html_to_pdf(report["html_path"], report["pdf_path"], report["title"]):
            success_count += 1
    
    # Print summary
    print(f"\nSummary: Successfully converted {success_count} of {len(reports)} reports to PDF.")
    print(f"PDF files are available in the '{output_dir}' directory.")

if __name__ == "__main__":
    main()
