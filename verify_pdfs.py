#!/usr/bin/env python3
"""
Verify PDF Reports
-----------------
This script verifies that the PDF reports were created successfully and provides information about them.
"""

import os
import datetime

def get_file_info(file_path):
    """Get file information including size and creation time."""
    if not os.path.exists(file_path):
        return None

    file_stats = os.stat(file_path)
    size_bytes = file_stats.st_size
    size_mb = size_bytes / (1024 * 1024)

    # Get creation time
    creation_time = datetime.datetime.fromtimestamp(file_stats.st_ctime)

    return {
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "creation_time": creation_time
    }

def main():
    """Main function to verify PDF reports."""
    pdf_dir = "pdf_reports"

    if not os.path.exists(pdf_dir):
        print(f"Error: Directory '{pdf_dir}' does not exist.")
        return

    # Define expected PDF files
    expected_pdfs = [
        {
            "file_name": "YouTube_Analytics_Impact_Report.pdf",
            "description": "Main impact report with key findings and recommendations"
        },
        {
            "file_name": "YouTube_Analytics_Statistical_Report.pdf",
            "description": "Statistical analysis report with regression analysis and feature importance"
        },
        {
            "file_name": "YouTube_Analytics_Cluster_Analysis_Report.pdf",
            "description": "Cluster analysis deep dive report with visualizations and video lists"
        }
    ]

    print("\n" + "=" * 80)
    print("PDF REPORTS VERIFICATION SUMMARY")
    print("=" * 80)

    all_exist = True

    for pdf in expected_pdfs:
        file_path = os.path.join(pdf_dir, pdf["file_name"])
        file_info = get_file_info(file_path)

        if file_info:
            print(f"\n✅ {pdf['file_name']}")
            print(f"   Description: {pdf['description']}")
            print(f"   Size: {file_info['size_mb']:.2f} MB ({file_info['size_bytes']:,} bytes)")
            print(f"   Created: {file_info['creation_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Path: {os.path.abspath(file_path)}")
        else:
            print(f"\n❌ {pdf['file_name']} - NOT FOUND")
            all_exist = False

    print("\n" + "-" * 80)
    if all_exist:
        print("✅ All PDF reports were generated successfully!")
        print("\nYou can now send these high-definition PDF reports to your boss.")
        print("The PDFs include all visualizations, tables, and scrollable content from the HTML reports.")
    else:
        print("❌ Some PDF reports are missing. Please check the errors above.")

    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
