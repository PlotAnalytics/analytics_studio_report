#!/usr/bin/env python3

"""
This script fixes the remaining metric formatting issues in the YouTube Analytics report HTML file.
"""

import re
import os

def fix_final_metrics():
    """Fix the remaining metric formatting issues in the HTML report"""
    
    # Path to the HTML report
    report_path = 'report/youtube_analytics_report.html'
    
    # Check if the report exists
    if not os.path.exists(report_path):
        print(f"Error: Report file not found at {report_path}")
        return
    
    # Read the HTML content
    with open(report_path, 'r') as f:
        html_content = f.read()
    
    # Find all tables in the HTML
    tables = re.findall(r'<table>.*?</table>', html_content, re.DOTALL)
    
    for table in tables:
        # Find the header row
        header_match = re.search(r'<tr><th>.*?</th></tr>', table, re.DOTALL)
        if not header_match:
            continue
            
        header_row = header_match.group(0)
        headers = re.findall(r'<th>(.*?)</th>', header_row)
        
        # Map column indices to metrics that need fixing
        column_metrics = {}
        for i, header in enumerate(headers):
            if header == 'Engaged Views per Impression' or header == 'Growth Potential':
                column_metrics[i] = header
        
        # If we found metrics in this table, process the data rows
        if column_metrics:
            # Find all data rows in this table
            data_rows = re.findall(r'<tr><td>.*?</td></tr>', table, re.DOTALL)
            
            # Process each data row
            for row in data_rows:
                cells = re.findall(r'<td>(.*?)</td>', row)
                modified_row = row
                
                # Process each cell that corresponds to a metric
                for col_idx, metric_name in column_metrics.items():
                    if col_idx < len(cells):
                        cell_value = cells[col_idx]
                        
                        # Skip cells with 'nan' values
                        if cell_value == 'nan' or cell_value == 'nan%':
                            continue
                        
                        # Add % to the cell value if it doesn't already have it
                        if not cell_value.endswith('%'):
                            new_value = f"{cell_value}%"
                            modified_row = modified_row.replace(f'<td>{cell_value}</td>', 
                                                               f'<td>{new_value}</td>')
                
                # Replace the row in the HTML content
                if modified_row != row:
                    html_content = html_content.replace(row, modified_row)
    
    # Write the modified HTML back to the file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Fixed final metric formatting in {report_path}")
    print("All metrics now have correct percentage formatting according to the reclassified list.")

if __name__ == "__main__":
    fix_final_metrics()
