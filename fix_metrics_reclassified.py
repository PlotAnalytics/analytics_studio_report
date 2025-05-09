#!/usr/bin/env python3

"""
This script fixes the metric formatting in the YouTube Analytics report HTML file
according to the reclassified metrics list.
"""

import re
import os

def fix_metrics_formatting():
    """Fix the metric formatting in the HTML report according to the reclassified metrics list"""
    
    # Path to the HTML report
    report_path = 'report/youtube_analytics_report.html'
    
    # Check if the report exists
    if not os.path.exists(report_path):
        print(f"Error: Report file not found at {report_path}")
        return
    
    # Read the HTML content
    with open(report_path, 'r') as f:
        html_content = f.read()
    
    # Define which metrics should be percentages and which should not
    percentage_metrics = [
        'Stayed to watch (%)',
        'Average percentage viewed (%)',
        'Comments to Engaged Views Ratio',
        'Likes to Engaged Views Ratio',
        'Comments to Likes Ratio',
        'Engagement Rate',
        'Swipe Away Ratio',
        'Completion Rate',
        'Virality Score',
        'Watch Time Efficiency',
        'Retention Efficiency',
        'Impressions click-through rate (%)'
    ]
    
    non_percentage_metrics = [
        'Subscribers Gained per 1000 Engaged Views',
        'Engaged Views per Impression',
        'Watch Time per Engaged View (seconds)',
        'Growth Potential'
    ]
    
    # Find all tables in the HTML
    tables = re.findall(r'<table>.*?</table>', html_content, re.DOTALL)
    
    for table in tables:
        # Find the header row
        header_match = re.search(r'<tr><th>.*?</th></tr>', table, re.DOTALL)
        if not header_match:
            continue
            
        header_row = header_match.group(0)
        headers = re.findall(r'<th>(.*?)</th>', header_row)
        
        # Map column indices to metrics
        column_metrics = {}
        for i, header in enumerate(headers):
            if header in percentage_metrics:
                column_metrics[i] = (header, True)  # Should have %
            elif header in non_percentage_metrics:
                column_metrics[i] = (header, False)  # Should NOT have %
        
        # If we found metrics in this table, process the data rows
        if column_metrics:
            # Find all data rows in this table
            data_rows = re.findall(r'<tr><td>.*?</td></tr>', table, re.DOTALL)
            
            # Process each data row
            for row in data_rows:
                cells = re.findall(r'<td>(.*?)</td>', row)
                modified_row = row
                
                # Process each cell that corresponds to a metric
                for col_idx, (metric_name, should_have_percent) in column_metrics.items():
                    if col_idx < len(cells):
                        cell_value = cells[col_idx]
                        
                        # Skip cells with 'nan' values
                        if cell_value == 'nan' or cell_value == 'nan%':
                            continue
                            
                        if should_have_percent:
                            # Should be a percentage
                            if not cell_value.endswith('%'):
                                new_value = f"{cell_value}%"
                                modified_row = modified_row.replace(f'<td>{cell_value}</td>', 
                                                                   f'<td>{new_value}</td>')
                        else:
                            # Should not be a percentage
                            if cell_value.endswith('%'):
                                new_value = cell_value.rstrip('%')
                                modified_row = modified_row.replace(f'<td>{cell_value}</td>', 
                                                                   f'<td>{new_value}</td>')
                
                # Replace the row in the HTML content
                if modified_row != row:
                    html_content = html_content.replace(row, modified_row)
    
    # Write the modified HTML back to the file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Fixed metric formatting in {report_path}")
    print("All metrics now have correct percentage formatting according to the reclassified list.")

if __name__ == "__main__":
    fix_metrics_formatting()
