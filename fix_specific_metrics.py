#!/usr/bin/env python3

"""
This script fixes specific metric formatting issues in the YouTube Analytics report HTML file:
1. Engaged Views per Impression - Remove %, use decimal
2. Completion Rate - Recheck formula
3. Virality Score - Express as a score/index, not %
4. Growth Potential - Use % only if it's a comparative metric
5. NaN values - Replace or handle for downstream processing
"""

import re
import os

def fix_specific_metrics():
    """Fix specific metric formatting issues in the HTML report"""
    
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
            if header == 'Engaged Views per Impression':
                column_metrics[i] = ('Engaged Views per Impression', 'decimal')
            elif header == 'Completion Rate':
                column_metrics[i] = ('Completion Rate', 'decimal')
            elif header == 'Virality Score':
                column_metrics[i] = ('Virality Score', 'score')
            elif header == 'Growth Potential':
                column_metrics[i] = ('Growth Potential', 'decimal')
        
        # If we found metrics in this table, process the data rows
        if column_metrics:
            # Find all data rows in this table
            data_rows = re.findall(r'<tr><td>.*?</td></tr>', table, re.DOTALL)
            
            # Process each data row
            for row in data_rows:
                cells = re.findall(r'<td>(.*?)</td>', row)
                modified_row = row
                
                # Process each cell that corresponds to a metric
                for col_idx, (metric_name, format_type) in column_metrics.items():
                    if col_idx < len(cells):
                        cell_value = cells[col_idx]
                        
                        # Handle NaN values
                        if cell_value == 'nan' or cell_value == 'nan%':
                            new_value = '—'  # Em dash as a placeholder for NaN
                            modified_row = modified_row.replace(f'<td>{cell_value}</td>', 
                                                               f'<td>{new_value}</td>')
                            continue
                        
                        # Skip empty cells
                        if not cell_value:
                            continue
                        
                        # Process based on format type
                        if format_type == 'decimal':
                            # Remove % if present and ensure it's a decimal
                            if cell_value.endswith('%'):
                                # Convert percentage to decimal
                                try:
                                    value = float(cell_value.rstrip('%')) / 100
                                    new_value = f"{value:.4f}"
                                    modified_row = modified_row.replace(f'<td>{cell_value}</td>', 
                                                                       f'<td>{new_value}</td>')
                                except ValueError:
                                    # If conversion fails, just remove the %
                                    new_value = cell_value.rstrip('%')
                                    modified_row = modified_row.replace(f'<td>{cell_value}</td>', 
                                                                       f'<td>{new_value}</td>')
                        
                        elif format_type == 'score':
                            # Express as a score/index, not %
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
    
    print(f"Fixed specific metric formatting in {report_path}")
    print("1. Engaged Views per Impression - Now using decimal format")
    print("2. Completion Rate - Now using decimal format")
    print("3. Virality Score - Now expressed as a score/index, not %")
    print("4. Growth Potential - Now using decimal format")
    print("5. NaN values - Replaced with em dashes for better display")

if __name__ == "__main__":
    fix_specific_metrics()
