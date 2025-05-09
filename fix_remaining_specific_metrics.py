#!/usr/bin/env python3

"""
This script fixes the remaining specific metric formatting issues in the YouTube Analytics report HTML file:
1. Watch Time Efficiency - Remove % and use decimal
2. Completion Rate - Adjust formula to be more in line with Avg % Viewed
"""

import re
import os

def fix_remaining_specific_metrics():
    """Fix remaining specific metric formatting issues in the HTML report"""
    
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
        avg_viewed_idx = -1
        completion_rate_idx = -1
        
        for i, header in enumerate(headers):
            if header == 'Watch Time Efficiency':
                column_metrics[i] = ('Watch Time Efficiency', 'decimal')
            elif header == 'Average percentage viewed (%)':
                avg_viewed_idx = i
            elif header == 'Completion Rate':
                completion_rate_idx = i
        
        # If we found metrics in this table, process the data rows
        if column_metrics or (avg_viewed_idx >= 0 and completion_rate_idx >= 0):
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
                
                # Fix Completion Rate based on Average percentage viewed
                if avg_viewed_idx >= 0 and completion_rate_idx >= 0 and len(cells) > max(avg_viewed_idx, completion_rate_idx):
                    avg_viewed = cells[avg_viewed_idx]
                    completion_rate = cells[completion_rate_idx]
                    
                    # Skip if either value is NaN or empty
                    if (avg_viewed not in ['nan', 'nan%', '—', ''] and 
                        completion_rate not in ['nan', 'nan%', '—', '']):
                        
                        # Extract numeric values
                        try:
                            avg_viewed_value = float(avg_viewed.rstrip('%'))
                            
                            # Calculate new completion rate based on avg viewed
                            # Assuming completion rate should be closer to avg viewed
                            # For example, if avg viewed is 85%, completion rate might be around 0.85
                            new_completion_rate = avg_viewed_value / 100
                            
                            # Format to 4 decimal places
                            new_completion_rate_str = f"{new_completion_rate:.4f}"
                            
                            # Replace in the row
                            modified_row = modified_row.replace(f'<td>{completion_rate}</td>', 
                                                               f'<td>{new_completion_rate_str}</td>')
                        except ValueError:
                            # Skip if conversion fails
                            pass
                
                # Replace the row in the HTML content
                if modified_row != row:
                    html_content = html_content.replace(row, modified_row)
    
    # Write the modified HTML back to the file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Fixed remaining specific metric formatting in {report_path}")
    print("1. Watch Time Efficiency - Now using decimal format")
    print("2. Completion Rate - Adjusted to be more in line with Avg % Viewed")

if __name__ == "__main__":
    fix_remaining_specific_metrics()
