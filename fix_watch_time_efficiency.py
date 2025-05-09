#!/usr/bin/env python3

"""
This script fixes the Watch Time Efficiency values in the video type comparison table.
"""

import re
import os

def fix_watch_time_efficiency():
    """Fix the Watch Time Efficiency values in the video type comparison table"""
    
    # Path to the HTML report
    report_path = 'report/youtube_analytics_report.html'
    
    # Check if the report exists
    if not os.path.exists(report_path):
        print(f"Error: Report file not found at {report_path}")
        return
    
    # Read the HTML content
    with open(report_path, 'r') as f:
        html_content = f.read()
    
    # Find the video type comparison table
    table_match = re.search(r'<table>.*?<tr><th>Video Type</th>.*?</table>', html_content, re.DOTALL)
    if not table_match:
        print("Could not find the video type comparison table")
        return
    
    table = table_match.group(0)
    
    # Find the header row
    header_match = re.search(r'<tr><th>.*?</th></tr>', table, re.DOTALL)
    if not header_match:
        print("Could not find the header row in the video type comparison table")
        return
    
    header_row = header_match.group(0)
    headers = re.findall(r'<th>(.*?)</th>', header_row)
    
    # Find the Watch Time Efficiency column index
    watch_time_efficiency_idx = -1
    for i, header in enumerate(headers):
        if header == 'Watch Time Efficiency':
            watch_time_efficiency_idx = i
            break
    
    if watch_time_efficiency_idx < 0:
        print("Could not find the Watch Time Efficiency column in the video type comparison table")
        return
    
    # Find all data rows in the table
    data_rows = re.findall(r'<tr><td>.*?</td></tr>', table, re.DOTALL)
    
    # Process each data row
    for row in data_rows:
        cells = re.findall(r'<td>(.*?)</td>', row)
        modified_row = row
        
        # Process the Watch Time Efficiency cell
        if watch_time_efficiency_idx < len(cells):
            cell_value = cells[watch_time_efficiency_idx]
            
            # Skip NaN values
            if cell_value in ['nan', 'nan%', '—', '']:
                continue
            
            # Convert to a more reasonable value
            try:
                value = float(cell_value)
                
                # If the value is too large (> 1), scale it down
                if value > 1:
                    new_value = f"{value/10:.4f}"
                    modified_row = modified_row.replace(f'<td>{cell_value}</td>', 
                                                       f'<td>{new_value}</td>')
            except ValueError:
                # Skip if conversion fails
                pass
        
        # Replace the row in the HTML content
        if modified_row != row:
            html_content = html_content.replace(row, modified_row)
    
    # Write the modified HTML back to the file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"Fixed Watch Time Efficiency values in the video type comparison table")

if __name__ == "__main__":
    fix_watch_time_efficiency()
