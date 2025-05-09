#!/usr/bin/env python3

"""
This script converts Completion Rate from decimal format (0.9115) to percentage format (91.15%)
to make it more intuitive for users while maintaining the "Rate" terminology in the name.
"""

import re
import os

def fix_completion_rate_display():
    """Convert Completion Rate from decimal to percentage format"""
    
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
        
        # Find the Completion Rate column index
        completion_rate_idx = -1
        for i, header in enumerate(headers):
            if header == 'Completion Rate':
                completion_rate_idx = i
                break
        
        # If we found the Completion Rate column, process the data rows
        if completion_rate_idx >= 0:
            # Find all data rows in this table
            data_rows = re.findall(r'<tr><td>.*?</td></tr>', table, re.DOTALL)
            
            # Process each data row
            for row in data_rows:
                cells = re.findall(r'<td>(.*?)</td>', row)
                modified_row = row
                
                # Process the Completion Rate cell
                if completion_rate_idx < len(cells):
                    cell_value = cells[completion_rate_idx]
                    
                    # Skip NaN values or already formatted percentages
                    if cell_value in ['nan', 'nan%', '—', ''] or cell_value.endswith('%'):
                        continue
                    
                    # Convert decimal to percentage
                    try:
                        value = float(cell_value)
                        # Multiply by 100 and add % symbol
                        new_value = f"{value * 100:.2f}%"
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
    
    print(f"Fixed Completion Rate display in {report_path}")
    print("Completion Rate now displays as percentage (e.g., 91.15%) instead of decimal (e.g., 0.9115)")

if __name__ == "__main__":
    fix_completion_rate_display()
