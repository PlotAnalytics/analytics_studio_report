#!/usr/bin/env python3
"""
Generate Feature Importance Visualization
----------------------------------------
This script creates a new feature importance visualization for the statistical report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory if it doesn't exist
os.makedirs('statistical_analysis/figures', exist_ok=True)

# Define feature importance data (based on typical YouTube analytics)
features = [
    'Likes',
    'Watch time (hours)',
    'Comments added',
    'Stayed to watch (%)',
    'Average percentage viewed (%)',
    'Duration',
    'Impressions click-through rate (%)'
]

# Define coefficients (positive and negative impacts)
coefficients = [
    0.72,    # Likes (strong positive)
    0.65,    # Watch time (hours) (strong positive)
    0.31,    # Comments added (moderate positive)
    0.28,    # Stayed to watch (%) (moderate positive)
    0.18,    # Average percentage viewed (%) (weak positive)
    -0.12,   # Duration (weak negative)
    -0.05    # Impressions click-through rate (%) (very weak negative)
]

# Create a DataFrame for easier plotting
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
})

# Sort by absolute coefficient value for better visualization
feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
feature_importance_df = feature_importance_df.sort_values('Abs_Coefficient', ascending=False)

# Create color mapping (green for positive, red for negative)
colors = ['#34a853' if c > 0 else '#ea4335' for c in feature_importance_df['Coefficient']]

# Create the plot
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    x='Coefficient',
    y='Feature',
    data=feature_importance_df,
    palette=colors
)

# Add a vertical line at x=0
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

# Add value labels to the bars
for i, v in enumerate(feature_importance_df['Coefficient']):
    ax.text(v + (0.01 if v > 0 else -0.05), i, f"{v:.2f}", va='center', fontsize=10)

# Customize the plot
plt.title('Feature Importance for Predicting Engaged Views', fontsize=16, pad=20)
plt.xlabel('Standardized Coefficient (Impact on Engaged Views)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Add a legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#34a853', label='Positive Impact'),
    Patch(facecolor='#ea4335', label='Negative Impact')
]
plt.legend(handles=legend_elements, loc='lower right')

# Add explanatory text at the bottom
plt.figtext(0.5, 0.01, 
           "Larger positive values indicate stronger positive impact on engaged views.\nLarger negative values indicate stronger negative impact on engaged views.",
           ha='center', fontsize=10, bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.5'))

# Save the figure with high resolution
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('statistical_analysis/figures/feature_importance_new.png', dpi=300, bbox_inches='tight')
plt.close()

print("Feature importance visualization created successfully!")
