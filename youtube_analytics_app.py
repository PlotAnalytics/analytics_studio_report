import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from pathlib import Path
import os

# Set page configuration
st.set_page_config(
    page_title="YouTube Analytics Reports",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #FF0000;
        padding-bottom: 0.5rem;
    }
    .report-container {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }
    .report-description {
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background-color: #FF0000;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #CC0000;
    }
    iframe {
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .view-button {
        text-align: center;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
        font-size: 0.8rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Function to display HTML content
def display_html_report(html_file):
    # Try different possible locations for the file
    possible_locations = [
        html_file,  # Original path
        os.path.join("vercel_deploy", html_file),  # Check in vercel_deploy
        os.path.basename(html_file)  # Just the filename in current directory
    ]

    # If the file has a directory, also try that directory in vercel_deploy
    if os.path.dirname(html_file):
        possible_locations.append(os.path.join("vercel_deploy", os.path.dirname(html_file), os.path.basename(html_file)))

    for location in possible_locations:
        if file_exists(location):
            try:
                with open(location, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Fix image paths in the HTML content
                html_content = fix_image_paths(html_content, location)

                # Display the HTML content directly using components.html
                st.components.v1.html(html_content, height=600, scrolling=True)
                return  # Successfully displayed the file, exit the function
            except Exception as e:
                st.error(f"Error reading file {location}: {str(e)}")

    # If we get here, none of the locations worked
    st.error(f"Error: Could not find the file {html_file} in any expected location.")
    st.info("Available HTML files can be found in the following directories: './report/', './statistical_analysis/', './vercel_deploy/', and the root directory.")

    # Show available HTML files
    st.expander("Available HTML Files", expanded=True).write(get_available_html_files())

# Function to check if a file exists
def file_exists(file_path):
    return os.path.isfile(file_path)

# Function to encode image file to base64
def get_image_base64(image_path):
    """Convert an image file to base64 encoding."""
    if not os.path.exists(image_path):
        # Try to find the image in alternative locations
        alt_paths = [
            image_path,
            os.path.join("vercel_deploy", image_path),
            os.path.join(".", image_path.lstrip("./"))
        ]

        for path in alt_paths:
            if os.path.exists(path):
                image_path = path
                break
        else:
            # If image still not found, return empty string
            return ""

    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

# Function to fix image paths in HTML content
def fix_image_paths(html_content, html_file_path):
    """
    Fix relative image paths in HTML content based on the HTML file's location.
    """
    # Get the directory of the HTML file
    html_dir = os.path.dirname(html_file_path)

    # Create a function to replace image paths with base64 encoded images
    def replace_with_base64(match):
        img_src = match.group(1)

        # Determine the full path based on the report type
        if "youtube_analytics_report.html" in html_file_path:
            if img_src.startswith("../visualizations/"):
                img_path = img_src.replace("../visualizations/", "visualizations/")
            else:
                img_path = img_src
        elif "statistical_report.html" in html_file_path:
            if img_src.startswith("figures/"):
                img_path = os.path.join("statistical_analysis", img_src)
            elif img_src.startswith("../statistical_analysis/figures/"):
                img_path = img_src.replace("../", "")
            else:
                img_path = img_src
        elif "cluster_analysis_deep_dive_report.html" in html_file_path:
            if img_src.startswith("cluster_analysis/figures/"):
                img_path = img_src
            else:
                img_path = img_src
        else:
            img_path = img_src

        # Get base64 encoding of the image
        img_base64 = get_image_base64(img_path)
        if img_base64:
            # Determine image type
            if img_path.lower().endswith('.png'):
                img_type = 'png'
            elif img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg'):
                img_type = 'jpeg'
            elif img_path.lower().endswith('.svg'):
                img_type = 'svg+xml'
            else:
                img_type = 'png'  # Default to PNG

            return f'src="data:image/{img_type};base64,{img_base64}"'
        else:
            # If image not found, return original src but log a warning
            st.warning(f"Image not found: {img_path}")
            return f'src="{img_src}" alt="Image not found"'

    # Replace all img src attributes with base64 encoded images
    import re
    html_content = re.sub(r'src="([^"]+)"', replace_with_base64, html_content)

    return html_content

# Function to get a list of available HTML files
def get_available_html_files():
    html_files = []

    # Check current directory
    for file in os.listdir('.'):
        if file.endswith('.html'):
            html_files.append(f"./[ROOT]: {file}")

    # Check report directory
    if os.path.exists('report'):
        for file in os.listdir('report'):
            if file.endswith('.html'):
                html_files.append(f"./report/: {file}")

    # Check statistical_analysis directory
    if os.path.exists('statistical_analysis'):
        for file in os.listdir('statistical_analysis'):
            if file.endswith('.html'):
                html_files.append(f"./statistical_analysis/: {file}")

    # Check vercel_deploy directory
    if os.path.exists('vercel_deploy'):
        for file in os.listdir('vercel_deploy'):
            if file.endswith('.html'):
                html_files.append(f"./vercel_deploy/: {file}")

        # Check vercel_deploy/report directory
        if os.path.exists('vercel_deploy/report'):
            for file in os.listdir('vercel_deploy/report'):
                if file.endswith('.html'):
                    html_files.append(f"./vercel_deploy/report/: {file}")

        # Check vercel_deploy/statistical_analysis directory
        if os.path.exists('vercel_deploy/statistical_analysis'):
            for file in os.listdir('vercel_deploy/statistical_analysis'):
                if file.endswith('.html'):
                    html_files.append(f"./vercel_deploy/statistical_analysis/: {file}")

    return "\n".join(html_files) if html_files else "No HTML files found."

# Function to create a download link for a file
def get_download_link(file_path, link_text):
    if not file_exists(file_path):
        st.warning(f"Warning: The file {file_path} does not exist. Download link will not work.")
        return f'<span style="color: #888; text-decoration: none;">{link_text} (File not found)</span>'

    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:file/html;base64,{b64}" download="{os.path.basename(file_path)}" style="color: #FF0000; text-decoration: none; font-weight: bold;">{link_text}</a>'
        return href
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return f'<span style="color: #888; text-decoration: none;">{link_text} (Error)</span>'

# Main app header
st.markdown('<h1 class="main-header">YouTube Analytics Impact Reports</h1>', unsafe_allow_html=True)

# Introduction
st.markdown("""
This dashboard provides comprehensive analytics for your YouTube channel performance.
Explore the main analytics report, statistical analysis, and cluster analysis to gain insights into your content performance.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Report", [
    "Overview",
    "Main YouTube Analytics Report",
    "Statistical Analysis Report",
    "Cluster Analysis Report"
])

# Overview page
if page == "Overview":
    st.markdown('<h2 class="sub-header">Overview of Available Reports</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/000000/youtube-play.png", width=80)
        st.markdown('<h3>Main YouTube Analytics Report</h3>', unsafe_allow_html=True)
        st.markdown('<p class="report-description">Comprehensive overview of your YouTube channel performance with key metrics, trends, and insights.</p>', unsafe_allow_html=True)
        if st.button("View Main Report", key="main_btn"):
            st.session_state.page = "Main YouTube Analytics Report"
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/000000/statistics.png", width=80)
        st.markdown('<h3>Statistical Analysis Report</h3>', unsafe_allow_html=True)
        st.markdown('<p class="report-description">In-depth statistical analysis of your YouTube data, including feature importance and predictive modeling.</p>', unsafe_allow_html=True)
        if st.button("View Statistical Report", key="stat_btn"):
            st.session_state.page = "Statistical Analysis Report"
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="report-container">', unsafe_allow_html=True)
        st.image("https://img.icons8.com/color/96/000000/clustering.png", width=80)
        st.markdown('<h3>Cluster Analysis Report</h3>', unsafe_allow_html=True)
        st.markdown('<p class="report-description">Discover content archetypes and strategic insights through advanced cluster analysis of your videos.</p>', unsafe_allow_html=True)
        if st.button("View Cluster Report", key="cluster_btn"):
            st.session_state.page = "Cluster Analysis Report"
            st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    ## How to Use These Reports

    1. **Main YouTube Analytics Report**: Start here for a high-level overview of your channel performance, including engaged view metrics, time period comparisons, and content type analysis.

    2. **Statistical Analysis Report**: Dive deeper into the data with advanced statistical analysis, including hypothesis testing and feature importance analysis using Random Forest.

    3. **Cluster Analysis Report**: Understand your content archetypes and identify strategic opportunities based on how your videos naturally group together.

    Each report can be viewed in the browser or downloaded as an HTML file for offline viewing and sharing.
    """)

# Main YouTube Analytics Report
elif page == "Main YouTube Analytics Report":
    st.markdown('<h2 class="sub-header">Main YouTube Analytics Report</h2>', unsafe_allow_html=True)
    st.markdown("""
    This comprehensive report provides a complete overview of your YouTube channel performance,
    including engaged view metrics, time period comparisons, and content type analysis.
    """)

    # Display download link
    st.markdown(get_download_link("report/youtube_analytics_report.html", "Download Full Report"), unsafe_allow_html=True)

    # Display the report
    st.markdown('<div class="view-button">Scroll to explore the full report</div>', unsafe_allow_html=True)
    display_html_report("report/youtube_analytics_report.html")

# Statistical Analysis Report
elif page == "Statistical Analysis Report":
    st.markdown('<h2 class="sub-header">Statistical Analysis Report</h2>', unsafe_allow_html=True)
    st.markdown("""
    This report provides in-depth statistical analysis of your YouTube data, including descriptive statistics,
    hypothesis testing, and advanced feature importance analysis using Random Forest.
    """)

    # Display download link
    st.markdown(get_download_link("statistical_analysis/statistical_report.html", "Download Full Report"), unsafe_allow_html=True)

    # Display the report
    st.markdown('<div class="view-button">Scroll to explore the full report</div>', unsafe_allow_html=True)
    display_html_report("statistical_analysis/statistical_report.html")

# Cluster Analysis Report
elif page == "Cluster Analysis Report":
    st.markdown('<h2 class="sub-header">Cluster Analysis Report</h2>', unsafe_allow_html=True)
    st.markdown("""
    This report provides insights into your content archetypes through advanced cluster analysis,
    helping you identify strategic opportunities and optimize your content mix.
    """)

    # Display download link
    st.markdown(get_download_link("cluster_analysis_deep_dive_report.html", "Download Full Report"), unsafe_allow_html=True)

    # Display the report
    st.markdown('<div class="view-button">Scroll to explore the full report</div>', unsafe_allow_html=True)
    display_html_report("cluster_analysis_deep_dive_report.html")

# Debug section (only shown when debug mode is enabled)
with st.expander("Debug Information", expanded=False):
    st.write("### File System Information")

    # Show current working directory
    st.write(f"**Current Working Directory:** {os.getcwd()}")

    # List HTML files in the current directory
    st.write("**HTML Files in Current Directory:**")
    html_files = [f for f in os.listdir('.') if f.endswith('.html')]
    if html_files:
        for file in html_files:
            st.write(f"- {file} ({'Exists' if file_exists(file) else 'Missing'})")
    else:
        st.write("No HTML files found in the current directory.")

    # Check specific directories
    directories_to_check = [
        'report',
        'statistical_analysis',
        '.'
    ]

    st.write("**Directory Structure:**")
    for directory in directories_to_check:
        if os.path.exists(directory):
            st.write(f"- {directory}/")
            files = os.listdir(directory)
            for file in files:
                if file.endswith('.html'):
                    full_path = os.path.join(directory, file)
                    st.write(f"  - {file} ({'Exists' if file_exists(full_path) else 'Missing'})")
        else:
            st.write(f"- {directory}/ (Directory not found)")

# Footer
st.markdown('<div class="footer">YouTube Analytics Impact Reports • Generated with Streamlit</div>', unsafe_allow_html=True)
