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
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Create a temporary HTML file with the content
    with open("temp_report.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    
    # Display the HTML content in an iframe
    st.components.v1.iframe("temp_report.html", height=600, scrolling=True)

# Function to create a download link for a file
def get_download_link(file_path, link_text):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/html;base64,{b64}" download="{os.path.basename(file_path)}" style="color: #FF0000; text-decoration: none; font-weight: bold;">{link_text}</a>'
    return href

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
    st.markdown(get_download_link("youtube_analytics_report.html", "Download Full Report"), unsafe_allow_html=True)
    
    # Display the report
    st.markdown('<div class="view-button">Scroll to explore the full report</div>', unsafe_allow_html=True)
    display_html_report("youtube_analytics_report.html")

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

# Footer
st.markdown('<div class="footer">YouTube Analytics Impact Reports • Generated with Streamlit</div>', unsafe_allow_html=True)
