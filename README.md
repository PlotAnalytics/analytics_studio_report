# YouTube Analytics Reports

This repository contains HTML reports for YouTube analytics, including:

1. **Main Impact Report** - Overview of YouTube analytics with key findings and recommendations
2. **Statistical Analysis Report** - Detailed statistical analysis including advanced feature importance analysis
3. **Cluster Analysis Report** - Deep dive into content archetypes identified through cluster analysis

## Deployment Options

### Option 1: Deploy with Streamlit (Interactive Dashboard)

This repository now includes a Streamlit app that provides an interactive dashboard for all three reports.

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run youtube_analytics_app.py
   ```

3. Open the provided URL in your browser to view the interactive dashboard.

#### Deploy to Streamlit Cloud (Recommended for Sharing)

1. Create a free account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub account
3. Create a new repository and upload all files from this folder
4. Deploy the app on Streamlit Cloud by selecting the repository and the `youtube_analytics_app.py` file
5. Share the provided URL with your team

### Option 2: Deploy to Vercel (Static HTML)

#### Deploy with Vercel CLI

1. Install Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Navigate to the project directory and run:
   ```
   vercel
   ```

3. Follow the prompts to deploy your project.

#### Deploy via GitHub Integration

1. Push this repository to GitHub.
2. Go to [Vercel](https://vercel.com) and sign in.
3. Click "New Project" and import your GitHub repository.
4. Configure the project settings (the defaults should work fine).
5. Click "Deploy" to deploy your project.

#### Deploy via Vercel Web Interface

1. Compress this directory into a ZIP file.
2. Go to [Vercel](https://vercel.com) and sign in.
3. Click "New Project" and then "Upload" to upload your ZIP file.
4. Configure the project settings (the defaults should work fine).
5. Click "Deploy" to deploy your project.

## Project Structure

```
/
├── youtube_analytics_app.py    # Streamlit app for interactive dashboard
├── requirements.txt            # Python dependencies for Streamlit
├── index.html                  # Main landing page with links to all reports (for Vercel)
├── 404.html                    # Custom 404 page (for Vercel)
├── vercel.json                 # Vercel configuration
├── youtube_analytics_report.html  # Main impact report
├── statistical_analysis/
│   ├── statistical_report.html        # Statistical analysis report
│   └── figures/                       # Statistical visualizations
├── cluster_analysis_deep_dive_report.html  # Cluster analysis report
└── README.md                   # This documentation file
```

## Notes

- All reports focus on "Engaged views" rather than raw "Views" as the primary metric
- The HTML reports are static files and do not require a server to run
- All visualizations are included as static images
- The Streamlit app provides an interactive way to view and navigate between reports
- The statistical analysis uses advanced Random Forest methods for more accurate feature importance
- All reports are based on actual YouTube analytics data from your channel
