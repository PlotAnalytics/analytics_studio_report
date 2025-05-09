# YouTube Analytics Reports

This repository contains HTML reports for YouTube analytics, including:

1. **Main Impact Report** - Overview of YouTube analytics with key findings and recommendations
2. **Statistical Analysis Report** - Detailed statistical analysis including regression analysis and feature importance
3. **Cluster Analysis Report** - Deep dive into content archetypes identified through cluster analysis

## Deploying to Vercel

### Option 1: Deploy with Vercel CLI

1. Install Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Navigate to the project directory and run:
   ```
   vercel
   ```

3. Follow the prompts to deploy your project.

### Option 2: Deploy via GitHub Integration

1. Push this repository to GitHub.
2. Go to [Vercel](https://vercel.com) and sign in.
3. Click "New Project" and import your GitHub repository.
4. Configure the project settings (the defaults should work fine).
5. Click "Deploy" to deploy your project.

### Option 3: Deploy via Vercel Web Interface

1. Compress this directory into a ZIP file.
2. Go to [Vercel](https://vercel.com) and sign in.
3. Click "New Project" and then "Upload" to upload your ZIP file.
4. Configure the project settings (the defaults should work fine).
5. Click "Deploy" to deploy your project.

## Project Structure

```
/
├── index.html                  # Main landing page with links to all reports
├── 404.html                    # Custom 404 page
├── vercel.json                 # Vercel configuration
├── report/
│   └── youtube_analytics_report.html  # Main impact report
├── statistical_analysis/
│   └── statistical_report.html        # Statistical analysis report
└── cluster_analysis_deep_dive_report.html  # Cluster analysis report
```

## Notes

- All reports focus on "Engaged views" rather than raw "Views" as the primary metric
- The reports are static HTML files and do not require a server to run
- All visualizations are included as static images
