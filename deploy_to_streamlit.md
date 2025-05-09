# Deploying to Streamlit Cloud

Follow these steps to deploy your YouTube Analytics Reports to Streamlit Cloud:

## 1. Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in or create an account
2. Click the "+" icon in the top right and select "New repository"
3. Name your repository (e.g., "youtube-analytics-dashboard")
4. Make it public or private as needed
5. Click "Create repository"

## 2. Upload Your Files

### Option A: Using GitHub Web Interface

1. In your new repository, click "uploading an existing file"
2. Drag and drop all files from this folder or use the file selector
3. Commit the changes

### Option B: Using Git Command Line

1. Initialize a git repository in this folder:
   ```
   git init
   ```

2. Add all files:
   ```
   git add .
   ```

3. Commit the files:
   ```
   git commit -m "Initial commit with YouTube analytics reports"
   ```

4. Add your GitHub repository as remote:
   ```
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
   ```

5. Push the files:
   ```
   git push -u origin main
   ```

## 3. Deploy to Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account
2. Click "New app"
3. Select your repository, branch (main), and the main file path (youtube_analytics_app.py)
4. Click "Deploy"
5. Wait for the deployment to complete (this may take a few minutes)

## 4. Share with Your Boss

1. Once deployed, you'll get a URL like: https://username-repo-name-streamlit-app-randomstring.streamlit.app/
2. Share this URL with your boss and team members
3. They can access the dashboard from any device with a web browser
4. No login required to view the reports

## Notes

- Streamlit Cloud's free tier should be sufficient for this application
- The app will automatically update if you push changes to your GitHub repository
- You can set up a custom subdomain in the Streamlit Cloud settings if needed
- For private repositories, you'll need to authorize Streamlit Cloud to access your private repos
