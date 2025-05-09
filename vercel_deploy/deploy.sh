#!/bin/bash

# Deploy to Vercel script
echo "Deploying YouTube Analytics Reports to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null
then
    echo "Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Deploy to Vercel
echo "Running Vercel deployment..."
vercel --prod

echo "Deployment complete! Your YouTube Analytics Reports are now live."
