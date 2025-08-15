#!/bin/bash
# File: startup.sh
# Enhanced Azure App Service startup script with comprehensive error handling

set -e  # Exit on any error

echo "Starting Production Monitor deployment process..."
echo "Current working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Check if we're in Azure App Service
if [ -n "$WEBSITE_INSTANCE_ID" ]; then
    echo "Running in Azure App Service"
    echo "Website name: $WEBSITE_SITE_NAME"
    echo "Instance ID: $WEBSITE_INSTANCE_ID"
else
    echo "Running locally"
fi

# Create necessary directories
echo "Creating .streamlit directory..."
mkdir -p .streamlit
mkdir -p /tmp/.streamlit

# Check if requirements are installed
echo "Checking Python dependencies..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Streamlit not found. Installing dependencies..."
    
    # Check if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo "Installing from requirements.txt..."
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
        echo "Dependencies installed successfully"
    else
        echo "ERROR: requirements.txt not found"
        echo "Installing minimal dependencies..."
        python -m pip install --upgrade pip
        python -m pip install streamlit pandas numpy plotly
    fi
else
    echo "Dependencies already installed"
fi

# Verify critical dependencies
echo "Verifying dependencies..."
python -c "
try:
    import streamlit
    import pandas
    import numpy
    import plotly
    print('✓ Core dependencies verified')
except ImportError as e:
    print(f'✗ Dependency error: {e}')
    exit(1)
"

# Create secrets.toml from environment variables
echo "Configuring secrets..."
if [ ! -z "$GOOGLE_API_KEY" ]; then
    cat > .streamlit/secrets.toml << EOF
GOOGLE_API_KEY = "$GOOGLE_API_KEY"
EOF
    echo "✓ API key configured from environment"
else
    cat > .streamlit/secrets.toml << EOF
# No API key configured - AI features will be disabled
# To enable AI features, set GOOGLE_API_KEY environment variable
EOF
    echo "⚠ No API key found - AI features disabled"
fi

# Create Streamlit config
echo "Creating Streamlit configuration..."
cat > .streamlit/config.toml << EOF
[server]
headless = true
port = 8000
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 50

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1E40AF"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#1F2937"

[logger]
level = "info"
EOF

# Set environment variables for Streamlit
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=${PORT:-8000}
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "ERROR: app.py not found in $(pwd)"
    echo "Files in current directory:"
    ls -la
    exit 1
fi

echo "Starting Streamlit application..."
echo "Command: python -m streamlit run app.py --server.port=${PORT:-8000} --server.address=0.0.0.0 --server.headless=true"

# Start the application with error handling
exec python -m streamlit run app.py \
    --server.port=${PORT:-8000} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --logger.level=info
