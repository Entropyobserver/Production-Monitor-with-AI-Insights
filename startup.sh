#!/bin/bash
# File: startup.sh
# Azure App Service startup script with secrets file creation

# Create .streamlit directory if it doesn't exist
mkdir -p /tmp/.streamlit
mkdir -p .streamlit

# Create secrets.toml file from environment variables if GOOGLE_API_KEY is set
if [ ! -z "$GOOGLE_API_KEY" ]; then
    cat > .streamlit/secrets.toml << EOF
GOOGLE_API_KEY = "$GOOGLE_API_KEY"
EOF
    echo "Created secrets.toml with API key from environment"
else
    # Create empty secrets file to prevent NotADirectoryError
    cat > .streamlit/secrets.toml << EOF
# No API key configured - AI features will be disabled
EOF
    echo "Created empty secrets.toml - AI features disabled"
fi

# Start Streamlit
python -m streamlit run app.py --server.port ${PORT:-8000} --server.address 0.0.0.0 --server.headless true
