# setup.sh

#!/bin/bash

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install the required dependencies
pip install -r requirements.txt

echo "Setup complete. To activate the virtual environment, use 'source venv/bin/activate'."
