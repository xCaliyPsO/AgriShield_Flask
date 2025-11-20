#!/usr/bin/env python3
"""
WSGI entry point for production deployment
Use with Gunicorn, uWSGI, or mod_wsgi
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import app from app.py (now has all features from pest_detection_api.py)
from app import app

# This is the WSGI application object
application = app

if __name__ == "__main__":
    # For testing WSGI locally (same as running app.py directly)
    print("Testing WSGI entry point...")
    print("For production, use: gunicorn -c gunicorn_config.py wsgi:application")
    app.run(host="0.0.0.0", port=5001, debug=False)

