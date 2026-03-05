#!/usr/bin/env python3
"""
Main entry point to run Layer 4 API
"""

import os
import sys
import uvicorn
import logging

try:
    from dotenv import load_dotenv
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# Load env variables from root .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("layer4.runner")

def run_layer4(host="127.0.0.1", port=8000, reload=False):
    """Start Layer 4 FastAPI application using uvicorn"""
    logger.info(f"Starting Layer 4 API on {host}:{port}...")
    
    # Check dependencies before starting
    try:
        import fastapi
        import uvicorn
        import pydantic
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Please install missing dependencies: pip install fastapi uvicorn pydantic")
        return
        
    try:
        import layer4.main
    except ModuleNotFoundError as e:
        logger.error(f"Could not load layer4 module: {e}")
        logger.info("Ensure you are running this script from the stochclaim directory.")
        return

    # Start the server
    uvicorn.run(
        "layer4.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Layer 4 API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind socket to this host")
    parser.add_argument("--port", type=int, default=8000, help="Bind socket to this port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    
    run_layer4(host=args.host, port=args.port, reload=args.reload)
