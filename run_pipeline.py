"""
Complete pipeline runner - trains models and starts API
"""
import subprocess
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required packages are installed"""
    logger.info("Checking dependencies...")
    try:
        import pandas
        import numpy
        import sklearn
        import statsmodels
        import prophet
        import xgboost
        import tensorflow
        import fastapi
        import uvicorn
        logger.info("✓ All dependencies installed")
        return True
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        logger.info("Please run: pip install -r requirements.txt")
        return False


def check_data_file():
    """Check if data file exists"""
    logger.info("Checking data file...")
    data_file = Path("Forecasting Case- Study.xlsx")
    if data_file.exists():
        logger.info(f"✓ Data file found: {data_file}")
        return True
    else:
        logger.error(f"✗ Data file not found: {data_file}")
        logger.info("Please ensure the Excel file is in the project directory")
        return False


def run_training():
    """Run the training pipeline"""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: TRAINING MODELS")
    logger.info("="*70)
    
    try:
        result = subprocess.run([sys.executable, "train.py"], check=True)
        logger.info("✓ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Training failed: {e}")
        return False


def run_visualization():
    """Generate visualizations"""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: GENERATING VISUALIZATIONS")
    logger.info("="*70)
    
    try:
        result = subprocess.run([sys.executable, "visualize_results.py"], check=True)
        logger.info("✓ Visualizations generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Visualization failed: {e}")
        return False


def start_api():
    """Start the API server"""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: STARTING API SERVER")
    logger.info("="*70)
    logger.info("API will be available at: http://localhost:8000")
    logger.info("API Documentation: http://localhost:8000/docs")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("="*70 + "\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        logger.info("\n✓ API server stopped")


def main():
    """Run complete pipeline"""
    print("\n" + "="*70)
    print("SALES FORECASTING SYSTEM - COMPLETE PIPELINE")
    print("="*70 + "\n")
    
    # Check prerequisites
    if not check_dependencies():
        sys.exit(1)
    
    if not check_data_file():
        sys.exit(1)
    
    # Run training
    if not run_training():
        logger.error("Training failed. Exiting...")
        sys.exit(1)
    
    # Generate visualizations
    run_visualization()
    
    # Start API
    logger.info("\nStarting API server in 3 seconds...")
    time.sleep(3)
    start_api()


if __name__ == "__main__":
    main()
