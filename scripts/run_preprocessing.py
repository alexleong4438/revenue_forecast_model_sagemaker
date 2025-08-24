#!/usr/bin/env python3
"""Script to run data preprocessing."""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.preprocessor import preprocess_data, create_rolling_splits
from src.config.settings import settings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main preprocessing function."""
    try:
        logger.info("Starting data preprocessing")
        
        # Check if raw data file exists
        raw_data_path = Path(settings.RAW_DATA_PATH)
        if not raw_data_path.exists():
            logger.error(f"Raw data file not found: {raw_data_path}")
            logger.info("Please ensure the raw data file exists at the specified path")
            return 1
        
        # Run preprocessing
        logger.info(f"Processing raw data from: {raw_data_path}")
        preprocess_data(str(raw_data_path), settings.PROCESSED_DATA_DIR)
        
        # Create rolling forecast splits
        logger.info("Creating rolling forecast splits")
        create_rolling_splits(settings.PROCESSED_DATA_DIR, settings.SPLIT_DATA_DIR)
        
        logger.info("Data preprocessing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
