"""Basic tests for the revenue forecast model."""

import unittest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings


class TestSettings(unittest.TestCase):
    """Test configuration settings."""
    
    def test_settings_exist(self):
        """Test that settings can be imported and have expected attributes."""
        self.assertIsNotNone(settings.AWS_REGION)
        self.assertIsNotNone(settings.S3_BUCKET)
        self.assertIsNotNone(settings.PREDICTION_LENGTH)
        self.assertIsInstance(settings.FORECAST_QUANTILES, list)
        self.assertIsInstance(settings.AUTOML_ALGORITHMS, list)
    
    def test_s3_paths(self):
        """Test S3 path generation."""
        self.assertTrue(settings.s3_data_path.startswith("s3://"))
        self.assertTrue(settings.s3_output_path.startswith("s3://"))
        self.assertIn(settings.S3_BUCKET, settings.s3_data_path)
        self.assertIn(settings.S3_BUCKET, settings.s3_output_path)


class TestDataProcessing(unittest.TestCase):
    """Test data processing functionality."""
    
    def test_import_preprocessor(self):
        """Test that preprocessor module can be imported."""
        try:
            from src.data.preprocessor import DataPreprocessor, RollingForecastSplitter
            self.assertTrue(True)  # If no exception, test passes
        except ImportError as e:
            self.fail(f"Failed to import preprocessor: {e}")


class TestPrediction(unittest.TestCase):
    """Test prediction functionality."""
    
    def test_import_predictor(self):
        """Test that predictor module can be imported."""
        try:
            from src.prediction.predictor import RevenuePredictor
            self.assertTrue(True)  # If no exception, test passes
        except ImportError as e:
            self.fail(f"Failed to import predictor: {e}")


if __name__ == '__main__':
    unittest.main()
