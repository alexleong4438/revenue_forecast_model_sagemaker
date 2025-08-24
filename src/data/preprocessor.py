"""Data preprocessing and splitting utilities."""

import pandas as pd
import json
import os
import logging
from typing import Tuple, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data preprocessing for revenue forecasting."""
    
    def __init__(self, file_path: str, output_dir: str = "data/processed"):
        """
        Initialize the preprocessor.
        
        Args:
            file_path: Path to the raw CSV file
            output_dir: Directory to save processed files
        """
        self.file_path = file_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and validate CSV
        self.df = self._load_and_validate_data()
        
        # Process the data
        self._process_data()
    
    def _load_and_validate_data(self) -> pd.DataFrame:
        """Load CSV and validate required columns."""
        try:
            df = pd.read_csv(self.file_path)
            required_columns = ['partner_id', 'period', 'report_type', 'value']
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Expected: {required_columns}")
            
            logger.info(f"Loaded {len(df)} rows from {self.file_path}")
            return df[required_columns]
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _process_data(self) -> None:
        """Process the loaded data."""
        # Rename 'value' column to 'revenue'
        self.df = self.df.rename(columns={'value': 'revenue'})
        
        # Parse 'period' column
        period_data = self.df['period'].apply(self._parse_period)
        self.df[['period_type', 'start_date']] = pd.DataFrame(
            period_data.tolist(), index=self.df.index
        )
        self.df.drop(columns=['period'], inplace=True)
        
        # Convert start_date to datetime format
        self.df['start_date'] = pd.to_datetime(self.df['start_date'], errors='coerce')
        
        # Remove rows with invalid dates
        invalid_dates = self.df['start_date'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Removing {invalid_dates} rows with invalid dates")
            self.df = self.df.dropna(subset=['start_date'])
        
        logger.info(f"Processed data shape: {self.df.shape}")
    
    def _parse_period(self, period_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse period string to extract type and start_date.
        
        Args:
            period_str: Period string in JSON format
            
        Returns:
            Tuple of (period_type, start_date)
        """
        try:
            # Convert single quotes to double quotes for JSON parsing
            period_dict = json.loads(period_str.replace("'", "\""))
            period_type = period_dict.get('type', {}).get('S', None)
            start_date = period_dict.get('start_date', {}).get('S', None)
            return period_type, start_date
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Error parsing period '{period_str}': {e}")
            return None, None
    
    def split_and_save(self) -> None:
        """Split data by partner_id and period_type, then save separate CSV files."""
        saved_files = []
        
        for (partner_id, period_type), group in self.df.groupby(["partner_id", "period_type"]):
            if group.empty:
                continue
                
            filename = self.output_dir / f"{partner_id}_{period_type}.csv"
            group.to_csv(filename, index=False)
            saved_files.append(filename)
            logger.info(f"Saved: {filename} ({len(group)} rows)")
        
        logger.info(f"Split data into {len(saved_files)} files")
        return saved_files


class RollingForecastSplitter:
    """Creates rolling forecast splits for time series data."""
    
    def __init__(self, input_dir: str = "data/processed", output_dir: str = "data/split_data"):
        """
        Initialize the splitter.
        
        Args:
            input_dir: Directory containing processed CSV files
            output_dir: Directory to save split files
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_files(self, min_data_points: int = 8) -> None:
        """
        Iterate over all processed files and apply Rolling Forecast Split.
        
        Args:
            min_data_points: Minimum number of data points required for splitting
        """
        csv_files = list(self.input_dir.glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.input_dir}")
            return
        
        logger.info(f"Processing {len(csv_files)} files for rolling forecast split")
        
        for file_path in csv_files:
            self._process_single_file(file_path, min_data_points)
    
    def _process_single_file(self, file_path: Path, min_data_points: int) -> None:
        """
        Process a single file for rolling forecast split.
        
        Args:
            file_path: Path to the CSV file
            min_data_points: Minimum number of data points required
        """
        try:
            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Ensure datetime format for sorting
            df["start_date"] = pd.to_datetime(df["start_date"], errors='coerce')
            
            # Sort data by date (important for time-series)
            df = df.sort_values(by="start_date").reset_index(drop=True)
            
            # Check if we have enough data points
            if len(df) < min_data_points:
                logger.warning(f"Not enough data points in {file_path.name} ({len(df)} < {min_data_points}). Skipping.")
                return
            
            # Define splits
            train_df = df.iloc[:-8]  # Train set: All except the last 8 timestamps
            initial_test_df = df.iloc[-8:]  # Initial test set: Last 8 timestamps
            final_test_df = initial_test_df.iloc[:-4]  # Final test set: Remove last 4 timestamps
            
            # Save results
            base_name = file_path.stem
            train_filename = self.output_dir / f"{base_name}_train.csv"
            initial_test_filename = self.output_dir / f"{base_name}_initial_test.csv"
            final_test_filename = self.output_dir / f"{base_name}_final_test.csv"
            
            train_df.to_csv(train_filename, index=False)
            initial_test_df.to_csv(initial_test_filename, index=False)
            final_test_df.to_csv(final_test_filename, index=False)
            
            logger.info(f"✅ Rolling Forecast Split completed for {file_path.name}")
            logger.info(f"   🔹 Train: {train_filename} ({len(train_df)} rows)")
            logger.info(f"   🔹 Initial Test: {initial_test_filename} ({len(initial_test_df)} rows)")
            logger.info(f"   🔹 Final Test: {final_test_filename} ({len(final_test_df)} rows)")
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")


def preprocess_data(file_path: str, output_dir: str = "data/processed") -> None:
    """
    Main function to preprocess data.
    
    Args:
        file_path: Path to the raw CSV file
        output_dir: Directory to save processed files
    """
    preprocessor = DataPreprocessor(file_path, output_dir)
    preprocessor.split_and_save()


def create_rolling_splits(input_dir: str = "data/processed", output_dir: str = "data/split_data") -> None:
    """
    Main function to create rolling forecast splits.
    
    Args:
        input_dir: Directory containing processed CSV files
        output_dir: Directory to save split files
    """
    splitter = RollingForecastSplitter(input_dir, output_dir)
    splitter.process_files()
