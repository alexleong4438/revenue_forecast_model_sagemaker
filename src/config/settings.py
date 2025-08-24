"""Configuration settings for the revenue forecast model."""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # AWS Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "eu-west-1")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "prediction-test-bucket-01")
    S3_PREFIX: str = os.getenv("S3_PREFIX", "autopilot-demo-notebook")
    
    # SageMaker Configuration
    SAGEMAKER_ENDPOINT_NAME: str = os.getenv("SAGEMAKER_ENDPOINT_NAME", "canvas-prediction-test-model")
    SAGEMAKER_EXECUTION_ROLE: str = os.getenv("SAGEMAKER_EXECUTION_ROLE", "")
    
    # Model Configuration
    PREDICTION_LENGTH: int = int(os.getenv("PREDICTION_LENGTH", "4"))
    FORECAST_FREQUENCY: str = os.getenv("FORECAST_FREQUENCY", "M")
    MAX_AUTOML_JOB_RUNTIME: int = int(os.getenv("MAX_AUTOML_JOB_RUNTIME", "1800"))  # 30 minutes
    
    # Data Configuration
    RAW_DATA_PATH: str = os.getenv("RAW_DATA_PATH", "data/raw/raw_data.csv")
    PROCESSED_DATA_DIR: str = os.getenv("PROCESSED_DATA_DIR", "data/processed")
    SPLIT_DATA_DIR: str = os.getenv("SPLIT_DATA_DIR", "data/split_data")
    
    # Model Parameters
    FORECAST_QUANTILES: List[str] = ["p10", "p50", "p90"]
    AUTOML_ALGORITHMS: List[str] = [
        "cnn-qr",
        "deepar", 
        "prophet",
        "npts",
        "arima",
        "ets"
    ]
    
    # Instance quota mapping
    EP_INSTANCE_QUOTA_CODE_MAP: Dict[str, str] = {
        "ml.m5.4xlarge": "L-E2649D46",
        "ml.m5.xlarge": "L-2F737F8D",
    }
    
    @property
    def s3_data_path(self) -> str:
        """S3 path for data storage."""
        return f"s3://{self.S3_BUCKET}/data"
    
    @property
    def s3_output_path(self) -> str:
        """S3 path for output artifacts."""
        return f"s3://{self.S3_BUCKET}/artifact"
    
    def get_automl_input_config(self, data_filename: str) -> List[Dict[str, Any]]:
        """Get AutoML input data configuration."""
        return [{
            'ChannelType': 'training',
            'ContentType': 'text/csv;header=present',
            'CompressionType': 'None',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f'{self.s3_data_path}/{data_filename}',
                }
            }
        }]
    
    def get_automl_problem_config(self) -> Dict[str, Any]:
        """Get AutoML problem type configuration for time series forecasting."""
        return {
            'TimeSeriesForecastingJobConfig': {
                'CompletionCriteria': {
                    'MaxAutoMLJobRuntimeInSeconds': self.MAX_AUTOML_JOB_RUNTIME
                },
                'ForecastFrequency': self.FORECAST_FREQUENCY,
                'ForecastHorizon': self.PREDICTION_LENGTH,
                'ForecastQuantiles': self.FORECAST_QUANTILES,
                'Transformations': {
                    'Filling': {
                        'consumption': {
                            'middlefill': 'zero',
                            'backfill': 'zero'
                        },                      
                    }
                },
                'TimeSeriesConfig': {
                    'TargetAttributeName': 'revenue',
                    'TimestampAttributeName': 'start_date',
                    'ItemIdentifierAttributeName': 'partner_id',
                },
                'CandidateGenerationConfig': {
                    'AlgorithmsConfig': [
                        {
                            'AutoMLAlgorithms': self.AUTOML_ALGORITHMS
                        },
                    ]
                }
            }
        }


# Global settings instance
settings = Settings()
