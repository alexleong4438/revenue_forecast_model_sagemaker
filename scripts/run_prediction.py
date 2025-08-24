#!/usr/bin/env python3
"""Script to make revenue predictions using trained SageMaker model."""

import sys
from pathlib import Path
import argparse
import json

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.prediction.predictor import RevenuePredictor, make_prediction
from src.config.settings import settings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_prediction_data():
    """Create sample prediction data for testing."""
    return {
        "partner_id": ['ACC-000991282', 'ACC-000991283', 'ACC-000991284'],  
        "start_date": ["2024-07-01", "2024-08-01", "2024-09-01"],  
        "revenue": [227.12, 231.01, 160.77]  
    }


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make revenue predictions')
    parser.add_argument(
        '--endpoint-name', 
        type=str, 
        default=settings.SAGEMAKER_ENDPOINT_NAME,
        help=f'SageMaker endpoint name (default: {settings.SAGEMAKER_ENDPOINT_NAME})'
    )
    parser.add_argument(
        '--input-file', 
        type=str, 
        help='CSV file with prediction data (optional, uses sample data if not provided)'
    )
    parser.add_argument(
        '--output-file', 
        type=str, 
        help='File to save prediction results (optional)'
    )
    parser.add_argument(
        '--region', 
        type=str, 
        default=settings.AWS_REGION,
        help=f'AWS region (default: {settings.AWS_REGION})'
    )
    parser.add_argument(
        '--sample-data', 
        action='store_true',
        help='Use sample prediction data instead of file input'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting revenue prediction")
        logger.info(f"Endpoint: {args.endpoint_name}")
        logger.info(f"Region: {args.region}")
        
        # Initialize predictor
        predictor = RevenuePredictor(
            endpoint_name=args.endpoint_name,
            region_name=args.region
        )
        
        # Prepare prediction data
        if args.input_file and not args.sample_data:
            logger.info(f"Loading prediction data from: {args.input_file}")
            input_path = Path(args.input_file)
            if not input_path.exists():
                logger.error(f"Input file not found: {input_path}")
                return 1
            predictions = predictor.predict_from_file(str(input_path))
        else:
            logger.info("Using sample prediction data")
            sample_data = create_sample_prediction_data()
            logger.info(f"Sample data: {sample_data}")
            predictions = predictor.predict_from_dict(sample_data)
        
        # Display results
        logger.info("Prediction Results:")
        logger.info("=" * 50)
        
        if isinstance(predictions, dict):
            if "raw_response" in predictions:
                print(predictions["raw_response"])
            else:
                print(json.dumps(predictions, indent=2, default=str))
        else:
            print(str(predictions))
        
        # Save results if output file specified
        if args.output_file:
            output_path = Path(args.output_file)
            with open(output_path, 'w') as f:
                if isinstance(predictions, dict):
                    json.dump(predictions, f, indent=2, default=str)
                else:
                    f.write(str(predictions))
            logger.info(f"Results saved to: {output_path}")
        
        logger.info("Prediction completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
