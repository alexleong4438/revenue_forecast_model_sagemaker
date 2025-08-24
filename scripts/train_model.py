#!/usr/bin/env python3
"""Script to train the revenue forecasting model using SageMaker AutoPilot."""

import sys
from pathlib import Path
import argparse

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.sagemaker_trainer import SageMakerTrainer
from src.config.settings import settings
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train revenue forecasting model')
    parser.add_argument(
        '--data-file', 
        type=str, 
        default='RevenueShare_Month.csv',
        help='Name of the data file in S3 (default: RevenueShare_Month.csv)'
    )
    parser.add_argument(
        '--job-suffix', 
        type=str, 
        help='Optional suffix for the AutoML job name'
    )
    parser.add_argument(
        '--region', 
        type=str, 
        default=settings.AWS_REGION,
        help=f'AWS region (default: {settings.AWS_REGION})'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting model training with SageMaker AutoPilot")
        logger.info(f"Data file: {args.data_file}")
        logger.info(f"Region: {args.region}")
        
        # Initialize trainer
        trainer = SageMakerTrainer(region_name=args.region)
        
        # Check prerequisites
        logger.info("Checking AWS prerequisites...")
        if not settings.SAGEMAKER_EXECUTION_ROLE:
            logger.warning("SAGEMAKER_EXECUTION_ROLE not set in environment variables")
            logger.info("Will attempt to use default SageMaker execution role")
        
        # Train model
        logger.info("Starting training process...")
        result = trainer.train_model(
            data_filename=args.data_file,
            job_name_suffix=args.job_suffix
        )
        
        # Log results
        logger.info("Training completed successfully!")
        logger.info(f"Job Name: {result['job_name']}")
        logger.info(f"Best Candidate: {result['best_candidate'].get('CandidateName', 'Unknown')}")
        
        # Save results summary
        summary_file = Path("training_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Training Summary\\n")
            f.write(f"================\\n")
            f.write(f"Job Name: {result['job_name']}\\n")
            f.write(f"Status: {result['job_response']['AutoMLJobStatus']}\\n")
            f.write(f"Best Candidate: {result['best_candidate'].get('CandidateName', 'Unknown')}\\n")
            if 'FinalAutoMLJobObjectiveMetric' in result['best_candidate']:
                metric = result['best_candidate']['FinalAutoMLJobObjectiveMetric']
                f.write(f"Best Metric ({metric['MetricName']}): {metric['Value']}\\n")
        
        logger.info(f"Training summary saved to: {summary_file}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
