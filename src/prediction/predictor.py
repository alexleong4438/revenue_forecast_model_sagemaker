"""Prediction and inference module."""

import boto3
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
import json

from ..config.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RevenuePredictor:
    """Handles revenue predictions using SageMaker endpoints."""
    
    def __init__(self, endpoint_name: Optional[str] = None, region_name: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            endpoint_name: SageMaker endpoint name (defaults to settings)
            region_name: AWS region name (defaults to settings)
        """
        self.endpoint_name = endpoint_name or settings.SAGEMAKER_ENDPOINT_NAME
        self.region_name = region_name or settings.AWS_REGION
        
        # Initialize SageMaker runtime client
        self.sagemaker_runtime = boto3.client(
            "sagemaker-runtime", 
            region_name=self.region_name
        )
        
        logger.info(f"Initialized predictor with endpoint: {self.endpoint_name}")
    
    def prepare_prediction_data(self, prediction_data: Dict[str, List]) -> pd.DataFrame:
        """
        Prepare prediction data as a DataFrame.
        
        Args:
            prediction_data: Dictionary with lists of partner_id, start_date, and revenue
            
        Returns:
            DataFrame ready for prediction
        """
        try:
            df = pd.DataFrame(prediction_data)
            
            # Validate required columns
            required_columns = ['partner_id', 'start_date', 'revenue']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert start_date to datetime if it's not already
            df['start_date'] = pd.to_datetime(df['start_date'])
            
            logger.info(f"Prepared prediction data with {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error preparing prediction data: {e}")
            raise
    
    def predict(self, data: pd.DataFrame, content_type: str = "text/csv") -> Dict[str, Any]:
        """
        Make predictions using the SageMaker endpoint.
        
        Args:
            data: DataFrame with prediction data
            content_type: Content type for the request
            
        Returns:
            Prediction results
        """
        try:
            # Convert DataFrame to appropriate format
            if content_type == "text/csv":
                body = data.to_csv(index=False).encode("utf-8")
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            logger.info(f"Making prediction request to endpoint: {self.endpoint_name}")
            
            # Make prediction request
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType=content_type,
                Body=body,
                Accept="application/json"
            )
            
            # Parse response
            result = response['Body'].read().decode("utf-8")
            
            try:
                # Try to parse as JSON
                parsed_result = json.loads(result)
                logger.info("Successfully made prediction")
                return parsed_result
            except json.JSONDecodeError:
                # Return raw string if not JSON
                logger.info("Prediction returned non-JSON response")
                return {"raw_response": result}
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_from_dict(self, prediction_data: Dict[str, List]) -> Dict[str, Any]:
        """
        Make predictions from a dictionary of data.
        
        Args:
            prediction_data: Dictionary with prediction data
            
        Returns:
            Prediction results
        """
        df = self.prepare_prediction_data(prediction_data)
        return self.predict(df)
    
    def predict_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Make predictions from a CSV file.
        
        Args:
            file_path: Path to CSV file with prediction data
            
        Returns:
            Prediction results
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded prediction data from {file_path}")
            return self.predict(df)
        except Exception as e:
            logger.error(f"Error loading prediction data from file: {e}")
            raise
    
    def format_predictions(self, predictions: Dict[str, Any]) -> pd.DataFrame:
        """
        Format prediction results into a readable DataFrame.
        
        Args:
            predictions: Raw prediction results
            
        Returns:
            Formatted DataFrame with predictions
        """
        try:
            # This is a generic implementation - you may need to adjust based on 
            # the actual format of your SageMaker endpoint responses
            if isinstance(predictions, dict) and "predictions" in predictions:
                # If predictions are in a standard format
                pred_data = predictions["predictions"]
                if isinstance(pred_data, list):
                    return pd.DataFrame(pred_data)
            
            # If raw response, return as-is in a DataFrame
            return pd.DataFrame([predictions])
            
        except Exception as e:
            logger.error(f"Error formatting predictions: {e}")
            # Return raw predictions in a simple format
            return pd.DataFrame([{"raw_predictions": str(predictions)}])


def make_prediction(
    prediction_data: Dict[str, List], 
    endpoint_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to make a prediction.
    
    Args:
        prediction_data: Dictionary with prediction data
        endpoint_name: Optional endpoint name
        
    Returns:
        Prediction results
    """
    predictor = RevenuePredictor(endpoint_name=endpoint_name)
    return predictor.predict_from_dict(prediction_data)
