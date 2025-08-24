"""SageMaker AutoPilot training module."""

import boto3
import sagemaker
import logging
from time import gmtime, strftime, sleep
import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from ..config.settings import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SageMakerTrainer:
    """Handles SageMaker AutoPilot training for time series forecasting."""
    
    def __init__(self, region_name: Optional[str] = None):
        """
        Initialize the SageMaker trainer.
        
        Args:
            region_name: AWS region name (defaults to settings)
        """
        self.region_name = region_name or settings.AWS_REGION
        
        # Initialize AWS clients and sessions
        self.boto_session = boto3.Session(region_name=self.region_name)
        self.sagemaker_session = sagemaker.Session(boto_session=self.boto_session)
        self.sm_client = boto3.client("sagemaker", region_name=self.region_name)
        self.quotas_client = boto3.client("service-quotas", region_name=self.region_name)
        
        # Get execution role
        self.sm_role = settings.SAGEMAKER_EXECUTION_ROLE or sagemaker.get_execution_role(
            sagemaker_session=self.sagemaker_session
        )
        
        logger.info(f"Initialized SageMaker trainer in region: {self.region_name}")
    
    def check_quota(self, instance_type: str, min_count: int = 1) -> bool:
        """
        Check if the account has sufficient quota for the specified instance type.
        
        Args:
            instance_type: EC2 instance type
            min_count: Minimum required count
            
        Returns:
            True if quota is sufficient
        """
        quota_code = settings.EP_INSTANCE_QUOTA_CODE_MAP.get(instance_type)
        if not quota_code:
            logger.warning(f"No quota code found for instance type: {instance_type}")
            return False
        
        try:
            response = self.quotas_client.get_service_quota(
                ServiceCode="sagemaker",
                QuotaCode=quota_code,
            )
            
            quota_value = response["Quota"]["Value"]
            quota_name = response["Quota"]["QuotaName"]
            
            is_sufficient = quota_value >= min_count
            
            if is_sufficient:
                logger.info(f"✅ Quota {quota_value} for {quota_name} >= required {min_count}")
            else:
                logger.warning(f"⚠️ Quota {quota_value} for {quota_name} < required {min_count}")
            
            return is_sufficient
            
        except Exception as e:
            logger.error(f"Error checking quota for {instance_type}: {e}")
            return False
    
    def get_best_instance(self) -> str:
        """
        Get the first available instance type with sufficient quota.
        
        Returns:
            Instance type string or empty string if none available
        """
        available_instances = [
            instance_type for instance_type in settings.EP_INSTANCE_QUOTA_CODE_MAP.keys()
            if self.check_quota(instance_type)
        ]
        
        if available_instances:
            best_instance = available_instances[0]
            logger.info(f"Selected instance type: {best_instance}")
            return best_instance
        else:
            logger.error("No instance types with sufficient quota found")
            return ""
    
    def create_automl_job(self, data_filename: str, job_name_suffix: Optional[str] = None) -> str:
        """
        Create and start an AutoML job for time series forecasting.
        
        Args:
            data_filename: Name of the CSV file in S3
            job_name_suffix: Optional suffix for job name
            
        Returns:
            AutoML job name
        """
        # Generate job name
        timestamp_suffix = strftime("%Y%m%d-%H%M%S", gmtime())
        if job_name_suffix:
            job_name = f"automl-revenue-{job_name_suffix}-{timestamp_suffix}"
        else:
            job_name = f"automl-revenue-{timestamp_suffix}"
        
        logger.info(f"Creating AutoML job: {job_name}")
        
        # Configure job parameters
        input_data_config = settings.get_automl_input_config(data_filename)
        output_data_config = {'S3OutputPath': settings.s3_output_path}
        problem_type_config = settings.get_automl_problem_config()
        automl_job_objective = {'MetricName': 'AverageWeightedQuantileLoss'}
        
        try:
            # Create the AutoML job
            self.sm_client.create_auto_ml_job_v2(
                AutoMLJobName=job_name,
                AutoMLJobInputDataConfig=input_data_config,
                OutputDataConfig=output_data_config,
                AutoMLProblemTypeConfig=problem_type_config,
                AutoMLJobObjective=automl_job_objective,
                RoleArn=self.sm_role,
            )
            
            logger.info(f"AutoML job '{job_name}' created successfully")
            return job_name
            
        except Exception as e:
            logger.error(f"Error creating AutoML job: {e}")
            raise
    
    def monitor_job(self, job_name: str, check_interval: int = 180) -> Dict[str, Any]:
        """
        Monitor an AutoML job until completion.
        
        Args:
            job_name: Name of the AutoML job
            check_interval: Time interval between status checks (seconds)
            
        Returns:
            Job description dict
        """
        logger.info(f"Monitoring AutoML job: {job_name}")
        
        while True:
            try:
                response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
                job_status = response["AutoMLJobStatus"]
                
                if job_status in ("Failed", "Completed", "Stopped"):
                    break
                
                secondary_status = response.get("AutoMLJobSecondaryStatus", "Unknown")
                logger.info(f"{datetime.datetime.now()} - {job_status} - {secondary_status}")
                
                sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring job: {e}")
                raise
        
        # Get final job details
        final_response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        final_status = final_response["AutoMLJobStatus"]
        secondary_status = final_response.get("AutoMLJobSecondaryStatus", "Unknown")
        
        # Calculate duration
        if 'EndTime' in final_response and 'CreationTime' in final_response:
            duration_minutes = (final_response['EndTime'] - final_response['CreationTime']).seconds // 60
        else:
            duration_minutes = "Unknown"
        
        logger.info(f"Job {job_name} finished with status: {final_status} - {secondary_status}")
        logger.info(f"Total duration: {duration_minutes} minutes")
        
        return final_response
    
    def get_best_candidate(self, job_name: str) -> Dict[str, Any]:
        """
        Get the best candidate from an AutoML job.
        
        Args:
            job_name: Name of the AutoML job
            
        Returns:
            Best candidate information
        """
        try:
            job_response = self.sm_client.describe_auto_ml_job_v2(AutoMLJobName=job_name)
            
            if job_response.get('BestCandidate'):
                logger.info("Found best candidate in job details")
                best_candidate = job_response['BestCandidate']
            else:
                logger.info("Searching for best candidate in individual training jobs")
                candidates_response = self.sm_client.list_candidates_for_auto_ml_job(
                    AutoMLJobName=job_name,
                    StatusEquals='Completed',
                    SortOrder='Ascending',
                    SortBy='FinalObjectiveMetricValue',
                )
                
                candidates = candidates_response.get('Candidates', [])
                if not candidates:
                    raise ValueError(f"No completed candidates found for job {job_name}")
                
                # Log all candidates
                logger.info("Completed candidates:")
                for candidate in candidates:
                    metric_value = candidate.get('FinalAutoMLJobObjectiveMetric', {}).get('Value', 'N/A')
                    logger.info(f"  {candidate['CandidateName']}: {metric_value}")
                
                best_candidate = candidates[0]
            
            logger.info(f"Best candidate: {best_candidate.get('CandidateName', 'Unknown')}")
            return best_candidate
            
        except Exception as e:
            logger.error(f"Error getting best candidate: {e}")
            raise
    
    def train_model(self, data_filename: str, job_name_suffix: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete training pipeline: create job, monitor, and get best candidate.
        
        Args:
            data_filename: Name of the CSV file in S3
            job_name_suffix: Optional suffix for job name
            
        Returns:
            Best candidate information
        """
        # Check instance availability
        best_instance = self.get_best_instance()
        if not best_instance:
            raise RuntimeError("No suitable instance types available for training")
        
        # Create and monitor job
        job_name = self.create_automl_job(data_filename, job_name_suffix)
        job_response = self.monitor_job(job_name)
        
        # Check if job completed successfully
        if job_response["AutoMLJobStatus"] != "Completed":
            raise RuntimeError(f"AutoML job failed with status: {job_response['AutoMLJobStatus']}")
        
        # Get best candidate
        best_candidate = self.get_best_candidate(job_name)
        
        logger.info("Training completed successfully!")
        return {
            'job_name': job_name,
            'job_response': job_response,
            'best_candidate': best_candidate
        }
