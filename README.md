# Revenue Forecast Model with SageMaker

A time series forecasting project that predicts partner revenue using AWS SageMaker AutoPilot.

## Project Structure

```
revenue_forecast_model_sagemaker/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessor.py          # Data preprocessing logic
│   ├── models/
│   │   ├── __init__.py
│   │   └── sagemaker_trainer.py     # SageMaker AutoPilot training
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py             # Prediction service
│   └── config/
│       ├── __init__.py
│       └── settings.py              # Configuration settings
├── data/
│   ├── raw/                         # Raw data files
│   ├── processed/                   # Processed data files
│   └── split_data/                  # Train/test splits
├── scripts/
│   ├── run_preprocessing.py         # Data preprocessing script
│   ├── train_model.py              # Model training script
│   └── run_prediction.py           # Prediction script
├── tests/
│   └── __init__.py
├── .env.example                     # Environment variables template
├── .gitignore
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package setup
└── README.md                       # This file
```

## Features

- **Data Preprocessing**: Parse and clean time series revenue data
- **Rolling Forecast Split**: Create proper train/test splits for time series
- **SageMaker AutoPilot**: Automated machine learning for time series forecasting
- **Prediction Service**: Make predictions using trained models
- **Multiple Algorithms**: Supports CNN-QR, DeepAR, Prophet, NPTS, ARIMA, ETS

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure AWS settings
4. Configure AWS credentials for SageMaker access

## Usage

### Data Preprocessing
```bash
python scripts/run_preprocessing.py
```

### Model Training
```bash
python scripts/train_model.py
```

### Making Predictions
```bash
python scripts/run_prediction.py
```

## Configuration

Update `src/config/settings.py` to modify:
- AWS region and S3 bucket settings
- Model parameters
- Forecast horizon and quantiles
- Algorithm selection

## Data Format

Input data should be CSV with columns:
- `partner_id`: Partner identifier
- `start_date`: Date in YYYY-MM-DD format
- `revenue`: Revenue value
- `period_type`: MONTH or WEEK
- `report_type`: Report type identifier

## AWS Resources Required

- SageMaker execution role
- S3 bucket for data storage
- Appropriate service quotas for endpoint instances
