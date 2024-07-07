# Fraud Detection Project

## Project Overview

This project aims to develop a machine learning model to detect fraudulent transactions for a Integra's credit card system. The goal is to identify transactions that are likely to be fraudulent so that a team of fraud analysts can review them. The model will help in prioritizing the transactions for review, thereby maximizing the amount of fraud value that can be prevented.

## Problem Statement

A bank is facing a growing fraud problem where its customers' cards are being defrauded, causing customer dissatisfaction. Integra plans to introduce better transactional monitoring to identify and prevent fraudulent transactions. A small team of fraud analysts will review up to 400 transactions per month flagged by the model. The primary objective is to build a model that predicts the likelihood of a transaction being fraudulent based on historical data.

## Data

The bank has provided 1 year of historical transactional data along with fraud flags. Additionally, a brief data dictionary describing some general payment terms is provided.

### Data Files

- `data/raw/labels_obf.csv`: Contains the labels indicating whether each transaction is fraudulent.
- `data/raw/transactions_obf.csv`: Contains the transactional data.

### Processed Data

- `data/processed/`: Directory for storing processed data.

## Project Structure

The project is organized as follows:

```
FRAUD_ANALYSIS/
│
├── data/
│   ├── processed/
│   │   ├── feature_engineered_data.csv
│   └── raw/
│       ├── labels_obf.csv
│       └── transactions_obf.csv
│
├── models/
│   ├── Random Forest_best_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── reports/
│   └── Random Forest_predictions.csv
│
├── scripts/
│   ├── __init__.py
│   ├── predict_pipeline.py
│   └── run_pipeline.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   └── utils.py
│
├── README.md
└── requirements.txt
```

## Usage

### Setup

1. Navigate to the project directory:
   ```
   cd fraud_analysis
   ```

2. Set the path to the project repository:
   ```
   export PYTHONPATH=$PYTHONPATH:/path/to/the/fraud_analysis/

   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```


### Exploratory Data Analysis (EDA)

Explore the data using the Jupyter notebook:
```
notebooks/exploratory_analysis.ipynb
```


### Run Complete Pipeline, takes approximately[] 

To run the complete pipeline from data ingestion to prediction, use the run pipeline script:
```bash
python scripts/run_pipeline.py
```

## Results
The results of the model predictions can be found in the `reports` directory:
- `Random Forest_predictions.csv`

### Model Storage

The trained models and preprocessing pipelines are saved in the `models` directory. These files can be loaded later for making predictions or further analysis:
- `Random Forest_best_model.pkl`
- `preprocessor.pkl`


## Contact

For any questions or inquiries, please contact the FeatureSpace ML Team.
