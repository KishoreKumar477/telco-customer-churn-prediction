"""
preprocessing.py
Complete preprocessing pipeline for Telco Customer Churn dataset.
Includes feature engineering, binary mapping, one-hot encoding, and scaling.
All column names are configurable via config.yaml.
"""


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config_loader import load_config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that creates new features from raw Telco data.
    Expected columns: tenure, MonthlyCharges, TotalCharges, and service columns.
    """
    def __init__(self, service_cols=None):
        if service_cols is None:
            self.service_cols = [
                'PhoneService', 'MultipleLines', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies'
            ]
        else:
            self.service_cols = service_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Ensure numeric columns are properly typed
        X['tenure'] = pd.to_numeric(X['tenure'], errors='coerce').fillna(0)
        X['MonthlyCharges'] = pd.to_numeric(X['MonthlyCharges'], errors='coerce').fillna(0)
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)

        # 1. Average Monthly Spend
        X['AvgMonthlySpend'] = np.where(
            X['tenure'] > 0,
            X['TotalCharges'] / X['tenure'],
            0
        )

        # 2. Tenure groups (for one‑hot encoding later)
        X['TenureGroup'] = pd.cut(
            X['tenure'],
            bins=[0, 6, 12, 24, np.inf],
            labels=['0-6', '6-12', '12-24', '24+'],
            right=False
        )

        # 3. Service count – only count columns that exist in the data
        #feature aggregation.

        #to sum the cols value individually 1,0,1,1=3
        existing_service = [c for c in self.service_cols if c in X.columns]
        X['ServiceCount'] = (X[existing_service] == 'Yes').sum(axis=1)

        # 4. Interaction feature
        X['ChargeTenureInteraction'] = X['MonthlyCharges'] * X['tenure']

        return X


def map_binary_values(df):
    """
    Convert binary categorical columns to 0/1.
    - Yes/No columns: Partner, Dependents, PhoneService, PaperlessBilling
    - Gender: Male → 0, Female → 1
    """
    df = df.copy()
    try:
        yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in yes_no_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
        return df
    except KeyError as e:
        print(f"Error in mapping binary values: Missing column {e}")
        raise


def build_preprocessor(config):
    """
    Build the ColumnTransformer using column lists from the configuration.
    """
    binary_cols = config['columns']['binary']
    cat_cols = config['columns']['categorical']
    num_cols = config['columns']['numeric']

    # Create a FunctionTransformer for binary mapping inside the ColumnTransformer
    binary_mapper = FunctionTransformer(map_binary_values, validate=False)

    preprocessor = ColumnTransformer([
        ('binary', binary_mapper, binary_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols)
    ])
    return preprocessor


def build_full_pipeline(config):
    """
    Return the complete pipeline that first applies feature engineering,
    then the preprocessing column transformer.
    """
    service_cols = config['columns'].get('service_cols', None)
    engineer = FeatureEngineer(service_cols=service_cols)
    preprocessor = build_preprocessor(config)

    pipeline = Pipeline([
        ('engineer', engineer),
        ('preprocessor', preprocessor)
    ])
    return pipeline


if __name__ == "__main__":
    # Quick test: load config and print the pipeline
    config = load_config()
    pipe = build_full_pipeline(config)
    print(pipe)