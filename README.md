Telco Customer Churn Prediction
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/scikit--learn-1.2+-orange
https://img.shields.io/badge/Streamlit-1.20+-red
https://img.shields.io/badge/License-MIT-yellow.svg

A complete end-to-end machine learning project for predicting customer churn in the telecommunications industry. The project includes data preprocessing, feature engineering, multiple model training (Logistic Regression, Random Forest, XGBoost, SVM, LightGBM), hyperparameter tuning, model evaluation, and an interactive Streamlit dashboard for real-time predictions.

https://webapp.png <!-- Add a screenshot of your app -->

📌 Table of Contents
Project Overview

Dataset

Features

Project Structure

Installation

Usage

Train Models

Run Streamlit App

Make Predictions

Models & Performance

Results

Future Work

License

🎯 Project Overview
Customer churn is a critical metric for subscription-based businesses. This project aims to predict whether a customer will leave (churn) based on their demographic, account, and service usage information. The main goals are:

Build a robust preprocessing pipeline to handle missing values, encode categorical variables, and create meaningful features.

Train and compare multiple classification models using grid search with cross-validation.

Evaluate models using precision, recall, F1-score, ROC-AUC, and confusion matrices.

Deploy the best model via a user-friendly Streamlit web application for interactive predictions.

📊 Dataset
The dataset used is the IBM Telco Customer Churn dataset, which contains 7,043 customers with 21 features. It includes:

Demographics: gender, senior citizen, partner, dependents

Services: phone, multiple lines, internet, online security, backup, device protection, tech support, streaming TV, streaming movies

Account information: tenure, contract type, payment method, paperless billing, monthly charges, total charges

Target: Churn (Yes/No)

Source: Kaggle - Telco Customer Churn

🔧 Features
Modular codebase: Separate modules for configuration, data loading, preprocessing, training, evaluation, and prediction.

Custom feature engineering: Created AvgMonthlySpend, TenureGroup, ServiceCount, and ChargeTenureInteraction.

Config-driven: All column names and model parameters are managed via config.yaml.

Multiple models: Logistic Regression, Random Forest, XGBoost, SVM, LightGBM.

Hyperparameter tuning: GridSearchCV with F1-score optimization.

Comprehensive evaluation: Classification reports, confusion matrices, ROC curves saved as images.

Interactive dashboard: Built with Streamlit for real-time churn probability predictions.

📁 Project Structure
text
Telco Customer Churn/
├── .vscode/                    # VS Code settings (optional)
├── models/                      # Saved models, pipelines, test data, plots (created after training)
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── config_loader.py         # Loads config.yaml
│   ├── data_loader.py           # Loads raw data
│   ├── logger.py                 # Logging setup
│   ├── preprocessing.py         # Custom feature engineering and pipeline
│   ├── train.py                 # Trains a single model (argument: --model)
│   ├── evaluate.py              # Evaluates a trained model (argument: --model)
│   └── predict.py               # Makes predictions on new data
├── config.yaml                   # Configuration: data path, column lists, split params
├── dataset.csv                    # Raw data (place your CSV here)
├── main.py                        # Orchestrates training and evaluation for all models
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup (optional)
├── webapp.png                      # Screenshot of Streamlit app
└── README.md                       # This file
⚙️ Installation
Clone the repository

bash
git clone https://github.com/yourusername/telco-churn.git
cd telco-churn
Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Place your dataset as dataset.csv in the project root (or update config.yaml with the correct path).

🚀 Usage
Train Models
To train a single model:

bash
python src/train.py --model logistic   # options: logistic, rf, xgb, svm, lgbm
To train and evaluate all models at once:

bash
python main.py
Trained models, the preprocessing pipeline, test data, and evaluation plots will be saved in the models/ directory.

Run Streamlit App
After training at least one model, launch the interactive dashboard:

bash
streamlit run app.py   # Make sure you have an app.py file (see below)
If you haven't created app.py yet, here's a minimal version:

python
# app.py
import streamlit as st
import joblib
import pandas as pd

pipeline = joblib.load('models/pipeline.pkl')
model = joblib.load('models/xgb_best.pkl')   # change to your best model

st.title("Telco Customer Churn Prediction")
# ... add input fields and prediction logic
Make Predictions
Use the command-line prediction script:

bash
python src/predict.py
Modify the sample dictionary inside predict.py to test your own customer data.

🤖 Models & Performance
The following models were trained with grid search (5-fold cross-validation) optimizing for F1-score:

Model	Best Parameters	CV F1 (mean)	Test F1	ROC-AUC
Logistic Regression	{'C': 1, 'max_iter': 1000}	0.XX	0.XX	0.XX
Random Forest	{'max_depth': 10, 'n_estimators': 200}	0.XX	0.XX	0.XX
XGBoost	{'learning_rate': 0.1, 'n_estimators': 200}	0.XX	0.XX	0.XX
SVM	{'C': 10, 'gamma': 'scale'}	0.XX	0.XX	0.XX
LightGBM	{'n_estimators': 200, 'num_leaves': 50}	0.XX	0.XX	0.XX
Detailed evaluation reports, confusion matrices, and ROC curves are saved in the models/ folder.

📈 Results
The best performing model was XGBoost with an F1-score of 0.XX and ROC-AUC of 0.XX on the test set.

Feature importance analysis (from tree-based models) revealed that contract type, tenure, and monthly charges are the strongest predictors of churn.

The Streamlit app provides an intuitive interface to explore predictions and understand the key drivers of churn for individual customers.

🔮 Future Work
Incorporate SHAP explanations in the dashboard.

Deploy the app online (Streamlit Cloud, Heroku, etc.).

Experiment with deep learning models (simple neural networks).

Automate retraining with new data.

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

Author: Your Name
GitHub: yourusername
LinkedIn: Your Profile

If you find this project useful, please give it a ⭐!

