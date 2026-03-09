📊 Telco Customer Churn Prediction System

## 🎥 Demo

<p align="center">
  <img src="demo.gif" alt="App Demo" width="800">
</p>


An end-to-end Machine Learning system that predicts whether a telecom customer is likely to churn (leave the service).
```
This project demonstrates a complete ML lifecycle including:
	•	Data preprocessing pipelines
	•	Multiple machine learning models
	•	Hyperparameter tuning
	•	Cross-validation model comparison
	•	Model evaluation and visualization
	•	Interactive Streamlit web application
```
🚀 Demo

Web Application
```
streamlit run app.py
```



Machine Learning Pipeline

The system follows a modular ML architecture.

```
Raw Dataset
     │
     ▼
Data Preprocessing
     │
     ▼
Train / Test Split
     │
     ▼
Pipeline + ML Models
     │
     ▼
GridSearchCV (Hyperparameter Tuning)
     │
     ▼
Cross Validation
     │
     ▼
Model Evaluation
     │
     ▼
Model Comparison
     │
     ▼
Streamlit Web App
```

Models Implemented

The following algorithms were trained and evaluated:

```
Model                    Purpose
Logistic Regression      Baseline linear classifier
Random Forest            Ensemble tree model
SVM                      Margin-based classifier
XGBoost                  Gradient boosting model


Model selection was performed using 5-fold cross validation with F1 Score.
```
📈 Model Evaluation Metrics

Evaluation metrics include:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1 Score
	•	ROC Curve
	•	Precision-Recall Curve
	•	Confusion Matrix

Model Comparison

All models are compared using cross-validation scores.

Output visualization:

Model Performance (F1 Score)

XGBoost         ███████████
RandomForest    ████████████  
LogisticReg     ███████████
SVM             ██████████


📂 Project Structure
```
Telco-Customer-Churn
│
├── data
│   ├── Telco-Customer-Churn.csv
│   └── test.csv
│
├── models
│   ├── trained
│   │   ├── LogisticRegression_best.pkl
│   │   ├── RandomForest_best.pkl
│   │   ├── SVM_best.pkl
│   │   └── XGBoost_best.pkl
│   │
│   └── test_data
│       └── test_data_*.pkl
│
├── reports
│   ├── plots
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── pr_curve.png
│   │   ├── feature_importance.png
│   │   └── model_comparison.png
│   │
│   └── metrics
│       ├── classification_report.csv
│       └── *_cv_results.csv
│
├── src
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── model_comparison.py
│   └── logger.py
│
├── app.py
├── main.py
├── config.yaml
├── requirements.txt
└── README.md
```
Installation

Clone the repository:
```
git clone https://github.com/kishorekumar040707/telco-churn-prediction.git
cd telco-churn-prediction
```


Create environment:
```
python -m venv customer_churn
source customer_churn/bin/activate
```
Install dependencies:
```
pip install -r requirements.txt
```
🏋️ Training Models
```
python main.py --mode train --model LogisticRegression
python main.py --mode train --model RandomForest
python main.py --mode train --model SVM
python main.py --mode train --model XGBoost
```

 Evaluate Models
```
python main.py evaluate --model LogisticRegression
```

 Predict Churn

Run prediction:
```
python main.py --mode predict --model LogisticRegression --input data/test.csv
```

Run Web Application
```
streamlit run app.py
```
This launches an interactive churn prediction dashboard.


Feature Importance
```
Key factors affecting churn include:
	•	Contract type
	•	Monthly charges
	•	Tenure
	•	Internet service
	•	Payment method
```

Dataset

Dataset used:

Telco Customer Churn Dataset
```
Features include:
	•	Customer demographics
	•	Account information
	•	Service subscriptions
	•	Billing details

Target variable:
Churn (Yes / No)
```
```
Key Insights
	•	Customers with month-to-month contracts churn more frequently
	•	Higher monthly charges correlate with churn
	•	Longer tenure reduces churn probability
```
Technologies Used
```
Category          Tools
Programming       Python
Data Processing   Pandas, NumPy
Machine Learning  Scikit-Learn, XGBoost, LightGBM
Visualization     Matplotlib
Web App           Streamlit
```



Author
```
Kishore Kumar
Machine Learning & Data Science Enthusiast
sunday march 8 2026
```

