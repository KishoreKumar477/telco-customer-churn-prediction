# src/train.py
import argparse
import joblib
import pandas as pd
import os
import sys
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings

import matplotlib.pyplot as plt

# Add project root to path (if running directly)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.preprocessing import build_full_pipeline
from src.config_loader import load_config
from src.logger import get_logger

logger = get_logger(__name__)

MODEL_CLASSES = {
    'LogisticRegression': LogisticRegression,
    'RandomForest': RandomForestClassifier,
    'XGBoost': XGBClassifier,
    'SVM': SVC
}

def train_model(model_name, config):
    # Load data
    data_path = config['data']['path']
    df = load_data(data_path)
    target = config['data']['target']
    X = df.drop(columns=[target])
    y = df[target].map({'Yes': 1, 'No': 0})

    # Split
    test_size = config['split']['test_size']
    random_state = config['split']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Build preprocessing pipeline (do NOT fit yet)
    preprocessing_pipeline = build_full_pipeline(config)

    # Save test data for later evaluation
    os.makedirs('models/test_data', exist_ok=True)
    joblib.dump((X_test, y_test), f'models/test_data/test_data_{model_name}.pkl')

    # Get model configuration
    model_cfg = config['models'][model_name]
    model_class = MODEL_CLASSES[model_name]
    fixed_params = model_cfg.get('fixed_params', {})
    param_grid = model_cfg['params']

    # Create base model with required defaults for specific models
    if model_name == "SVM":
        base_model = model_class(probability=True, **fixed_params)
    elif model_name == "XGBoost":
        # use_label_encoder is deprecated in newer XGBoost versions
        base_model = model_class(**fixed_params)
    else:
        base_model = model_class(**fixed_params)

    # Suppress LightGBM feature name warning (caused by sklearn internal validation)
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
        category=UserWarning
    )
    # Grid search
    logger.info(f"Starting GridSearchCV for {model_name}...")
    scoring_metric = config.get("training", {}).get("scoring", "f1")

    from sklearn.pipeline import Pipeline

    full_pipeline = Pipeline([
        ("preprocessing", preprocessing_pipeline),
        ("model", base_model)
    ])

    # Prefix model parameters for GridSearch
    param_grid = {f"model__{k}": v for k, v in param_grid.items()}

    grid = GridSearchCV(full_pipeline, param_grid, cv=5,
                        scoring=scoring_metric, n_jobs=-1)

    grid.fit(X_train, y_train)

    logger.info(f"Best params for {model_name}: {grid.best_params_}")
    logger.info(f"Best cross-val F1: {grid.best_score_:.4f}")

    # Save best model
    os.makedirs('models/trained', exist_ok=True)
    joblib.dump(grid.best_estimator_, f'models/trained/{model_name}_best.pkl')

    # Feature importance visualization for Logistic Regression
    try:
        if model_name == "LogisticRegression":
            best_pipeline = grid.best_estimator_
            model = best_pipeline.named_steps["model"]
            preprocessing = best_pipeline.named_steps["preprocessing"]

            # Get transformed feature names from preprocessing pipeline
            try:
                # pass original column names so ColumnTransformer can expand OHE names
                feature_names = preprocessing.get_feature_names_out(X.columns)
            except Exception:
                # Fallback if transformer does not support feature names
                feature_names = [f"feature_{i}" for i in range(len(model.coef_[0]))]

            importance = model.coef_[0]

            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            plt.figure(figsize=(8,5))
            imp_df.head(10).plot(
                x="Feature",
                y="Importance",
                kind="barh",
                legend=False
            )
            plt.title("Top Features Affecting Churn")
            plt.tight_layout()
            os.makedirs("reports/plots", exist_ok=True)
            plt.savefig("reports/plots/feature_importance.png")
            plt.close()

            logger.info("Feature importance plot saved to reports/plots/feature_importance.png")
    except Exception as e:
        logger.warning(f"Could not generate feature importance plot: {e}")

    logger.info(f"Model {model_name} saved.")
    return grid.best_estimator_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Model name to train: LogisticRegression, RandomForest, XGBoost, SVM, LightGBM')
    args = parser.parse_args()

    config = load_config()
    if args.model not in config['models'] or not config['models'][args.model].get('active', False):
        logger.error(f"Model {args.model} is not active or not defined in config.")
        sys.exit(1)

    train_model(args.model, config)
