# src/evaluate.py
import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model_name,config):
    # Load model and test data
    model_path = f'models/trained/{model_name}_best.pkl'
    test_data_path = f'models/test_data/test_data_{model_name}.pkl'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")

    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    # Predict
    y_pred = model.predict(X_test)

    # Handle models without predict_proba
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        logger.warning(f"{model_name} does not support predict_proba. ROC and PR curves will be skipped.")
        y_proba = None

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    os.makedirs("reports/metrics", exist_ok=True)
    report_df.to_csv(f'reports/metrics/{model_name}_classification_report.csv')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    os.makedirs("reports/plots", exist_ok=True)
    plt.savefig(f'reports/plots/{model_name}_confusion_matrix.png')
    plt.close()

    if y_proba is not None:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.savefig(f'reports/plots/{model_name}_roc_curve.png')
        plt.close()

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.savefig(f'reports/plots/{model_name}_pr_curve.png')
        plt.close()

    logger.info(f"Evaluation for {model_name} completed. Metrics saved in reports/metrics and plots saved in reports/plots.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='Model name to evaluate')
    args = parser.parse_args()
    evaluate_model(args.model)