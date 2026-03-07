

# src/predict.py

import argparse
import joblib
import pandas as pd
import os
import sys

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import get_logger

logger = get_logger(__name__)


def predict(model_name, input_path, output_path=None):
    # Load trained model
    model_path = f"models/{model_name}_best.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Loaded model: {model_name}")

    # Load input data
    data = pd.read_csv(input_path)
    logger.info(f"Loaded input data from {input_path}")

    # Make predictions
    predictions = model.predict(data)

    # Add probabilities if supported
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)[:, 1]
        data["prediction_proba"] = probabilities

    data["prediction"] = predictions

    # Default output path
    if output_path is None:
        output_path = f"models/{model_name}_predictions.csv"

    # Save predictions
    data.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    print("Prediction completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name used during training")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=False,
                        help="Optional output CSV path")

    args = parser.parse_args()

    predict(args.model, args.input, args.output)



    