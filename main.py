# main.py

import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict
from src.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Telco Churn ML Pipeline")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "evaluate", "predict"],
                        help="Mode to run: train / evaluate / predict")

    parser.add_argument("--model", type=str, required=True,
                        help="Model name (LogisticRegression, RandomForest, XGBoost, SVM)")

    parser.add_argument("--input", type=str, required=False,
                        help="Input CSV path (required for predict mode)")

    parser.add_argument("--output", type=str, required=False,
                        help="Output CSV path (optional for predict mode)")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    if args.mode == "train":
        train_model(args.model, config)

    elif args.mode == "evaluate":
        evaluate_model(args.model, config)

    elif args.mode == "predict":
        if args.input is None:
            raise ValueError("Input file path is required for predict mode.")
        predict(args.model, args.input, args.output)


if __name__ == "__main__":
    main()


