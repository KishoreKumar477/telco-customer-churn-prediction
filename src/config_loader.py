# config_loader.py
import yaml

def load_config(config_path=r"Telco Customer Churn/config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

