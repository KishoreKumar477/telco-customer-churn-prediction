import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

# When running from src/, go one level up to project root
files = glob.glob("../reports/metrics/*cv_results.csv")

summary = []

for file in files:
    df = pd.read_csv(file)

    # Get best row from GridSearchCV results
    best = df.loc[df["rank_test_score"] == 1]

    model_name = os.path.basename(file).split("_")[0]

    summary.append({
        "Model": model_name,
        "Best CV F1": best["mean_test_score"].values[0]
    })

summary_df = pd.DataFrame(summary)

# Safety check if no CV files were found or parsed
if summary_df.empty:
    print("No CV result files found in ../reports/metrics/")
    print("Make sure evaluation/training has generated *_cv_results.csv files.")
    exit()

# Sort models
summary_df = summary_df.sort_values("Best CV F1", ascending=False)

print(summary_df)

# Plot comparison
plt.figure(figsize=(8,5))
summary_df.plot(
    x="Model",
    y="Best CV F1",
    kind="barh",
    legend=False
)

plt.title("Model Comparison (Cross Validation F1)")
plt.xlabel("F1 Score")
plt.tight_layout()

os.makedirs("../reports/plots", exist_ok=True)
plt.savefig("../reports/plots/model_comparison.png")

print("Model comparison plot saved to ../reports/plots/model_comparison.png")