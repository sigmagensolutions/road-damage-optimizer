import pandas as pd

# Load data
probs_df = pd.read_csv("inference_results_with_probs.csv")
truth_df = pd.read_csv("ground_truth_flags.csv")
merged = probs_df.merge(truth_df, on="image")

# Define all quadrant outcomes
classes = ["D00", "D10", "D20", "D40"]
costs = {
    "D00": {"TP": 1000, "TN": 0, "FP": 1200, "FN": 2000},
    "D10": {"TP": 900,  "TN": 0, "FP": 1100, "FN": 1800},
    "D20": {"TP": 1400, "TN": 0, "FP": 1600, "FN": 2200},
    "D40": {"TP": 1800, "TN": 0, "FP": 2000, "FN": 3000},
}


def evaluate_thresholds(thresholds):
    total_cost = 0

    for i, cls in enumerate(classes):
        threshold = thresholds[i]
        prob_col = f"{cls}_prob"
        actual_col = cls
        pred_col = f"{cls}_predicted"

        # Prediction based on threshold
        merged[pred_col] = merged[prob_col] > threshold

        for _, row in merged.iterrows():
            predicted = row[pred_col]
            actual = row[actual_col]

            if predicted and actual:
                total_cost += costs[cls]["TP"]
            elif predicted and not actual:
                total_cost += costs[cls]["FP"]
            elif not predicted and actual:
                total_cost += costs[cls]["FN"]
            else:
                total_cost += costs[cls]["TN"]

    return total_cost

# Example test
thresholds = [0.6, 0.5, 0.7, 0.8]
score = evaluate_thresholds(thresholds)
print(f"🔍 Total net cost for {thresholds}: ${score}")
