import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from evaluate_thresholds import evaluate_thresholds
from genetic_optimizer import genetic_algorithm

# Set paths
LABEL_DIR = "data/labels"
CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_MAP = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

def run_classifier():
    print("📸 Running classifier (classify.py)...")
    os.system("python classify.py")  # Or import classify and call a function

def build_ground_truth_map():
    print("📄 Building ground truth from YOLO labels...")
    records = []
    for label_file in os.listdir(LABEL_DIR):
        if not label_file.endswith(".txt"):
            continue
        filepath = os.path.join(LABEL_DIR, label_file)
        image_name = os.path.splitext(label_file)[0] + ".jpg"
        class_flags = {cls: 0 for cls in CLASS_NAMES}
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                cls_name = CLASS_MAP.get(cls_id)
                if cls_name:
                    class_flags[cls_name] = 1
        record = {"image": image_name}
        record.update(class_flags)
        records.append(record)
    df = pd.DataFrame(records)
    df.to_csv("ground_truth_flags.csv", index=False)
    print("✅ Ground truth saved to ground_truth_flags.csv")

def plot_cost_history():
    if not os.path.exists("cost_history.csv"):
        print("⚠️ cost_history.csv not found.")
        return
    df = pd.read_csv("cost_history.csv")
    plt.plot(df["Generation"], df["Best_Cost"], marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Cost ($)")
    plt.title("Cost Evolution Over Generations")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cost_evolution.png")
    plt.show()
    print("📊 Saved: cost_evolution.png")

def run_pipeline():
    print("\n🔁 Starting full pipeline...\n")

    # STEP 1: Run classification
    run_classifier()

    # STEP 2: Generate ground truth flags
    build_ground_truth_map()

    # STEP 3: Run genetic optimizer
    print("\n⚙️ Optimizing thresholds...")
    best_thresholds, best_cost = genetic_algorithm()

    # STEP 4: Visualize
    print("\n📈 Generating plot...")
    plot_cost_history()

    # STEP 5: Output final result
    print("\n✅ Final thresholds and cost:")
    with open("best_thresholds.json") as f:
        print(json.dumps(json.load(f), indent=2))

if __name__ == "__main__":
    run_pipeline()
