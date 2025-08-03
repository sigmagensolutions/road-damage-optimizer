# 🛣️ Road Damage Optimizer

**A real-world computer vision pipeline for optimizing road damage detection using class-level threshold tuning and cost-aware decision logic.**

---

## 🚀 Overview

This project demonstrates a practical, deployment-ready machine learning pipeline for automated road damage detection using drone or street-level imagery. It combines:

- A fine-tuned YOLOv8 object detection model trained on [RDD2022](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022?resource=download)
- Softmax-style class probability output for each image
- A custom cost-based evaluation of the confusion matrix
- A genetic algorithm to optimize classification thresholds per damage type
- Outputs ready for post-deployment monitoring, feedback loops, and retraining

Rather than optimizing for **accuracy**, this pipeline focuses on **minimizing real-world cost** — such as the financial impact of missing a needed road repair or executing an unnecessary one.

---

## 🔧 Pipeline Diagram

Images → YOLOv8 → Class Probabilities
        ↓
  Ground Truth Flags (YOLO labels)
        ↓
   Threshold Optimizer (Genetic Algorithm)
        ↓
   Optimized Decision Logic (Per-class)

## Project Structure

road_damage_project/
├── main.py                         # Orchestrates the full pipeline
├── classify.py                     # Runs YOLOv8 inference and outputs class probabilities
├── build_ground_truth.py           # Converts YOLO .txt labels into per-image class flags
├── evaluate_thresholds.py          # Defines cost-based evaluation for a threshold set
├── genetic_optimizer.py            # Evolves per-class thresholds using GA
├── data/
│   ├── images/                     # Road images
│   └── labels/                     # YOLOv8 label files (.txt)
├── model/
│   └── YOLOv8_Small_RDD.pt         # Fine-tuned YOLOv8 weights from oracl4
├── inference_results_with_probs.csv
├── ground_truth_flags.csv
├── best_thresholds.json
├── cost_history.csv
├── cost_evolution.png
└── rdd2022.yaml

## 🧠 Key Concepts

Confusion matrix costs: Every outcome (TP, FP, FN, TN) has an associated dollar value.
Threshold tuning: Instead of using a fixed 0.5 threshold, this pipeline learns the best per-class thresholds.
Heuristic optimization: A simple, robust genetic algorithm searches for the lowest total cost.
Deployment-aware: Designed with real-world use cases, budget constraints, and future feedback mechanisms in mind.

## 📊 Sample Outputs

inference_results_with_probs.csv — Probabilities per image per class
ground_truth_flags.csv — Binary class presence flags from YOLO labels
cost_evolution.png — Chart of best cost per generation
best_thresholds.json — Final optimized thresholds and cost

## 🏗️ Setup

1. Clone the repo
    git clone https://github.com/sigmagensolutions/road-damage-optimizer.git
    cd road-damage-optimizer
2. Set up your environment
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
3. Add images and labels
    Place your .jpg images in data/images/ and the corresponding YOLO .txt files in data/labels/.
4. Download the YOLOv8 model weights
    From: https://github.com/oracl4/RoadDamageDetection
    Place the file here:  model/YOLOv8_Small_RDD.pt

## ✅ Run the Full Pipeline

python main.py

This will:
Run inference
Generate ground truth flags
Optimize per-class thresholds using a genetic algorithm
Output cost evolution and the final recommended threshold set

## 🔬 Data & Model Credits

Dataset: RDD2022 Road Damage Dataset
Model weights: YOLOv8_Small_RDD.pt by oracl4
YOLOv8: Ultralytics YOLOv8

## Potential Next Steps

Add real-time feedback integration (crew review, confirmed false positives/negatives)
Automate retraining and re-optimization over time
Extend to other asset classes: sidewalks, signs, bridges
Deploy to an embedded or edge device for real-time use

## 👋 Author
Built by a data science executive with over two decades of experience applying machine learning and optimization techniques to real-world business problems. This project showcases not just predictive modeling but a deployment mindset and a focus on cost-aware automation.