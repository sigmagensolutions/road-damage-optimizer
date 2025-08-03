# 🛣️ Road Damage Optimizer

A real-world computer vision pipeline for optimizing road damage detection using class-level threshold tuning and cost-aware decision logic.

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

## 🛠️ Pipeline Diagram


Images → YOLOv8 → Class Probabilities  
             ↓  
      Ground Truth Flags (YOLO labels)  
             ↓  
Threshold Optimizer (Genetic Algorithm)  
             ↓  
Optimized Decision Logic (Per-class)


## 📁 Project Structure

```
road_damage_project/
├── main.py                             # Orchestrates the full pipeline
├── classify.py                         # Runs inference and outputs class probabilities
├── build_ground_truth.py               # Parses YOLO labels into per-image flags
├── evaluate_thresholds.py              # Evaluates cost given thresholds and predictions
├── genetic_optimizer.py                # Genetic algorithm optimizer
├── visualize.py                        # Plots cost over generations
├── cost_history.csv                    # Cost values across generations
├── cost_evolution.png                  # Visual plot of optimization
├── inference_results_with_probs.csv    # Image predictions with class probabilities
├── best_thresholds.json                # Saved optimal thresholds
├── ground_truth_flags.csv              # Per-image binary flags (ground truth)
├── model/
│   └── YOLOv8_Small_RDD.pt             # Fine-tuned model weights from [oracl4 repo](https://github.com/oracl4/RoadDamageDetection)
├── data/
│   ├── images/                         # Inference image set
│   └── labels/                         # YOLO-format annotation text files
```

## 💡 Background & Rationale

Many computer vision deployments rely on accuracy or F1 score — but in the real world, *not all mistakes are equal*. A missed road repair (false negative) might cost thousands in damage or liability, while an unnecessary repair (false positive) may simply be wasteful.

This pipeline uses **custom cost functions** to guide model threshold tuning via **evolutionary optimization**, a method I've used throughout my career for complex, non-differentiable problems. 

The goal is to model real-world deployment as closely as possible, including the ability to:

- Adjust decision thresholds post hoc
- Integrate into maintenance workflows
- Set up ongoing feedback mechanisms

---

## 🧠 Credits

- 🔗 Dataset: [RDD2022 Road Damage Dataset](https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022?resource=download)
- 🧠 Model: Fine-tuned YOLOv8 weights from [oracl4/RoadDamageDetection](https://github.com/oracl4/RoadDamageDetection)

---

## 📈 Potential Future Improvements

- Add real-time feedback integration for retraining
- Evaluate multiple models or ensembling strategies
- Integrate cloud deployment or mobile edge inference

---

## 🚀 How to Run

Make sure you have Python 3.9+ and `virtualenv` installed. Then follow these steps:

1. Clone the repository
git clone https://github.com/sigmagensolutions/road-damage-optimizer.git
cd road-damage-optimizer

2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\\Scripts\\activate

3. Install dependencies
pip install -r requirements.txt

4. Download the model weights
Place the YOLOv8_Small_RDD.pt file into the 'model/' directory

5. Download the RDD2022 dataset from Kaggle and extract it to 'data/images'
You should have image-label pairs in the appropriate structure

6. Run the full pipeline
python main.py

Note: you'll want to update the costs in the genetic optimizer.py file to suit your situation.

---

## 🔁 License

MIT License unless otherwise specified. (see credits links for their individual licenses)
