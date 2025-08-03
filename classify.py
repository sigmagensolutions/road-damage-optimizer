from ultralytics import YOLO
import os
import pandas as pd
import numpy as np

# Load fine-tuned road damage model
model = YOLO("model/YOLOv8_Small_RDD.pt")
image_dir = "data/images"
results = []

print("📁 Class labels:", model.names)

image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
total_images = len(image_files)
print(f"🖼️ Found {total_images} images in '{image_dir}'")

for idx, img_file in enumerate(image_files, 1):
    path = os.path.join(image_dir, img_file)
    print(f"🔍 Processing image {idx}/{total_images}: {img_file}")

    output = model(path, verbose=False)[0]

    # Collect class confidence scores
    class_scores = [0.0] * 4  # D00–D40
    class_counts = [0] * 4

    for box in output.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if 0 <= cls_id < 4:
            class_scores[cls_id] += conf
            class_counts[cls_id] += 1

    # Normalize scores by counts (mean confidence per class)
    avg_scores = [
        (class_scores[i] / class_counts[i]) if class_counts[i] > 0 else 0.0
        for i in range(4)
    ]

    results.append({
        "image": img_file,
        "predicted_class": int(np.argmax(avg_scores)),
        "D00_prob": avg_scores[0],
        "D10_prob": avg_scores[1],
        "D20_prob": avg_scores[2],
        "D40_prob": avg_scores[3],
    })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("inference_results_with_probs.csv", index=False)
print("✅ Saved inference results to 'inference_results_with_probs.csv'")
