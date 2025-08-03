import os
import pandas as pd

LABEL_DIR = "data/labels"
OUTPUT_CSV = "ground_truth_flags.csv"

class_names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

records = []

for label_file in os.listdir(LABEL_DIR):
    if not label_file.endswith(".txt"):
        continue

    filepath = os.path.join(LABEL_DIR, label_file)
    image_name = os.path.splitext(label_file)[0] + ".jpg"

    class_flags = {cls: 0 for cls in class_names.values()}

    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            class_name = class_names.get(cls_id)
            if class_name:
                class_flags[class_name] = 1

    record = {"image": image_name}
    record.update(class_flags)
    records.append(record)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved ground truth flags to {OUTPUT_CSV}")
print(df.head())
