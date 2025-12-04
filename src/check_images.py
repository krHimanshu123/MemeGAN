import os
import json

# Path to your original train.jsonl
jsonl_file = "data/processed/train.jsonl"

# Counters
total_lines = 0
valid_samples = 0
missing_images = 0

with open(jsonl_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        total_lines += 1
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            img_path = sample.get("img_path")

            if not img_path:
                print(f"⚠️ Line {i}: img_path is None")
                missing_images += 1
                continue

            # Normalize path for Windows
            img_path = os.path.normpath(img_path)
            sample["img_path"] = img_path

            if not os.path.exists(img_path):
                print(f"⚠️ Line {i}: Image file not found at {img_path}")
                missing_images += 1
            else:
                valid_samples += 1

        except Exception as e:
            print(f"❌ Line {i}: JSON parse error: {e}")

print("\n===== Summary =====")
print(f"Total lines in file: {total_lines}")
print(f"Valid samples with existing images: {valid_samples}")
print(f"Samples with missing or invalid images: {missing_images}")
