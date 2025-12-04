import os
import json

# Path to your original train.jsonl
jsonl_file = "data/processed/train.jsonl"

with open("missing_images_report.txt", "w", encoding="utf-8") as report:
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
                if not sample.get("img_path") or not os.path.exists(sample.get("img_path", "")):
                    report.write(f"Line {i}: template={sample.get('template')}, img_path={sample.get('img_path')}\n")
            except:
                continue

print("✅ Missing images report generated: missing_images_report.txt")
