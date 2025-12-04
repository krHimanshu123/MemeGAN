import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class MemeDataset(Dataset):
    def __init__(self, jsonl_file, transform=None, tokenizer=None, task="template"):
        self.samples = []
        self.label2id = {}
        self.id2label = {}

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)

                    # Skip if img_path is missing or None
                    img_path = sample.get("img_path")
                    if not img_path:
                        continue

                    # Normalize Windows paths
                    img_path = os.path.normpath(img_path)
                    sample["img_path"] = img_path

                    if not os.path.exists(img_path):
                        continue  # skip if image file does not exist

                    # Create template label mapping
                    if "template" in sample:
                        if sample["template"] not in self.label2id:
                            new_id = len(self.label2id)
                            self.label2id[sample["template"]] = new_id
                            self.id2label[new_id] = sample["template"]
                        sample["template_id"] = self.label2id[sample["template"]]

                    self.samples.append(sample)

                except Exception as e:
                    print(f"❌ JSON parse error on line {i}: {e}")

        print(f"✅ Loaded {len(self.samples)} valid samples from {jsonl_file}")
        print(f"📌 Found {len(self.label2id)} unique templates")

        self.transform = transform
        self.tokenizer = tokenizer
        self.task = task  # "template" or "caption"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.task == "template":
            return sample["sentence"], sample["template_id"]

        elif self.task == "caption":
            try:
                inputs = self.tokenizer(
                    sample["sentence"], return_tensors="pt",
                    truncation=True, padding="max_length", max_length=64
                )
                labels = self.tokenizer(
                    sample["caption"], return_tensors="pt",
                    truncation=True, padding="max_length", max_length=32
                ).input_ids
                return {**inputs, "labels": labels}
            except Exception as e:
                print(f"❌ Tokenizer error at index {idx}: {e}")
                return None

        return sample
