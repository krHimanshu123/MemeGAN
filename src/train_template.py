import os
import json
from torch.utils.data import DataLoader
from data_utils import MemeDataset
from template_selector import TemplateSelector
from caption_generator import CaptionGenerator

def main():
    os.makedirs("models", exist_ok=True)

    # --- Template training ---
    print("🎯 Training Template Selector...")
    template_dataset = MemeDataset("data/processed/train.jsonl", task="template")

    template_loader = DataLoader(template_dataset, batch_size=16, shuffle=True)

    selector = TemplateSelector(num_labels=len(template_dataset.label2id))
    selector.train(template_loader)

    # Save selector + label mapping
    selector_dir = os.path.join("models", "template_selector")
    os.makedirs(selector_dir, exist_ok=True)
    selector.save(selector_dir)

    with open(os.path.join(selector_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(template_dataset.id2label, f, indent=2)

    # --- Caption training ---
    print("📝 Training Caption Generator...")
    captioner = CaptionGenerator()
    caption_dataset = MemeDataset("data/processed/train.jsonl", task="caption", tokenizer=captioner.tokenizer)
    caption_loader = DataLoader(caption_dataset, batch_size=8, shuffle=True)

    captioner.train(caption_loader)

    # Save captioner
    caption_dir = os.path.join("models", "caption_generator")
    os.makedirs(caption_dir, exist_ok=True)
    captioner.save(caption_dir)

    print("✅ Training complete! Models saved in /models/")

if __name__ == "__main__":
    main()
