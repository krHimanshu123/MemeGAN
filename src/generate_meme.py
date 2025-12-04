import os
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import json

# -------------------------------
# CONFIGURATION
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_FILE = os.path.join(BASE_DIR, "..", "data", "templates", "templates.jsonl")

MODEL_PATH = "trained_model"  # path where you save your TemplateSelector model
IMAGE_CACHE_DIR = "data/images"  # folder to store downloaded template images
FONT_PATH = "arial.ttf"  # Change to Impact.ttf if available
FONT_SIZE = 40


# -------------------------------
# LOAD DATASET
# -------------------------------

class MemeDataset:
    def __init__(self, jsonl_file):
        self.samples = []
        self.label2id = {}
        self.id2label = {}

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                if "template" in sample:
                    if sample["template"] not in self.label2id:
                        new_id = len(self.label2id)
                        self.label2id[sample["template"]] = new_id
                        self.id2label[new_id] = sample["template"]
                    sample["template_id"] = self.label2id[sample["template"]]
                self.samples.append(sample)


dataset = MemeDataset(JSONL_FILE)

# -------------------------------
# LOAD TEMPLATE SELECTOR
# -------------------------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(dataset.label2id)
)
model.eval()  # evaluation mode


# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------

def download_image(url, save_path):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image.save(save_path)
    return save_path


def overlay_text(image, top_text, bottom_text):
    draw = ImageDraw.Draw(image)

    # Try to load Impact font, fallback to Arial
    try:
        font = ImageFont.truetype("impact.ttf", FONT_SIZE)
    except:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Helper to draw outlined text
    def draw_outlined_text(text, position):
        x, y = position
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                draw.text((x + dx, y + dy), text, font=font, fill="black")
        draw.text((x, y), text, font=font, fill="white")

    # Uppercase for meme style
    top_text = top_text.upper()
    bottom_text = bottom_text.upper()

    # Wrap text so it doesn’t overflow
    def wrap_text(text, max_width):
        lines = []
        words = text.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if draw.textlength(test_line, font=font) <= max_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        lines.append(line)
        return lines

    # Top text
    top_lines = wrap_text(top_text, image.width - 20)
    y = 10
    for line in top_lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (image.width - w) / 2
        draw_outlined_text(line, (x, y))
        y += h

    # Bottom text
    bottom_lines = wrap_text(bottom_text, image.width - 20)
    y = image.height - FONT_SIZE * len(bottom_lines) - 10
    for line in bottom_lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (image.width - w) / 2
        draw_outlined_text(line, (x, y))
        y += h

    return image


# -------------------------------
# GENERATE MEME FUNCTION
# -------------------------------

def generate_meme(input_text):
    # Predict template
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    template_id = outputs.logits.argmax(dim=1).item()
    template_name = dataset.id2label[template_id]
    print(f"Predicted template: {template_name}")

    # Select a random template sample
    template_samples = [s for s in dataset.samples if s["template"] == template_name]
    sample = random.choice(template_samples)

    # Get fresh image (always re-download)
    if "template_url" in sample:
        if not os.path.exists(IMAGE_CACHE_DIR):
            os.makedirs(IMAGE_CACHE_DIR)
        file_name = os.path.basename(sample["template_url"])
        img_path = os.path.join(IMAGE_CACHE_DIR, file_name)
        download_image(sample["template_url"], img_path)
    else:
        raise ValueError("No valid image for this template.")

    image = Image.open(img_path)

    # Use input text as top/bottom captions
    if len(input_text) > 40:
        top_text = input_text[: len(input_text) // 2]
        bottom_text = input_text[len(input_text) // 2:]
    else:
        top_text = input_text
        bottom_text = ""

    # Overlay text
    meme_image = overlay_text(image, top_text, bottom_text)

    # Save meme
    output_path = "generated_meme.jpg"
    meme_image.save(output_path)
    meme_image.show()
    print(f"✅ Meme saved at {output_path}")


# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    input_text = input("Enter your meme text: ")
    generate_meme(input_text)
