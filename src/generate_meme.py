import os
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "templates")
JSONL_FILE = os.path.join(DATA_DIR, "templates.jsonl")

IMAGE_CACHE_DIR = os.path.join(BASE_DIR, "..", "data", "images")
FONT_PATH = os.path.join(BASE_DIR, "arial.ttf")
FONT_SIZE = 40

# -------------- Load model ONCE (critical for Render) --------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=50  # put your real number
)
model.eval()


# ------------------ Dataset loader ------------------

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
                label = sample["template"]

                if label not in self.label2id:
                    new_id = len(self.label2id)
                    self.label2id[label] = new_id
                    self.id2label[new_id] = label

                sample["template_id"] = self.label2id[label]
                self.samples.append(sample)


dataset = MemeDataset(JSONL_FILE)


# ------------------ Helpers ------------------

def download_image(url, save_path):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image.save(save_path)
    return save_path


def overlay_text(image, top_text, bottom_text):
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("impact.ttf", FONT_SIZE)
    except:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    top_text = top_text.upper()
    bottom_text = bottom_text.upper()

    def draw_outline(text, xy):
        x, y = xy
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                draw.text((x+dx, y+dy), text, fill="black", font=font)
        draw.text((x, y), text, fill="white", font=font)

    def wrap(text):
        words = text.split()
        lines = []
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if draw.textlength(test, font=font) <= image.width - 20:
                line = test
            else:
                lines.append(line)
                line = w
        lines.append(line)
        return lines

    # Top
    y = 10
    for line in wrap(top_text):
        w, h = draw.textlength(line, font=font), FONT_SIZE
        x = (image.width - w) / 2
        draw_outline(line, (x, y))
        y += h

    # Bottom
    y = image.height - FONT_SIZE * len(wrap(bottom_text)) - 10
    for line in wrap(bottom_text):
        w, h = draw.textlength(line, font=font), FONT_SIZE
        x = (image.width - w) / 2
        draw_outline(line, (x, y))
        y += h

    return image


# ------------------ MAIN API FUNCTION ------------------

def generate_meme(input_text, output_path):

    # 1. Predict template
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    template_id = outputs.logits.argmax(1).item()
    template_name = dataset.id2label[template_id]

    # 2. Pick a sample
    sample = next(s for s in dataset.samples if s["template"] == template_name)

    # 3. Download template image
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    file_name = os.path.basename(sample["template_url"])
    img_path = os.path.join(IMAGE_CACHE_DIR, file_name)
    download_image(sample["template_url"], img_path)

    image = Image.open(img_path)

    # 4. Split text
    if len(input_text) > 40:
        mid = len(input_text) // 2
        top = input_text[:mid]
        bottom = input_text[mid:]
    else:
        top = input_text
        bottom = ""

    # 5. Draw text
    meme = overlay_text(image, top, bottom)

    # 6. Save to API output path
    meme.save(output_path)
    return output_path
