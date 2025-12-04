import os
import json
import requests
from tqdm import tqdm

# Absolute paths for consistency
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "memes")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "train.jsonl")
IMG_DIR = os.path.join(BASE_DIR, "data", "images")

print("Current working directory:", os.getcwd())
print("Script base directory:", BASE_DIR)
print("RAW_DIR absolute path:", RAW_DIR)

# make sure processed folders exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

with open(PROCESSED_FILE, "w", encoding="utf-8") as outfile:
    for file in os.listdir(RAW_DIR):
        if file.endswith(".json"):
            filepath = os.path.join(RAW_DIR, file)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    memes = json.load(f)  # LIST of memes per template
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
                    continue

                for meme in tqdm(memes, desc=f"Processing {file}"):
                    image_url = meme.get("url")
                    template_name = file.replace(".json", "")

                    # Local filename for image
                    filename = f"{template_name}_{os.path.basename(image_url)}"
                    local_path = os.path.join(IMG_DIR, filename)

                    # Download image if not already saved
                    if image_url and not os.path.exists(local_path):
                        try:
                            r = requests.get(image_url, timeout=10)
                            if r.status_code == 200:
                                with open(local_path, "wb") as img_f:
                                    img_f.write(r.content)
                            else:
                                print(f"⚠️ Failed to download {image_url} - status code {r.status_code}")
                        except Exception as e:
                            print(f"⚠️ Skipping image {image_url}: {e}")
                            continue

                    # Build record
                    data = {
                        "template": template_name,
                        "image_url": image_url,
                        "img_path": local_path if os.path.exists(local_path) else None,
                        "post_url": meme.get("post"),
                        "views": meme.get("metadata", {}).get("views"),
                        "votes": meme.get("metadata", {}).get("img-votes"),
                        "title": meme.get("metadata", {}).get("title"),
                        "author": meme.get("metadata", {}).get("author"),
                        "captions": meme.get("boxes", []),
                    }

                    # For caption task: flatten into sentence/caption
                    if data["captions"]:
                        data["sentence"] = " ".join(data["captions"])  # input context
                        data["caption"] = data["captions"][0]          # target caption

                    outfile.write(json.dumps(data) + "\n")

print(f"✅ Preprocessing finished! Saved to {PROCESSED_FILE}")
