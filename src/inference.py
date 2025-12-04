import os
import json
from template_selector import TemplateSelector
from caption_generator import CaptionGenerator
from overlay import add_caption

def run_inference(sentence,
                  template_dir="data/raw/templates",
                  output_dir="outputs/generated_memes",
                  model_dir="models"):
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Normalize paths
    template_selector_dir = os.path.abspath(os.path.join(model_dir, "template_selector"))
    caption_generator_dir = os.path.abspath(os.path.join(model_dir, "caption_generator"))

    # Step 1: Template Selector
    selector = TemplateSelector()
    selector.load(template_selector_dir)

    # Load label mapping
    with open(os.path.join(template_selector_dir, "labels.json")) as f:
        id2label = json.load(f)

    template_id = selector.predict(sentence)
    template_name = id2label[str(template_id)]
    template_path = os.path.join(template_dir, f"{template_name}.jpg")

    # Step 2: Caption Generator
    caption_model = CaptionGenerator()
    caption_model.load(caption_generator_dir)
    caption = caption_model.generate(sentence)

    # Step 3: Overlay caption on image
    output_path = os.path.join(output_dir, f"meme_{template_name}.jpg")
    add_caption(template_path, caption, output_path)
    return output_path
