from PIL import Image, ImageDraw, ImageFont

def add_caption(img_path, caption, output_path, font_path="arial.ttf", font_size=32):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    w, h = img.size
    text_w, text_h = draw.textsize(caption, font=font)
    draw.text(((w - text_w) / 2, h - text_h - 10), caption, font=font, fill="white")
    img.save(output_path)
