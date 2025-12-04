# MemeBot 🤖😂
AI Meme Generator using the **ImgFlip575K Dataset**.

## 🔹 Features
- Template selection (BERT classifier)
- Caption generation (T5/BART)
- Text overlay (PIL)
- Optional GAN for template generation

## 🔹 Dataset
We use [ImgFlip575K](https://github.com/schesa/ImgFlip575K_Dataset).

## 🔹 Workflow
1. Preprocess dataset → `data/processed/train.jsonl`
2. Train template selector → `src/train_template.py`
3. Train caption generator → `src/train_caption.py`
4. Run inference → `src/inference.py`

## 🔹 Outputs
Generated memes are saved in:
