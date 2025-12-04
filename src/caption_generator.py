import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

class CaptionGenerator:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def train(self, dataloader, epochs=3, lr=3e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            print(f"🧪 Epoch {epoch + 1}/{epochs}")
            for step, batch in enumerate(dataloader):
                if batch is None:
                    continue
                outputs = self.model(
                    input_ids=batch["input_ids"].view(batch["input_ids"].size(0), -1),
                    attention_mask=batch["attention_mask"].view(batch["attention_mask"].size(0), -1),
                    labels=batch["labels"].view(batch["labels"].size(0), -1),
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if step % 10 == 0:
                    print(f"   🔁 Step {step}: Loss = {loss.item():.4f}")

    def generate(self, sentence, max_length=32):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save(self, path):
        path = os.path.abspath(path)  # ✅ normalize path
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        path = os.path.abspath(path)  # ✅ normalize path
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
