import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

class TemplateSelector:
    def __init__(self, model_name="bert-base-uncased", num_labels=100):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def train(self, train_dataloader, val_dataloader=None, epochs=5, lr=5e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for sentences, labels in train_dataloader:
                inputs = self.tokenizer(list(sentences), return_tensors="pt",
                                        padding=True, truncation=True)
                labels = labels.detach().clone() if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long)
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f"📚 Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    def predict(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt")
        outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()

    def save(self, path):
        path = os.path.abspath(path)  # ✅ normalize path
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        path = os.path.abspath(path)  # ✅ normalize path
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
