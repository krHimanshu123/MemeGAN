import torch
from torch.utils.data import DataLoader, Dataset
from data_utils import MemeDataset
from caption_generator import CaptionGenerator


# ✅ Filters out any dataset entries that are None
class FilteredDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.valid_indices = [i for i in range(len(dataset)) if dataset[i] is not None]

        if not self.valid_indices:
            raise ValueError("❌ No valid samples found in the dataset.")

        print(f"✅ Filtered dataset size: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.dataset[self.valid_indices[idx]]


# ✅ Collate function to merge dictionary batches into tensors
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    input_ids = torch.cat([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.cat([b["attention_mask"] for b in batch], dim=0)
    labels = torch.cat([b["labels"] for b in batch], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    print("🚀 Initializing Caption Generator...")
    generator = CaptionGenerator()

    print("📦 Loading dataset...")
    raw_dataset = MemeDataset("data/processed/train.jsonl", task="caption", tokenizer=generator.tokenizer)

    print("🧹 Filtering valid samples...")
    dataset = FilteredDataset(raw_dataset)

    print("📤 Preparing DataLoader...")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    print("🎯 Starting training...")
    generator.train(dataloader)


if __name__ == "__main__":
    main()
