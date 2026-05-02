import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from data import load_jsonl, VOCAB, MAX_LEN
from model import DyckTransformerClassifier


class DyckDetectionDataset(Dataset):
    """
    Dataset for binary error detection.

    Label:
        0 = valid Dyck string
        1 = corrupted / invalid string
    """

    def __init__(self, path: str, max_examples: int | None = None):
        self.examples = load_jsonl(path)

        if max_examples is not None:
            self.examples = self.examples[:max_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "label": torch.tensor(ex["is_error"], dtype=torch.long),
            "error_type": ex["error_type"],
            "depth": ex["depth"],
            "length": ex["length"],
        }


def get_device() -> torch.device:
    """
    Choose the best available device.

    Priority:
        1. CUDA GPU
        2. Apple Silicon MPS GPU
        3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=-1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        predictions = torch.argmax(logits, dim=-1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return avg_loss, accuracy


def save_checkpoint(model, path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--dev_path", type=str, default="data/processed/dev.jsonl")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/detection.pt")
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="results/detection_train_metrics.json",
    )

    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_dev_examples", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_dataset = DyckDetectionDataset(
        args.train_path,
        max_examples=args.max_train_examples,
    )
    dev_dataset = DyckDetectionDataset(
        args.dev_path,
        max_examples=args.max_dev_examples,
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Dev examples: {len(dev_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_config = {
        "vocab_size": len(VOCAB),
        "max_len": MAX_LEN,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "num_classes": 2,
    }

    model = DyckTransformerClassifier(**model_config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_dev_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        dev_loss, dev_accuracy = evaluate(
            model=model,
            dataloader=dev_loader,
            criterion=criterion,
            device=device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_accuracy,
        }

        history.append(epoch_metrics)

        print(epoch_metrics)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            save_checkpoint(
                model=model,
                path=args.checkpoint_path,
                config=model_config,
            )
            print(f"Saved best checkpoint to {args.checkpoint_path}")

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()