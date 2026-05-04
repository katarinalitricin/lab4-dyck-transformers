import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data import load_jsonl, VOCAB, MAX_LEN
from model import DyckTransformerClassifier
from train import get_device, save_checkpoint
from correction_data import CORRECTION_LABELS


class DyckCorrectionDataset(Dataset):
    """
    Dataset for token-level correction.

    Labels are aligned with tokenized input:
    [CLS] gets OK,
    bracket tokens get correction labels,
    [SEP] gets OK,
    [PAD] gets -100 and is ignored by CrossEntropyLoss.
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
            "labels": torch.tensor(ex["correction_label_ids"], dtype=torch.long),
            "error_type": ex["error_type"],
            "input_string": ex["input_string"],
            "clean": ex["clean"],
        }


class DyckTransformerCorrectionModel(nn.Module):
    """
    Token-level correction model.

    Reuses the Transformer encoder architecture from the detection model,
    but applies a linear classifier to every token representation.
    """

    def __init__(
        self,
        vocab_size: int = 7,
        max_len: int = 80,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_labels: int = 10,
    ):
        super().__init__()

        self.encoder_model = DyckTransformerClassifier(
            vocab_size=vocab_size,
            max_len=max_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_classes=2,
        )

        self.token_classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        _, hidden_states = self.encoder_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden=True,
        )

        logits = self.token_classifier(hidden_states)
        return logits


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Training correction"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100

        total_loss += loss.item() * mask.sum().item()
        total_correct += ((predictions == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / total_tokens
    token_accuracy = total_correct / total_tokens

    return avg_loss, token_accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    exact_match_correct = 0
    total_examples = 0

    for batch in tqdm(dataloader, desc="Evaluating correction"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        predictions = torch.argmax(logits, dim=-1)
        mask = labels != -100

        total_loss += loss.item() * mask.sum().item()
        total_correct += ((predictions == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

        for pred_row, label_row, mask_row in zip(predictions, labels, mask):
            pred_valid = pred_row[mask_row]
            label_valid = label_row[mask_row]

            if torch.equal(pred_valid.cpu(), label_valid.cpu()):
                exact_match_correct += 1

            total_examples += 1

    avg_loss = total_loss / total_tokens
    token_accuracy = total_correct / total_tokens
    exact_match_accuracy = exact_match_correct / total_examples

    return avg_loss, token_accuracy, exact_match_accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_path",
        type=str,
        default="data/processed/train_correction.jsonl",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="data/processed/dev_correction.jsonl",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/correction.pt",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="results/correction_train_metrics.json",
    )

    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_dev_examples", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_dataset = DyckCorrectionDataset(
        args.train_path,
        max_examples=args.max_train_examples,
    )
    dev_dataset = DyckCorrectionDataset(
        args.dev_path,
        max_examples=args.max_dev_examples,
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Dev examples: {len(dev_dataset)}")
    print(f"Correction labels: {CORRECTION_LABELS}")

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
        "num_labels": len(CORRECTION_LABELS),
    }

    model = DyckTransformerCorrectionModel(**model_config).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_dev_exact_match = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nCorrection epoch {epoch}/{args.epochs}")

        train_loss, train_token_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        dev_loss, dev_token_accuracy, dev_exact_match_accuracy = evaluate(
            model=model,
            dataloader=dev_loader,
            criterion=criterion,
            device=device,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_token_accuracy": train_token_accuracy,
            "dev_loss": dev_loss,
            "dev_token_accuracy": dev_token_accuracy,
            "dev_exact_match_accuracy": dev_exact_match_accuracy,
        }

        history.append(epoch_metrics)
        print(epoch_metrics)

        if dev_exact_match_accuracy > best_dev_exact_match:
            best_dev_exact_match = dev_exact_match_accuracy
            save_checkpoint(
                model=model,
                path=args.checkpoint_path,
                config=model_config,
            )
            print(f"Saved best correction checkpoint to {args.checkpoint_path}")

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved correction training metrics to {metrics_path}")


if __name__ == "__main__":
    main()