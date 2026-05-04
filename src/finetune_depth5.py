import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import VOCAB, MAX_LEN
from model import DyckTransformerClassifier
from train import DyckDetectionDataset, train_one_epoch, evaluate, get_device, save_checkpoint


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_checkpoint_path",
        type=str,
        default="checkpoints/detection.pt",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/processed/ood_depth5.jsonl",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default="data/processed/dev.jsonl",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/detection_finetuned_depth5.pt",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="results/finetune_depth5_metrics.json",
    )

    parser.add_argument("--max_train_examples", type=int, default=500)
    parser.add_argument("--max_dev_examples", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    base_checkpoint = torch.load(args.base_checkpoint_path, map_location=device)
    model_config = base_checkpoint["config"]

    # Safety fallback in case older checkpoints lack some config fields.
    model_config.setdefault("vocab_size", len(VOCAB))
    model_config.setdefault("max_len", MAX_LEN)
    model_config.setdefault("num_classes", 2)

    model = DyckTransformerClassifier(**model_config).to(device)
    model.load_state_dict(base_checkpoint["model_state_dict"])

    train_dataset = DyckDetectionDataset(
        args.train_path,
        max_examples=args.max_train_examples,
    )
    dev_dataset = DyckDetectionDataset(
        args.dev_path,
        max_examples=args.max_dev_examples,
    )

    print(f"Fine-tuning examples: {len(train_dataset)}")
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

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_dev_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nFine-tuning epoch {epoch}/{args.epochs}")

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
            print(f"Saved best fine-tuned checkpoint to {args.checkpoint_path}")

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved fine-tuning metrics to {metrics_path}")


if __name__ == "__main__":
    main()