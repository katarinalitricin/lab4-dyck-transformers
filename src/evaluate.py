import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

from data import VOCAB, MAX_LEN
from train import DyckDetectionDataset, get_device
from model import DyckTransformerClassifier


@torch.no_grad()
def predict(model, dataloader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_error_types = []
    all_depths = []
    all_lengths = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1).cpu()

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())
        all_error_types.extend(batch["error_type"])
        all_depths.extend(batch["depth"])
        all_lengths.extend(batch["length"])

    return {
        "labels": all_labels,
        "predictions": all_predictions,
        "error_types": all_error_types,
        "depths": all_depths,
        "lengths": all_lengths,
    }


def grouped_accuracy(labels, predictions, groups):
    correct_by_group = defaultdict(int)
    total_by_group = defaultdict(int)

    for gold, pred, group in zip(labels, predictions, groups):
        total_by_group[group] += 1
        if gold == pred:
            correct_by_group[group] += 1

    return {
        str(group): correct_by_group[group] / total_by_group[group]
        for group in sorted(total_by_group)
    }


def length_bucket(length: int) -> str:
    if length <= 20:
        return "0-20"
    if length <= 40:
        return "21-40"
    if length <= 60:
        return "41-60"
    return "61-80"


def evaluate_checkpoint(
    checkpoint_path: str,
    data_path: str,
    batch_size: int,
    output_path: str | None = None,
):
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = DyckTransformerClassifier(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = DyckDetectionDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    outputs = predict(model, dataloader, device)

    labels = outputs["labels"]
    predictions = outputs["predictions"]
    error_types = outputs["error_types"]
    depths = outputs["depths"]
    lengths = outputs["lengths"]

    metrics = {
        "data_path": data_path,
        "checkpoint_path": checkpoint_path,
        "num_examples": len(labels),
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            target_names=["valid", "error"],
            output_dict=True,
        ),
        "accuracy_by_error_type": grouped_accuracy(
            labels,
            predictions,
            error_types,
        ),
        "accuracy_by_depth": grouped_accuracy(
            labels,
            predictions,
            depths,
        ),
        "accuracy_by_length_bucket": grouped_accuracy(
            labels,
            predictions,
            [length_bucket(length) for length in lengths],
        ),
    }

    print(json.dumps(metrics, indent=2))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/detection.pt")
    parser.add_argument("--data_path", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()