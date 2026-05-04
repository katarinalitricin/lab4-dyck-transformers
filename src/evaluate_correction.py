import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train import get_device
from train_correction import DyckCorrectionDataset, DyckTransformerCorrectionModel
from correction_data import CORRECTION_LABELS


@torch.no_grad()
def evaluate_correction(model, dataloader, device):
    model.eval()

    total_correct = 0
    total_tokens = 0

    exact_match_correct = 0
    total_examples = 0

    correct_by_error_type = defaultdict(int)
    total_by_error_type = defaultdict(int)

    correct_tokens_by_error_type = defaultdict(int)
    total_tokens_by_error_type = defaultdict(int)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1)

        mask = labels != -100

        total_correct += ((predictions == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

        error_types = batch["error_type"]

        for pred_row, label_row, mask_row, error_type in zip(
            predictions, labels, mask, error_types
        ):
            pred_valid = pred_row[mask_row]
            label_valid = label_row[mask_row]

            example_correct = torch.equal(pred_valid.cpu(), label_valid.cpu())

            total_examples += 1
            total_by_error_type[error_type] += 1

            if example_correct:
                exact_match_correct += 1
                correct_by_error_type[error_type] += 1

            correct_tokens = (pred_valid == label_valid).sum().item()
            total_example_tokens = mask_row.sum().item()

            correct_tokens_by_error_type[error_type] += correct_tokens
            total_tokens_by_error_type[error_type] += total_example_tokens

    token_accuracy = total_correct / total_tokens
    exact_match_accuracy = exact_match_correct / total_examples

    exact_match_by_error_type = {
        error_type: correct_by_error_type[error_type] / total_by_error_type[error_type]
        for error_type in sorted(total_by_error_type)
    }

    token_accuracy_by_error_type = {
        error_type: correct_tokens_by_error_type[error_type]
        / total_tokens_by_error_type[error_type]
        for error_type in sorted(total_tokens_by_error_type)
    }

    return {
        "num_examples": total_examples,
        "token_accuracy": token_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "exact_match_by_error_type": exact_match_by_error_type,
        "token_accuracy_by_error_type": token_accuracy_by_error_type,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/correction.pt",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/test_correction.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/correction_test_metrics.json",
    )
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = DyckTransformerCorrectionModel(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = DyckCorrectionDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate_correction(model, dataloader, device)

    metrics["checkpoint_path"] = args.checkpoint_path
    metrics["data_path"] = args.data_path
    metrics["correction_labels"] = CORRECTION_LABELS

    print(json.dumps(metrics, indent=2))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved correction metrics to {output_path}")


if __name__ == "__main__":
    main()