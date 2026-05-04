import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data import load_jsonl, VOCAB, MAX_LEN
from model import DyckTransformerClassifier
from train import get_device


OPEN_TO_CLOSE = {"(": ")", "[": "]"}
CLOSE_TO_OPEN = {")": "(", "]": "["}

GLOBAL_COLOR = "#4C1D95"   # purple
LOCAL_COLOR = "#14B8A6"    # teal
ERROR_COLOR = "#F97316"    # orange
GRID_COLOR = "#E5E7EB"


def local_depths(s: str) -> list[int]:
    """
    Compute local stack depth at each token position.

    For an opening bracket, depth is recorded after pushing.
    For a closing bracket, depth is recorded before popping.
    """
    stack = []
    depths = []

    for ch in s:
        if ch in OPEN_TO_CLOSE:
            stack.append(ch)
            depths.append(len(stack))
        elif ch in CLOSE_TO_OPEN:
            depths.append(len(stack))
            if stack:
                stack.pop()
        else:
            depths.append(0)

    return depths


class ProbeDataset(Dataset):
    """
    Dataset for probing hidden representations from the trained detector.

    For global probing:
        [CLS] hidden state -> maximum nesting depth

    For local probing:
        token hidden state -> local depth at each bracket token
    """

    def __init__(
        self,
        path: str,
        model: DyckTransformerClassifier,
        device: torch.device,
        max_examples: int | None = None,
    ):
        self.examples = load_jsonl(path)

        if max_examples is not None:
            self.examples = self.examples[:max_examples]

        self.model = model
        self.device = device

    def __len__(self):
        return len(self.examples)

    @torch.no_grad()
    def __getitem__(self, idx):
        ex = self.examples[idx]

        input_ids = torch.tensor([ex["input_ids"]], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([ex["attention_mask"]], dtype=torch.long).to(
            self.device
        )

        _, hidden = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden=True,
        )

        hidden = hidden.squeeze(0).cpu()  # [seq_len, hidden_dim]

        input_string = ex["input_string"]
        seq_len = sum(ex["attention_mask"])

        cls_rep = hidden[0]

        # Maximum depth from metadata. Depth is based on the original clean string.
        global_depth = int(ex["depth"])

        # Local depth labels aligned to tokenized sequence.
        # [CLS] and [SEP] are ignored with -100.
        raw_depths = local_depths(input_string)
        local_labels = [-100] + raw_depths + [-100]

        while len(local_labels) < MAX_LEN:
            local_labels.append(-100)

        local_labels = torch.tensor(local_labels[:MAX_LEN], dtype=torch.long)

        return {
            "cls_rep": cls_rep,
            "hidden": hidden,
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "global_depth": torch.tensor(global_depth, dtype=torch.long),
            "local_depths": local_labels,
            "error_type": ex["error_type"],
        }


class GlobalDepthProbe(nn.Module):
    def __init__(self, hidden_dim: int, num_depth_classes: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_depth_classes)

    def forward(self, cls_rep):
        return self.classifier(cls_rep)


class LocalDepthProbe(nn.Module):
    def __init__(self, hidden_dim: int, num_depth_classes: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_depth_classes)

    def forward(self, hidden):
        return self.classifier(hidden)


def collate_probe_batch(batch):
    return {
        "cls_rep": torch.stack([item["cls_rep"] for item in batch]),
        "hidden": torch.stack([item["hidden"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "global_depth": torch.stack([item["global_depth"] for item in batch]),
        "local_depths": torch.stack([item["local_depths"] for item in batch]),
        "error_type": [item["error_type"] for item in batch],
    }


def train_global_probe(probe, train_loader, dev_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(probe.parameters(), lr=lr)

    best_dev_acc = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        probe.train()

        total_correct = 0
        total = 0
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Global probe train {epoch}/{epochs}"):
            cls_rep = batch["cls_rep"].to(device)
            labels = batch["global_depth"].to(device)

            optimizer.zero_grad()
            logits = probe(cls_rep)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = total_correct / total
        train_loss = total_loss / total

        dev_metrics = evaluate_global_probe(probe, dev_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_accuracy_by_error_type": dev_metrics["accuracy_by_error_type"],
        }
        history.append(row)

        print(row)

        best_dev_acc = max(best_dev_acc, dev_metrics["accuracy"])

    return history


@torch.no_grad()
def evaluate_global_probe(probe, dataloader, device):
    probe.eval()

    total_correct = 0
    total = 0

    correct_by_type = defaultdict(int)
    total_by_type = defaultdict(int)

    for batch in dataloader:
        cls_rep = batch["cls_rep"].to(device)
        labels = batch["global_depth"].to(device)
        error_types = batch["error_type"]

        logits = probe(cls_rep)
        preds = torch.argmax(logits, dim=-1)

        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

        for pred, label, error_type in zip(preds.cpu(), labels.cpu(), error_types):
            total_by_type[error_type] += 1
            if int(pred) == int(label):
                correct_by_type[error_type] += 1

    return {
        "accuracy": total_correct / total,
        "accuracy_by_error_type": {
            error_type: correct_by_type[error_type] / total_by_type[error_type]
            for error_type in sorted(total_by_type)
        },
    }


def train_local_probe(probe, train_loader, dev_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(probe.parameters(), lr=lr)

    history = []

    for epoch in range(1, epochs + 1):
        probe.train()

        total_correct = 0
        total_tokens = 0
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Local probe train {epoch}/{epochs}"):
            hidden = batch["hidden"].to(device)
            labels = batch["local_depths"].to(device)

            optimizer.zero_grad()

            logits = probe(hidden)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100

            total_loss += loss.item() * mask.sum().item()
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

        train_acc = total_correct / total_tokens
        train_loss = total_loss / total_tokens

        dev_metrics = evaluate_local_probe(probe, dev_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_token_accuracy": train_acc,
            "dev_token_accuracy": dev_metrics["token_accuracy"],
            "dev_token_accuracy_by_error_type": dev_metrics[
                "token_accuracy_by_error_type"
            ],
        }
        history.append(row)

        print(row)

    return history


@torch.no_grad()
def evaluate_local_probe(probe, dataloader, device):
    probe.eval()

    total_correct = 0
    total_tokens = 0

    correct_tokens_by_type = defaultdict(int)
    total_tokens_by_type = defaultdict(int)

    for batch in dataloader:
        hidden = batch["hidden"].to(device)
        labels = batch["local_depths"].to(device)
        error_types = batch["error_type"]

        logits = probe(hidden)
        preds = torch.argmax(logits, dim=-1)

        mask = labels != -100

        total_correct += ((preds == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

        for pred_row, label_row, mask_row, error_type in zip(
            preds.cpu(), labels.cpu(), mask.cpu(), error_types
        ):
            correct = ((pred_row == label_row) & mask_row).sum().item()
            count = mask_row.sum().item()

            correct_tokens_by_type[error_type] += correct
            total_tokens_by_type[error_type] += count

    return {
        "token_accuracy": total_correct / total_tokens,
        "token_accuracy_by_error_type": {
            error_type: correct_tokens_by_type[error_type]
            / total_tokens_by_type[error_type]
            for error_type in sorted(total_tokens_by_type)
        },
    }


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.45)
    ax.set_axisbelow(True)


def plot_probe_summary(metrics, output_dir):
    labels = ["Global depth\nprobe", "Local depth\nprobe"]
    values = [
        metrics["global_test"]["accuracy"],
        metrics["local_test"]["token_accuracy"],
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=[GLOBAL_COLOR, LOCAL_COLOR], width=0.55)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Depth information in Transformer representations", pad=14)
    style_axis(ax)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
        )

    fig.tight_layout()
    path = output_dir / "q13_q14_probe_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def plot_probe_by_error_type(metrics, output_dir):
    error_types = ["NONE", "E1", "E2", "E3", "E4"]

    global_values = [
        metrics["global_test"]["accuracy_by_error_type"][error_type]
        for error_type in error_types
    ]
    local_values = [
        metrics["local_test"]["token_accuracy_by_error_type"][error_type]
        for error_type in error_types
    ]

    labels = [
        "NONE",
        "E1\nmissing\ncloser",
        "E2\nspurious\nopener",
        "E3\ntype\nmismatch",
        "E4\npremature\nclose",
    ]

    x = range(len(error_types))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    global_bars = ax.bar(
        [i - width / 2 for i in x],
        global_values,
        width,
        label="Global depth probe",
        color=GLOBAL_COLOR,
    )
    local_bars = ax.bar(
        [i + width / 2 for i in x],
        local_values,
        width,
        label="Local depth probe",
        color=LOCAL_COLOR,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Depth probe accuracy by error type", pad=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    style_axis(ax)

    for bars in [global_bars, local_bars]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.2f}",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    fig.tight_layout()
    path = output_dir / "q15_probe_accuracy_by_error_type.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/detection.pt")
    parser.add_argument("--train_path", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--dev_path", type=str, default="data/processed/dev.jsonl")
    parser.add_argument("--test_path", type=str, default="data/processed/test.jsonl")

    parser.add_argument("--max_train_examples", type=int, default=10000)
    parser.add_argument("--max_dev_examples", type=int, default=1000)
    parser.add_argument("--max_test_examples", type=int, default=5000)

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--output_path", type=str, default="results/probing_metrics.json")
    parser.add_argument("--output_dir", type=str, default="report/figures")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = DyckTransformerClassifier(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = config["hidden_dim"]

    # Depth labels are 1–4 for train/dev/test. Class 0 is unused but harmless.
    num_depth_classes = 8

    print("Extracting train representations...")
    train_dataset = ProbeDataset(
        args.train_path,
        model=model,
        device=device,
        max_examples=args.max_train_examples,
    )

    print("Extracting dev representations...")
    dev_dataset = ProbeDataset(
        args.dev_path,
        model=model,
        device=device,
        max_examples=args.max_dev_examples,
    )

    print("Extracting test representations...")
    test_dataset = ProbeDataset(
        args.test_path,
        model=model,
        device=device,
        max_examples=args.max_test_examples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_probe_batch,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_probe_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_probe_batch,
    )

    global_probe = GlobalDepthProbe(
        hidden_dim=hidden_dim,
        num_depth_classes=num_depth_classes,
    ).to(device)

    local_probe = LocalDepthProbe(
        hidden_dim=hidden_dim,
        num_depth_classes=num_depth_classes,
    ).to(device)

    print("\nTraining global depth probe...")
    global_history = train_global_probe(
        probe=global_probe,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
    )

    print("\nTraining local depth probe...")
    local_history = train_local_probe(
        probe=local_probe,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
    )

    print("\nEvaluating probes on test set...")
    global_test = evaluate_global_probe(global_probe, test_loader, device)
    local_test = evaluate_local_probe(local_probe, test_loader, device)

    metrics = {
        "global_history": global_history,
        "local_history": local_history,
        "global_test": global_test,
        "local_test": local_test,
        "num_train_examples": len(train_dataset),
        "num_dev_examples": len(dev_dataset),
        "num_test_examples": len(test_dataset),
    }

    print(json.dumps(metrics, indent=2))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved probing metrics to {output_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_probe_summary(metrics, output_dir)
    plot_probe_by_error_type(metrics, output_dir)


if __name__ == "__main__":
    main()