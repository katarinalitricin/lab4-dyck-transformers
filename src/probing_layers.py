import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import load_jsonl, VOCAB, MAX_LEN
from model import DyckTransformerClassifier
from train import get_device


OPEN_TO_CLOSE = {"(": ")", "[": "]"}
CLOSE_TO_OPEN = {")": "(", "]": "["}

GLOBAL_COLOR = "#4C1D95"   # deep purple
LOCAL_COLOR = "#14B8A6"    # teal
GRID_COLOR = "#E5E7EB"


def local_depths(s: str) -> list[int]:
    """
    Compute approximate local stack depth for each token.

    For an opener: depth after pushing.
    For a closer: depth before popping.
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


@torch.no_grad()
def get_layer_representations(model, input_ids, attention_mask):
    """
    Manually run the Transformer and return representations from each stage.

    Returned list:
        layer 0 = embedding + position representation
        layer 1 = after encoder layer 1
        layer 2 = after encoder layer 2
        ...
    """
    x = model.token_embedding(input_ids)
    x = model.position_encoding(x)
    x = model.dropout(x)

    reps = [x.detach().cpu()]

    src_key_padding_mask = attention_mask == 0

    for layer in model.encoder.layers:
        if layer.norm_first:
            normed = layer.norm1(x)

            attn_output, _ = layer.self_attn(
                normed,
                normed,
                normed,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )

            x = x + layer.dropout1(attn_output)

            normed = layer.norm2(x)
            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(normed)))
            )
            x = x + layer.dropout2(ff_output)

        else:
            attn_output, _ = layer.self_attn(
                x,
                x,
                x,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )

            x = layer.norm1(x + layer.dropout1(attn_output))

            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x)))
            )
            x = layer.norm2(x + layer.dropout2(ff_output))

        reps.append(x.detach().cpu())

    if model.encoder.norm is not None:
        x = model.encoder.norm(x)
        reps[-1] = x.detach().cpu()

    return reps


class LayerProbeDataset(Dataset):
    """
    Precomputes frozen representations from all Transformer layers.

    This makes layer-wise probing faster and more stable.
    """

    def __init__(
        self,
        path: str,
        model: DyckTransformerClassifier,
        device: torch.device,
        max_examples: int | None = None,
    ):
        examples = load_jsonl(path)

        if max_examples is not None:
            examples = examples[:max_examples]

        self.items = []
        self.num_layers = None

        model.eval()

        print(f"Precomputing representations from {path}...")

        for ex in tqdm(examples):
            input_ids = torch.tensor([ex["input_ids"]], dtype=torch.long).to(device)
            attention_mask = torch.tensor([ex["attention_mask"]], dtype=torch.long).to(
                device
            )

            reps = get_layer_representations(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            reps = [rep.squeeze(0) for rep in reps]  # each [seq_len, hidden_dim]

            if self.num_layers is None:
                self.num_layers = len(reps)

            input_string = ex["input_string"]

            global_depth = int(ex["depth"])

            raw_local_depths = local_depths(input_string)
            local_labels = [-100] + raw_local_depths + [-100]

            while len(local_labels) < MAX_LEN:
                local_labels.append(-100)

            local_labels = torch.tensor(local_labels[:MAX_LEN], dtype=torch.long)

            self.items.append(
                {
                    "reps": reps,
                    "global_depth": torch.tensor(global_depth, dtype=torch.long),
                    "local_depths": local_labels,
                    "attention_mask": torch.tensor(
                        ex["attention_mask"], dtype=torch.long
                    ),
                    "error_type": ex["error_type"],
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_layer_probe_batch(batch):
    num_layers = len(batch[0]["reps"])

    reps_by_layer = []
    for layer_idx in range(num_layers):
        reps_by_layer.append(torch.stack([item["reps"][layer_idx] for item in batch]))

    return {
        "reps_by_layer": reps_by_layer,
        "global_depth": torch.stack([item["global_depth"] for item in batch]),
        "local_depths": torch.stack([item["local_depths"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "error_type": [item["error_type"] for item in batch],
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


def train_global_probe_for_layer(
    layer_idx,
    hidden_dim,
    num_depth_classes,
    train_loader,
    dev_loader,
    device,
    epochs,
    lr,
):
    probe = GlobalDepthProbe(hidden_dim, num_depth_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(probe.parameters(), lr=lr)

    best_dev_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        probe.train()

        total_correct = 0
        total = 0

        for batch in tqdm(
            train_loader,
            desc=f"Global probe layer {layer_idx} epoch {epoch}/{epochs}",
        ):
            reps = batch["reps_by_layer"][layer_idx].to(device)
            cls_rep = reps[:, 0, :]
            labels = batch["global_depth"].to(device)

            optimizer.zero_grad()
            logits = probe(cls_rep)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = total_correct / total
        dev_metrics = evaluate_global_probe_for_layer(
            probe=probe,
            layer_idx=layer_idx,
            dataloader=dev_loader,
            device=device,
        )

        best_dev_accuracy = max(best_dev_accuracy, dev_metrics["accuracy"])

        print(
            {
                "probe": "global",
                "layer": layer_idx,
                "epoch": epoch,
                "train_accuracy": train_accuracy,
                "dev_accuracy": dev_metrics["accuracy"],
            }
        )

    return probe, best_dev_accuracy


@torch.no_grad()
def evaluate_global_probe_for_layer(probe, layer_idx, dataloader, device):
    probe.eval()

    total_correct = 0
    total = 0

    correct_by_error_type = defaultdict(int)
    total_by_error_type = defaultdict(int)

    for batch in dataloader:
        reps = batch["reps_by_layer"][layer_idx].to(device)
        cls_rep = reps[:, 0, :]
        labels = batch["global_depth"].to(device)
        error_types = batch["error_type"]

        logits = probe(cls_rep)
        preds = torch.argmax(logits, dim=-1)

        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

        for pred, label, error_type in zip(preds.cpu(), labels.cpu(), error_types):
            total_by_error_type[error_type] += 1
            if int(pred) == int(label):
                correct_by_error_type[error_type] += 1

    return {
        "accuracy": total_correct / total,
        "accuracy_by_error_type": {
            error_type: correct_by_error_type[error_type] / total_by_error_type[error_type]
            for error_type in sorted(total_by_error_type)
        },
    }


def train_local_probe_for_layer(
    layer_idx,
    hidden_dim,
    num_depth_classes,
    train_loader,
    dev_loader,
    device,
    epochs,
    lr,
):
    probe = LocalDepthProbe(hidden_dim, num_depth_classes).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(probe.parameters(), lr=lr)

    best_dev_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        probe.train()

        total_correct = 0
        total_tokens = 0

        for batch in tqdm(
            train_loader,
            desc=f"Local probe layer {layer_idx} epoch {epoch}/{epochs}",
        ):
            reps = batch["reps_by_layer"][layer_idx].to(device)
            labels = batch["local_depths"].to(device)

            optimizer.zero_grad()
            logits = probe(reps)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100

            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

        train_token_accuracy = total_correct / total_tokens
        dev_metrics = evaluate_local_probe_for_layer(
            probe=probe,
            layer_idx=layer_idx,
            dataloader=dev_loader,
            device=device,
        )

        best_dev_accuracy = max(best_dev_accuracy, dev_metrics["token_accuracy"])

        print(
            {
                "probe": "local",
                "layer": layer_idx,
                "epoch": epoch,
                "train_token_accuracy": train_token_accuracy,
                "dev_token_accuracy": dev_metrics["token_accuracy"],
            }
        )

    return probe, best_dev_accuracy


@torch.no_grad()
def evaluate_local_probe_for_layer(probe, layer_idx, dataloader, device):
    probe.eval()

    total_correct = 0
    total_tokens = 0

    correct_by_error_type = defaultdict(int)
    total_by_error_type = defaultdict(int)

    for batch in dataloader:
        reps = batch["reps_by_layer"][layer_idx].to(device)
        labels = batch["local_depths"].to(device)
        error_types = batch["error_type"]

        logits = probe(reps)
        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100

        total_correct += ((preds == labels) & mask).sum().item()
        total_tokens += mask.sum().item()

        for pred_row, label_row, mask_row, error_type in zip(
            preds.cpu(), labels.cpu(), mask.cpu(), error_types
        ):
            correct = ((pred_row == label_row) & mask_row).sum().item()
            count = mask_row.sum().item()

            correct_by_error_type[error_type] += correct
            total_by_error_type[error_type] += count

    return {
        "token_accuracy": total_correct / total_tokens,
        "token_accuracy_by_error_type": {
            error_type: correct_by_error_type[error_type] / total_by_error_type[error_type]
            for error_type in sorted(total_by_error_type)
        },
    }


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.45)
    ax.set_axisbelow(True)


def plot_layerwise_accuracy(metrics, output_dir):
    layers = metrics["layers"]
    global_acc = [metrics["global_by_layer"][str(layer)]["test_accuracy"] for layer in layers]
    local_acc = [metrics["local_by_layer"][str(layer)]["test_token_accuracy"] for layer in layers]

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(
        layers,
        global_acc,
        marker="o",
        linewidth=2.5,
        label="Global depth probe",
        color=GLOBAL_COLOR,
    )
    ax.plot(
        layers,
        local_acc,
        marker="o",
        linewidth=2.5,
        label="Local depth probe",
        color=LOCAL_COLOR,
    )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Representation layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Layer-wise depth probe accuracy", pad=14)
    ax.set_xticks(layers)
    ax.set_xticklabels(["Emb"] + [f"L{i}" for i in layers[1:]])
    ax.legend(frameon=False)
    style_axis(ax)

    fig.tight_layout()
    path = output_dir / "q16_layerwise_probe_accuracy.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"Saved {path}")


def plot_best_layer_by_error_type(metrics, output_dir):
    layers = metrics["layers"]

    best_global_layer = max(
        layers,
        key=lambda layer: metrics["global_by_layer"][str(layer)]["test_accuracy"],
    )
    best_local_layer = max(
        layers,
        key=lambda layer: metrics["local_by_layer"][str(layer)]["test_token_accuracy"],
    )

    error_types = ["NONE", "E1", "E2", "E3", "E4"]
    labels = [
        "NONE",
        "E1\nmissing\ncloser",
        "E2\nspurious\nopener",
        "E3\ntype\nmismatch",
        "E4\npremature\nclose",
    ]

    global_values = [
        metrics["global_by_layer"][str(best_global_layer)]["test_accuracy_by_error_type"][e]
        for e in error_types
    ]
    local_values = [
        metrics["local_by_layer"][str(best_local_layer)]["test_token_accuracy_by_error_type"][e]
        for e in error_types
    ]

    x = range(len(error_types))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(
        [i - width / 2 for i in x],
        global_values,
        width,
        label=f"Global probe, layer {best_global_layer}",
        color=GLOBAL_COLOR,
    )
    bars2 = ax.bar(
        [i + width / 2 for i in x],
        local_values,
        width,
        label=f"Local probe, layer {best_local_layer}",
        color=LOCAL_COLOR,
    )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Best depth probe accuracy by error type", pad=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    style_axis(ax)

    for bars in [bars1, bars2]:
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
    path = output_dir / "q15_layerwise_probe_by_error_type.png"
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

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--output_path", type=str, default="results/probing_layers_metrics.json")
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
    num_depth_classes = 9

    train_dataset = LayerProbeDataset(
        args.train_path,
        model=model,
        device=device,
        max_examples=args.max_train_examples,
    )
    dev_dataset = LayerProbeDataset(
        args.dev_path,
        model=model,
        device=device,
        max_examples=args.max_dev_examples,
    )
    test_dataset = LayerProbeDataset(
        args.test_path,
        model=model,
        device=device,
        max_examples=args.max_test_examples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_layer_probe_batch,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_layer_probe_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_layer_probe_batch,
    )

    num_layers = train_dataset.num_layers
    layers = list(range(num_layers))

    metrics = {
        "layers": layers,
        "global_by_layer": {},
        "local_by_layer": {},
        "num_train_examples": len(train_dataset),
        "num_dev_examples": len(dev_dataset),
        "num_test_examples": len(test_dataset),
    }

    print(f"Number of representation layers: {num_layers}")
    print(f"Layers: {layers}")

    for layer_idx in layers:
        print(f"\n=== Global probe for representation layer {layer_idx} ===")
        global_probe, best_global_dev = train_global_probe_for_layer(
            layer_idx=layer_idx,
            hidden_dim=hidden_dim,
            num_depth_classes=num_depth_classes,
            train_loader=train_loader,
            dev_loader=dev_loader,
            device=device,
            epochs=args.epochs,
            lr=args.learning_rate,
        )

        global_test = evaluate_global_probe_for_layer(
            probe=global_probe,
            layer_idx=layer_idx,
            dataloader=test_loader,
            device=device,
        )

        metrics["global_by_layer"][str(layer_idx)] = {
            "best_dev_accuracy": best_global_dev,
            "test_accuracy": global_test["accuracy"],
            "test_accuracy_by_error_type": global_test["accuracy_by_error_type"],
        }

        print("Global test:", metrics["global_by_layer"][str(layer_idx)])

        print(f"\n=== Local probe for representation layer {layer_idx} ===")
        local_probe, best_local_dev = train_local_probe_for_layer(
            layer_idx=layer_idx,
            hidden_dim=hidden_dim,
            num_depth_classes=num_depth_classes,
            train_loader=train_loader,
            dev_loader=dev_loader,
            device=device,
            epochs=args.epochs,
            lr=args.learning_rate,
        )

        local_test = evaluate_local_probe_for_layer(
            probe=local_probe,
            layer_idx=layer_idx,
            dataloader=test_loader,
            device=device,
        )

        metrics["local_by_layer"][str(layer_idx)] = {
            "best_dev_token_accuracy": best_local_dev,
            "test_token_accuracy": local_test["token_accuracy"],
            "test_token_accuracy_by_error_type": local_test[
                "token_accuracy_by_error_type"
            ],
        }

        print("Local test:", metrics["local_by_layer"][str(layer_idx)])

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved layer-wise probing metrics to {output_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_layerwise_accuracy(metrics, output_dir)
    plot_best_layer_by_error_type(metrics, output_dir)


if __name__ == "__main__":
    main()