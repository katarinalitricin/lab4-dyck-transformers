import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_accuracy_by_depth(output_dir: Path):
    depth_files = {
        "5": "results/ood_depth5_metrics_before_finetune.json",
        "6": "results/ood_depth6_metrics_before_finetune.json",
        "7": "results/ood_depth7_metrics_before_finetune.json",
    }

    depths = []
    accuracies = []

    for depth, path in depth_files.items():
        metrics = load_json(path)
        depths.append(depth)
        accuracies.append(metrics["accuracy"])

    plt.figure()
    plt.bar(depths, accuracies)
    plt.ylim(0, 1.0)
    plt.xlabel("Nesting depth")
    plt.ylabel("Accuracy")
    plt.title("OOD accuracy by nesting depth")

    for i, value in enumerate(accuracies):
        plt.text(i, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()
    path = output_dir / "q7_ood_accuracy_by_depth.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved {path}")


def plot_accuracy_by_length(output_dir: Path):
    metrics = load_json("results/ood_metrics.json")
    acc_by_length = metrics["accuracy_by_length_bucket"]

    order = ["21-40", "41-60", "61-80"]
    values = [acc_by_length[key] for key in order]

    plt.figure()
    plt.bar(order, values)
    plt.ylim(0, 1.0)
    plt.xlabel("Length bucket")
    plt.ylabel("Accuracy")
    plt.title("OOD accuracy by sequence length")

    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f"{value:.3f}", ha="center")

    plt.tight_layout()
    path = output_dir / "q7_ood_accuracy_by_length.png"
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved {path}")


def main():
    output_dir = Path("report/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_by_depth(output_dir)
    plot_accuracy_by_length(output_dir)


if __name__ == "__main__":
    main()