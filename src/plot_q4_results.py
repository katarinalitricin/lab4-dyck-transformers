import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_confusion_matrix(metrics, output_dir: Path):
    cm = metrics["confusion_matrix"]

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["valid", "error"])
    ax.set_yticklabels(["valid", "error"])

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_title("Binary detection confusion matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()

    path = output_dir / "q4_confusion_matrix.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved {path}")


def plot_error_type_accuracy(metrics, output_dir: Path):
    acc_by_type = metrics["accuracy_by_error_type"]

    order = ["NONE", "E1", "E2", "E3", "E4"]
    labels = ["NONE", "E1\nmissing\ncloser", "E2\nspurious\nopener", "E3\ntype\nmismatch", "E4\npremature\nclose"]
    values = [acc_by_type[key] for key in order]

    fig, ax = plt.subplots()
    ax.bar(labels, values)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Detection accuracy by error type")

    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center")

    fig.tight_layout()

    path = output_dir / "q4_accuracy_by_error_type.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved {path}")


def main():
    metrics_path = "results/test_metrics.json"
    output_dir = Path("report/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_path)

    plot_confusion_matrix(metrics, output_dir)
    plot_error_type_accuracy(metrics, output_dir)


if __name__ == "__main__":
    main()