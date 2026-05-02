import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(metrics_path: str, output_path: str) -> None:
    metrics = load_metrics(metrics_path)

    epochs = [m["epoch"] for m in metrics]

    train_loss = [m["train_loss"] for m in metrics]
    dev_loss = [m["dev_loss"] for m in metrics]

    train_accuracy = [m["train_accuracy"] for m in metrics]
    dev_accuracy = [m["dev_accuracy"] for m in metrics]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    plt.plot(epochs, dev_loss, marker="o", label="dev loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and development loss")
    plt.legend()
    plt.tight_layout()
    loss_path = output_path.parent / "detection_loss_curve.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracy, marker="o", label="train accuracy")
    plt.plot(epochs, dev_accuracy, marker="o", label="dev accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and development accuracy")
    plt.legend()
    plt.tight_layout()
    acc_path = output_path.parent / "detection_accuracy_curve.png"
    plt.savefig(acc_path, dpi=200)
    plt.close()

    print(f"Saved loss curve to {loss_path}")
    print(f"Saved accuracy curve to {acc_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="results/detection_train_metrics.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/plots/detection_training_curves.png",
    )

    args = parser.parse_args()

    plot_training_curves(
        metrics_path=args.metrics_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()