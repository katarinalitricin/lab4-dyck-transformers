import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_accuracy(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics["accuracy"]


def main():
    output_dir = Path("report/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    depths = ["5", "6", "7"]

    before_paths = [
        "results/ood_depth5_metrics_before_finetune.json",
        "results/ood_depth6_metrics_before_finetune.json",
        "results/ood_depth7_metrics_before_finetune.json",
    ]

    after_paths = [
        "results/ood_depth5_metrics_after_finetune.json",
        "results/ood_depth6_metrics_after_finetune.json",
        "results/ood_depth7_metrics_after_finetune.json",
    ]

    before = [load_accuracy(path) for path in before_paths]
    after = [load_accuracy(path) for path in after_paths]

    x = range(len(depths))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar([i - width / 2 for i in x], before, width, label="Before fine-tuning")
    ax.bar([i + width / 2 for i in x], after, width, label="After fine-tuning")

    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Nesting depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("OOD accuracy before and after depth-5 fine-tuning")
    ax.set_xticks(list(x))
    ax.set_xticklabels(depths)
    ax.legend()

    for i, value in enumerate(before):
        ax.text(i - width / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)

    for i, value in enumerate(after):
        ax.text(i + width / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)

    fig.tight_layout()

    path = output_dir / "q9_finetune_before_after.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved {path}")


if __name__ == "__main__":
    main()