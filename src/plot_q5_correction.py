import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    output_dir = Path("report/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_metrics = load_json("results/correction_test_metrics.json")
    ood_metrics = load_json("results/correction_ood_metrics.json")

    error_types = ["NONE", "E1", "E2", "E3", "E4"]

    test_values = [
        test_metrics["exact_match_by_error_type"][error_type]
        for error_type in error_types
    ]

    ood_values = [
        ood_metrics["exact_match_by_error_type"][error_type]
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
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar([i - width / 2 for i in x], test_values, width, label="Test")
    ax.bar([i + width / 2 for i in x], ood_values, width, label="OOD")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("Correction exact-match accuracy by error type")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()

    for i, value in enumerate(test_values):
        ax.text(i - width / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)

    for i, value in enumerate(ood_values):
        ax.text(i + width / 2, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)

    fig.tight_layout()

    path = output_dir / "q5_correction_exact_match_by_error_type.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved {path}")

    # Also save a small summary plot for overall test vs OOD exact-match.
    fig, ax = plt.subplots()

    overall_labels = ["Test", "OOD"]
    overall_values = [
        test_metrics["exact_match_accuracy"],
        ood_metrics["exact_match_accuracy"],
    ]

    ax.bar(overall_labels, overall_values)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact-match accuracy")
    ax.set_title("Correction exact-match accuracy: test vs OOD")

    for i, value in enumerate(overall_values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center")

    fig.tight_layout()

    path = output_dir / "q5_correction_exact_match_test_vs_ood.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved {path}")


if __name__ == "__main__":
    main()