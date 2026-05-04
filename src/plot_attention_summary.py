import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import DyckTransformerClassifier
from train import get_device


OPEN_TO_CLOSE = {"(": ")", "[": "]"}
CLOSE_TO_OPEN = {")": "(", "]": "["}

MATCH_COLOR = "#7C3AED"      # purple
OTHER_COLOR = "#14B8A6"      # teal
ERROR_COLOR = "#F97316"      # orange
GRID_COLOR = "#E5E7EB"


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def find_matching_pairs(s: str):
    """
    Return valid matching bracket pairs as raw string positions.
    """
    stack = []
    pairs = []

    for i, ch in enumerate(s):
        if ch in OPEN_TO_CLOSE:
            stack.append((ch, i))
        elif ch in CLOSE_TO_OPEN:
            if stack:
                opener, opener_idx = stack.pop()
                if OPEN_TO_CLOSE[opener] == ch:
                    pairs.append((opener_idx, i))

    return pairs


def manual_encoder_forward_with_attention(model, input_ids, attention_mask):
    """
    Run Transformer manually and collect attention maps.

    Returns:
        attentions: list of tensors, one per layer.
        Each tensor shape: [batch, heads, seq_len, seq_len]
    """
    x = model.token_embedding(input_ids)
    x = model.position_encoding(x)
    x = model.dropout(x)

    src_key_padding_mask = attention_mask == 0
    attentions = []

    for layer in model.encoder.layers:
        if layer.norm_first:
            normed = layer.norm1(x)

            attn_output, attn_weights = layer.self_attn(
                normed,
                normed,
                normed,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )

            x = x + layer.dropout1(attn_output)

            normed = layer.norm2(x)
            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(normed)))
            )
            x = x + layer.dropout2(ff_output)

        else:
            attn_output, attn_weights = layer.self_attn(
                x,
                x,
                x,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )

            x = layer.norm1(x + layer.dropout1(attn_output))

            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x)))
            )
            x = layer.norm2(x + layer.dropout2(ff_output))

        attentions.append(attn_weights.detach().cpu())

    return attentions


def get_attention_for_example(model, example, device, layer_index=-1):
    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long).to(device)
    attention_mask = torch.tensor([example["attention_mask"]], dtype=torch.long).to(device)

    attentions = manual_encoder_forward_with_attention(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    attention = attentions[layer_index][0]  # [heads, seq_len, seq_len]
    seq_len = sum(example["attention_mask"])

    return attention[:, :seq_len, :seq_len]


def compute_matching_attention(model, examples, device, layer_index=-1, max_examples=200):
    """
    For valid examples, compare attention on true matching pairs
    against attention on non-matching bracket-token pairs.
    """
    matching_scores = []
    nonmatching_scores = []

    used = 0

    for example in examples:
        if example["error_type"] != "NONE":
            continue

        s = example["input_string"]
        pairs = find_matching_pairs(s)

        if not pairs:
            continue

        attention = get_attention_for_example(
            model=model,
            example=example,
            device=device,
            layer_index=layer_index,
        )

        # Average across heads.
        attention = attention.mean(dim=0)

        # +1 because raw string positions exclude [CLS].
        matching_positions = set()
        for open_idx, close_idx in pairs:
            open_tok = open_idx + 1
            close_tok = close_idx + 1

            matching_positions.add((open_tok, close_tok))
            matching_positions.add((close_tok, open_tok))

        bracket_positions = list(range(1, len(s) + 1))

        for query_pos in bracket_positions:
            for key_pos in bracket_positions:
                if query_pos == key_pos:
                    continue

                score = attention[query_pos, key_pos].item()

                if (query_pos, key_pos) in matching_positions:
                    matching_scores.append(score)
                else:
                    nonmatching_scores.append(score)

        used += 1
        if used >= max_examples:
            break

    return {
        "matching_attention": sum(matching_scores) / len(matching_scores),
        "nonmatching_attention": sum(nonmatching_scores) / len(nonmatching_scores),
        "num_examples": used,
    }


def compute_corruption_attention(model, examples, device, layer_index=-1, max_examples=200):
    """
    For erroneous examples, compare attention to the corrupted token
    against attention to all other bracket tokens.
    """
    corruption_scores = []
    other_scores = []

    used = 0

    for example in examples:
        if example["error_type"] == "NONE":
            continue

        position = example["corruption_position"]

        if position is None:
            continue

        s = example["input_string"]
        corruption_token_position = position + 1  # +1 for [CLS]

        attention = get_attention_for_example(
            model=model,
            example=example,
            device=device,
            layer_index=layer_index,
        )

        # Average across heads.
        attention = attention.mean(dim=0)

        bracket_positions = list(range(1, len(s) + 1))

        for query_pos in bracket_positions:
            if query_pos == corruption_token_position:
                continue

            corruption_scores.append(
                attention[query_pos, corruption_token_position].item()
            )

            for key_pos in bracket_positions:
                if key_pos == query_pos or key_pos == corruption_token_position:
                    continue

                other_scores.append(attention[query_pos, key_pos].item())

        used += 1
        if used >= max_examples:
            break

    return {
        "corruption_attention": sum(corruption_scores) / len(corruption_scores),
        "other_attention": sum(other_scores) / len(other_scores),
        "num_examples": used,
    }


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, alpha=0.45)
    ax.set_axisbelow(True)


def plot_q10(summary, output_path):
    labels = ["Matching\nbracket pairs", "Other\nbracket pairs"]
    values = [
        summary["matching_attention"],
        summary["nonmatching_attention"],
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=[MATCH_COLOR, OTHER_COLOR], width=0.55)

    ax.set_ylabel("Mean attention weight")
    ax.set_title("Q10: Attention to matching vs non-matching brackets")
    style_axis(ax)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value:.4f}",
            ha="center",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    print(f"Saved {output_path}")


def plot_q11(summary, output_path):
    labels = ["Corrupted\ntoken", "Other\ntokens"]
    values = [
        summary["corruption_attention"],
        summary["other_attention"],
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=[ERROR_COLOR, OTHER_COLOR], width=0.55)

    ax.set_ylabel("Mean attention weight")
    ax.set_title("Q11: Attention to corrupted token vs other tokens")
    style_axis(ax)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            f"{value:.4f}",
            ha="center",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)

    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/detection.pt")
    parser.add_argument("--data_path", type=str, default="data/processed/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="report/figures")
    parser.add_argument("--layer_index", type=int, default=-1)
    parser.add_argument("--max_examples", type=int, default=200)

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = DyckTransformerClassifier(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    examples = load_jsonl(args.data_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q10_summary = compute_matching_attention(
        model=model,
        examples=examples,
        device=device,
        layer_index=args.layer_index,
        max_examples=args.max_examples,
    )

    q11_summary = compute_corruption_attention(
        model=model,
        examples=examples,
        device=device,
        layer_index=args.layer_index,
        max_examples=args.max_examples,
    )

    print("Q10 summary:", q10_summary)
    print("Q11 summary:", q11_summary)

    with (output_dir / "q10_q11_attention_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "q10": q10_summary,
                "q11": q11_summary,
                "layer_index": args.layer_index,
                "max_examples": args.max_examples,
            },
            f,
            indent=2,
        )

    plot_q10(
        q10_summary,
        output_dir / "q10_matching_attention_summary.png",
    )

    plot_q11(
        q11_summary,
        output_dir / "q11_corruption_attention_summary.png",
    )


if __name__ == "__main__":
    main()