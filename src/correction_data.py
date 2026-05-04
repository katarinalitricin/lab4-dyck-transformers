from dataclasses import asdict
from pathlib import Path
import json

from data import (
    DyckExample,
    generate_split,
    save_jsonl,
    OPEN_TO_CLOSE,
    CLOSE_TO_OPEN,
    tokenize,
    MAX_LEN,
)


CORRECTION_LABELS = [
    "OK",
    "DELETE",
    "INSERT_(",
    "INSERT_)",
    "INSERT_[",
    "INSERT_]",
    "REPLACE_(",
    "REPLACE_)",
    "REPLACE_[",
    "REPLACE_]",
]

LABEL_TO_ID = {label: i for i, label in enumerate(CORRECTION_LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def find_deleted_closer(clean: str, corrupted: str, position: int) -> str:
    """
    For E1, the deleted character is the closer at the original corruption position.
    """
    return clean[position]


def label_none(input_string: str) -> list[str]:
    return ["OK"] * len(input_string)


def label_e1_missing_closer(clean: str, corrupted: str, position: int) -> list[str]:
    """
    E1 deletes a closing bracket from clean.

    Since sequence labelling attaches labels to existing tokens, we attach
    INSERT(x) to the token immediately before the missing closer position.
    If the missing closer was at position 0, attach to the first token.
    """
    labels = ["OK"] * len(corrupted)
    missing = find_deleted_closer(clean, corrupted, position)

    if len(labels) == 0:
        return labels

    insert_position = max(0, position - 1)
    insert_position = min(insert_position, len(labels) - 1)

    labels[insert_position] = f"INSERT_{missing}"
    return labels


def label_e2_spurious_opener(input_string: str, position: int) -> list[str]:
    """
    E2 inserts an extra opener at position.
    Mark that inserted token for deletion.
    """
    labels = ["OK"] * len(input_string)

    if 0 <= position < len(labels):
        labels[position] = "DELETE"

    return labels


def label_e3_type_mismatch(clean: str, corrupted: str, position: int) -> list[str]:
    """
    E3 replaces a closing bracket with the wrong closer type.
    Replace it with the original correct closer from the clean string.
    """
    labels = ["OK"] * len(corrupted)
    correct_token = clean[position]

    if 0 <= position < len(labels):
        labels[position] = f"REPLACE_{correct_token}"

    return labels


def label_e4_premature_close(input_string: str, position: int) -> list[str]:
    """
    E4 inserts a premature closing bracket.
    Mark it for deletion.
    """
    labels = ["OK"] * len(input_string)

    if 0 <= position < len(labels):
        labels[position] = "DELETE"

    return labels


def make_correction_labels(example: dict) -> list[str]:
    clean = example["clean"]
    input_string = example["input_string"]
    error_type = example["error_type"]
    position = example["corruption_position"]

    if error_type == "NONE":
        return label_none(input_string)

    if error_type == "E1":
        return label_e1_missing_closer(clean, input_string, position)

    if error_type == "E2":
        return label_e2_spurious_opener(input_string, position)

    if error_type == "E3":
        return label_e3_type_mismatch(clean, input_string, position)

    if error_type == "E4":
        return label_e4_premature_close(input_string, position)

    raise ValueError(f"Unknown error type: {error_type}")


def pad_correction_labels(labels: list[str], max_len: int = MAX_LEN) -> list[int]:
    """
    Align correction labels with tokenized input:
    [CLS] gets OK
    bracket tokens get their labels
    [SEP] gets OK
    [PAD] positions get -100 so they are ignored by CrossEntropyLoss
    """
    full_labels = ["OK"] + labels + ["OK"]

    if len(full_labels) > max_len:
        raise ValueError(f"Label sequence too long: {len(full_labels)} > {max_len}")

    label_ids = [LABEL_TO_ID[label] for label in full_labels]

    while len(label_ids) < max_len:
        label_ids.append(-100)

    return label_ids


def add_correction_labels_to_file(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            example = json.loads(line)
            labels = make_correction_labels(example)
            label_ids = pad_correction_labels(labels)

            example["correction_labels"] = labels
            example["correction_label_ids"] = label_ids

            f_out.write(json.dumps(example) + "\n")


def make_all_correction_files() -> None:
    files = [
        "train",
        "dev",
        "test",
        "ood",
    ]

    for split in files:
        input_path = f"data/processed/{split}.jsonl"
        output_path = f"data/processed/{split}_correction.jsonl"

        print(f"Creating {output_path}")
        add_correction_labels_to_file(input_path, output_path)


if __name__ == "__main__":
    make_all_correction_files()
    print("Correction label files created.")