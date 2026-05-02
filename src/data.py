import random
from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path


PAIRS = [("(", ")"), ("[", "]")]
OPEN_TO_CLOSE = {"(": ")", "[": "]"}
CLOSE_TO_OPEN = {")": "(", "]": "["}

VOCAB = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "(": 3,
    ")": 4,
    "[": 5,
    "]": 6,
}

ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}
MAX_LEN = 80


@dataclass
class DyckExample:
    clean: str
    corrupted: str
    input_string: str
    is_error: int
    error_type: str
    depth: int
    length: int
    corruption_position: Optional[int]
    input_ids: list[int]
    attention_mask: list[int]


def max_nesting_depth(s: str) -> int:
    """Return the maximum stack depth reached by a bracket string."""
    stack = []
    max_depth = 0

    for ch in s:
        if ch in OPEN_TO_CLOSE:
            stack.append(ch)
            max_depth = max(max_depth, len(stack))
        elif ch in CLOSE_TO_OPEN:
            if stack:
                stack.pop()

    return max_depth


def is_valid_dyck(s: str) -> bool:
    """Check whether a string is a valid Dyck string over () and []."""
    stack = []

    for ch in s:
        if ch in OPEN_TO_CLOSE:
            stack.append(ch)
        elif ch in CLOSE_TO_OPEN:
            if not stack:
                return False
            opener = stack.pop()
            if OPEN_TO_CLOSE[opener] != ch:
                return False
        else:
            return False

    return len(stack) == 0


def generate_dyck(length: int, max_depth: int, k: int = 2) -> Optional[str]:
    """
    Return a random Dyck string of exactly `length` tokens, with maximum
    nesting depth <= max_depth, or None on failure.
    """
    assert length % 2 == 0

    stack = []
    result = []
    depth = 0
    remaining = length

    for _ in range(length):
        must_close = len(stack) == remaining
        can_open = depth < max_depth and remaining > len(stack) + 1

        choices = []
        if can_open and not must_close:
            choices.append("open")
        if stack:
            choices.append("close")

        if not choices:
            return None

        choice = random.choice(choices)

        if choice == "open":
            pair = random.choice(PAIRS[:k])
            stack.append(pair)
            result.append(pair[0])
            depth += 1
        else:
            pair = stack.pop()
            result.append(pair[1])
            depth = len(stack)

        remaining -= 1

    candidate = "".join(result)

    if stack:
        return None

    return candidate


def generate_dyck_exact_depth(
    length: int,
    target_depth: int,
    max_attempts: int = 1000,
) -> str:
    """Generate a valid Dyck string with exactly target_depth."""
    for _ in range(max_attempts):
        s = generate_dyck(length=length, max_depth=target_depth)

        if s is not None and max_nesting_depth(s) == target_depth:
            return s

    raise RuntimeError(
        f"Could not generate string with length={length}, depth={target_depth}"
    )


def corrupt_missing_closer(s: str) -> tuple[str, int]:
    """E1: Delete a closing bracket."""
    closing_positions = [i for i, ch in enumerate(s) if ch in CLOSE_TO_OPEN]
    pos = random.choice(closing_positions)
    corrupted = s[:pos] + s[pos + 1 :]
    return corrupted, pos


def corrupt_spurious_opener(s: str) -> tuple[str, int]:
    """E2: Insert an extra opening bracket."""
    pos = random.randint(0, len(s))
    opener = random.choice(list(OPEN_TO_CLOSE.keys()))
    corrupted = s[:pos] + opener + s[pos:]
    return corrupted, pos


def corrupt_type_mismatch(s: str) -> tuple[str, int]:
    """E3: Replace a closing bracket with the wrong closer type."""
    closing_positions = [i for i, ch in enumerate(s) if ch in CLOSE_TO_OPEN]
    pos = random.choice(closing_positions)

    old = s[pos]
    new = ")" if old == "]" else "]"

    corrupted = s[:pos] + new + s[pos + 1 :]
    return corrupted, pos


def corrupt_premature_close(s: str) -> tuple[str, int]:
    """
    E4: Insert a closing bracket at the beginning, before any opener exists.
    This guarantees a premature close.
    """
    closer = random.choice(list(CLOSE_TO_OPEN.keys()))
    pos = 0
    corrupted = closer + s
    return corrupted, pos


def corrupt_string(s: str, error_type: str) -> tuple[str, int]:
    """Apply one corruption operation."""
    if error_type == "E1":
        return corrupt_missing_closer(s)
    if error_type == "E2":
        return corrupt_spurious_opener(s)
    if error_type == "E3":
        return corrupt_type_mismatch(s)
    if error_type == "E4":
        return corrupt_premature_close(s)

    raise ValueError(f"Unknown error type: {error_type}")


def tokenize(s: str, max_len: int = MAX_LEN) -> tuple[list[int], list[int]]:
    """
    Tokenise with [CLS] at the beginning and [SEP] at the end.
    Pad to max_len.
    """
    tokens = ["[CLS]"] + list(s) + ["[SEP]"]

    if len(tokens) > max_len:
        raise ValueError(f"Sequence too long: {len(tokens)} > {max_len}")

    input_ids = [VOCAB[tok] for tok in tokens]
    attention_mask = [1] * len(input_ids)

    while len(input_ids) < max_len:
        input_ids.append(VOCAB["[PAD]"])
        attention_mask.append(0)

    return input_ids, attention_mask


def make_example(clean: str, error_type: str = "NONE") -> DyckExample:
    """
    Create one example.

    If error_type == "NONE", the input is the clean string.
    Otherwise, the input is a corrupted version of the clean string.
    """
    depth = max_nesting_depth(clean)

    if error_type == "NONE":
        input_string = clean
        corrupted = clean
        is_error = 0
        corruption_position = None
    else:
        corrupted, corruption_position = corrupt_string(clean, error_type)
        input_string = corrupted
        is_error = 1

    input_ids, attention_mask = tokenize(input_string)

    return DyckExample(
        clean=clean,
        corrupted=corrupted,
        input_string=input_string,
        is_error=is_error,
        error_type=error_type,
        depth=depth,
        length=len(input_string),
        corruption_position=corruption_position,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


def random_even_length(min_len: int, max_len: int) -> int:
    """Sample a random even length between min_len and max_len."""
    possible_lengths = [x for x in range(min_len, max_len + 1) if x % 2 == 0]
    return random.choice(possible_lengths)


def generate_split(
    size: int,
    min_len: int,
    max_len: int,
    allowed_depths: list[int],
    seed: int,
) -> list[DyckExample]:
    """
    Generate a balanced dataset over these classes:

    NONE, E1, E2, E3, E4.
    """
    random.seed(seed)

    classes = ["NONE", "E1", "E2", "E3", "E4"]
    per_class = size // len(classes)
    examples = []

    for error_type in classes:
        for _ in range(per_class):
            target_depth = random.choice(allowed_depths)
            length = random_even_length(min_len, max_len)

            # Ensure the string is long enough to reach the target depth.
            while length < 2 * target_depth:
                length = random_even_length(min_len, max_len)

            clean = generate_dyck_exact_depth(
                length=length,
                target_depth=target_depth,
            )

            example = make_example(clean, error_type=error_type)
            examples.append(example)

    random.shuffle(examples)
    return examples


def save_jsonl(examples: list[DyckExample], path: str | Path) -> None:
    """Save examples as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(asdict(example)) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load examples from JSONL."""
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_all_splits(output_dir: str = "data/processed") -> None:
    """Generate train, dev, test, and OOD splits."""
    output_dir = Path(output_dir)

    specs = {
        "train": {
            "size": 50_000,
            "min_len": 4,
            "max_len": 40,
            "allowed_depths": [1, 2, 3, 4],
            "seed": 1,
        },
        "dev": {
            "size": 5_000,
            "min_len": 4,
            "max_len": 40,
            "allowed_depths": [1, 2, 3, 4],
            "seed": 2,
        },
        "test": {
            "size": 5_000,
            "min_len": 4,
            "max_len": 40,
            "allowed_depths": [1, 2, 3, 4],
            "seed": 3,
        },
        "ood": {
            "size": 5_000,
            "min_len": 40,
            "max_len": 80,
            "allowed_depths": [5, 6, 7],
            "seed": 4,
        },
    }

    for name, spec in specs.items():
        print(f"Generating {name} split...")
        examples = generate_split(**spec)
        output_path = output_dir / f"{name}.jsonl"
        save_jsonl(examples, output_path)
        print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    # Quick sanity checks on a tiny in-memory dataset.
    examples = generate_split(
        size=20,
        min_len=4,
        max_len=20,
        allowed_depths=[1, 2, 3],
        seed=42,
    )

    for example in examples[:10]:
        print(
            {
                "clean": example.clean,
                "input": example.input_string,
                "is_error": example.is_error,
                "error_type": example.error_type,
                "depth": example.depth,
                "length": example.length,
                "corruption_position": example.corruption_position,
                "valid_clean": is_valid_dyck(example.clean),
                "valid_input": is_valid_dyck(example.input_string),
            }
        )

    assert all(is_valid_dyck(example.clean) for example in examples)

    assert all(
        is_valid_dyck(example.input_string) == (example.error_type == "NONE")
        for example in examples
    )

    assert all(len(example.input_ids) == MAX_LEN for example in examples)
    assert all(len(example.attention_mask) == MAX_LEN for example in examples)

    print("All sanity checks passed.")

    # Temporary small saved dataset for testing save/load + baseline evaluation.
    small_examples = generate_split(
        size=100,
        min_len=4,
        max_len=20,
        allowed_depths=[1, 2, 3],
        seed=123,
    )

    save_jsonl(small_examples, "data/processed/sample.jsonl")
    print("Saved small sample dataset to data/processed/sample.jsonl")

    # Do not generate the full dataset yet.
    # generate_all_splits()
