from data import generate_split, save_jsonl


def main():
    specs = {
        "ood_depth5": {
            "size": 5000,
            "min_len": 40,
            "max_len": 76,
            "allowed_depths": [5],
            "seed": 50,
        },
        "ood_depth6": {
            "size": 5000,
            "min_len": 40,
            "max_len": 76,
            "allowed_depths": [6],
            "seed": 60,
        },
        "ood_depth7": {
            "size": 5000,
            "min_len": 40,
            "max_len": 76,
            "allowed_depths": [7],
            "seed": 70,
        },
    }

    for name, spec in specs.items():
        print(f"Generating {name}...")
        examples = generate_split(**spec)
        path = f"data/processed/{name}.jsonl"
        save_jsonl(examples, path)
        print(f"Saved {len(examples)} examples to {path}")


if __name__ == "__main__":
    main()