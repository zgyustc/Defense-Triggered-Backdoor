#!/usr/bin/env python3
"""Create an ImageFolder view of official TinyImageNet without moving images."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Extracted tiny-imagenet-200 root")
    parser.add_argument("--output", required=True, help="New ImageFolder root")
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output:
        parser.error("--output must differ from --source")
    annotations = source / "val/val_annotations.txt"
    classes = (source / "wnids.txt").read_text().split()
    if len(classes) != 200 or len(set(classes)) != 200:
        raise ValueError("Expected 200 unique TinyImageNet classes")
    entries = [line.split()[:2] for line in annotations.read_text().splitlines() if line.strip()]
    for filename, label in entries:
        if Path(filename).name != filename or label not in classes:
            raise ValueError(f"Invalid validation annotation: {filename}, {label}")
        if not (source / "val/images" / filename).is_file():
            raise FileNotFoundError(filename)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "train").exists() or (output / "train").is_symlink():
        raise FileExistsError("Output already contains train; use a new output directory")
    (output / "train").symlink_to(source / "train", target_is_directory=True)
    for label in classes:
        (output / "val" / label).mkdir(parents=True, exist_ok=True)
    for filename, label in entries:
        (output / "val" / label / filename).symlink_to(source / "val/images" / filename)
    print(f"Created {len(entries)} validation links under {output}")


if __name__ == "__main__":
    main()
