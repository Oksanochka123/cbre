#!/usr/bin/env python3
"""
Generate training data from mapper and config with preview.

Usage:
    python gen_data.py --mapper mapper.pkl --config configs/fields_config.yaml --output data/dataset.json
"""

import argparse
import json
from pathlib import Path

from scripts.optimization.data_utils import mapper_to_dspy


def preview_example(example, max_text_len: int = 500):
    """Print formatted preview of a single example."""
    print("\n" + "="*70)
    text = example.document_text[:max_text_len]
    print(f"Document text: {text}...")
    print("-"*70)
    
    # Show up to 5 fields
    fields = [k for k in example.toDict().keys() if k != "document_text"][:5]
    for field in fields:
        value = getattr(example, field, None)
        if isinstance(value, str) and len(value) > 100:
            value = value[:100] + "..."
        print(f"  {field}: {value}")
    
    if len(example.toDict()) > 6:
        print(f"  ... and {len(example.toDict()) - 6} more fields")


def save_json_dataset(trainset, valset, testset, output_path: Path):
    """Save datasets as single JSON file."""
    data = []
    
    for split_name, dataset in [("train", trainset), ("val", valset), ("test", testset)]:
        for example in dataset:
            record = example.toDict()
            record["_split"] = split_name
            data.append(record)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved {len(data)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training data with preview")
    parser.add_argument("--mapper", required=True, help="Path to mapper.pkl")
    parser.add_argument("--config", required=True, help="Path to fields_config.yaml")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--preview", type=int, default=2, help="Number of examples to preview")
    
    args = parser.parse_args()
    
    print(f"Loading mapper from {args.mapper}...")
    print(f"Loading config from {args.config}...")
    
    # Generate datasets
    trainset, valset, testset = mapper_to_dspy(
        Path(args.mapper),
        Path(args.config),
        train_frac=args.train_frac,
        val_frac=args.val_frac
    )
    
    # Preview examples
    if args.preview > 0:
        print(f"\n{'='*70}")
        print(f"PREVIEW: First {args.preview} training examples")
        print(f"{'='*70}")
        
        for i, example in enumerate(trainset[:args.preview], 1):
            print(f"\nExample {i}:")
            preview_example(example)
    
    # Save to JSON
    print(f"\n{'='*70}")
    print("Saving dataset...")
    save_json_dataset(trainset, valset, testset, Path(args.output))
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total examples: {len(trainset) + len(valset) + len(testset)}")
    print(f"  Train: {len(trainset)}")
    print(f"  Val: {len(valset)}")
    print(f"  Test: {len(testset)}")
    print(f"\nDataset ready for: python optimize_fields.py --data {args.output}")


if __name__ == "__main__":
    main()