#!/usr/bin/env python3
"""
Integrated field optimizer using YAML configuration and matcher-based metrics.

Usage:
    python optimize_fields.py --config field_config.yaml --data data.csv --api_key KEY --output ./optimized
"""

import argparse
from pathlib import Path

import dspy
import pandas as pd
from tqdm import tqdm

from scripts.optimizer.data_utils import df_to_dspy, save_datasets
from scripts.optimizer.field_optimizer import FieldOptimizer
from scripts.optimizer.optimizer_utils import load_config, save_optimization_summary


def setup_language_models(api_key: str, student_model: str, reflection_model: str) -> tuple[dspy.LM, dspy.LM]:
    """Setup DSPy language models."""
    lm = dspy.LM(student_model, api_key=api_key, temperature=1.0, max_tokens=20000)
    dspy.settings.configure(lm=lm)

    reflection_lm = dspy.LM(reflection_model, api_key=api_key, temperature=1.0, max_tokens=20000)

    return lm, reflection_lm


def determine_fields_to_optimize(config: dict, args) -> list[str]:
    """Determine which fields to optimize based on arguments."""
    field_configs = config["fields"]

    if args.group:
        if args.group not in config.get("field_groups", {}):
            raise ValueError(f"Group '{args.group}' not found in config")
        return config["field_groups"][args.group]

    if args.fields:
        return args.fields

    return list(field_configs.keys())


def optimize_single_field(
    field_name: str,
    field_config: dict,
    trainset: list,
    valset: list,
    reflection_lm: dspy.LM,
    output_dir: Path,
    enable_logging: bool,
) -> dict:
    """Optimize a single field."""
    log_dir = output_dir / "logs" / field_name
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        optimizer = FieldOptimizer(field_name=field_name, field_config=field_config, reflection_lm=reflection_lm)

        optimized = optimizer.optimize(
            trainset=trainset, valset=valset, enable_logging=enable_logging, log_dir=str(log_dir)
        )

        output_path = output_dir / f"{field_name}_optimized.json"
        optimized.save(str(output_path))

        return {"status": "success", "path": str(output_path)}

    except Exception as e:
        import traceback

        error_detail = traceback.format_exc()
        print(f"✗ Error optimizing {field_name}: {e}")
        print(error_detail)
        return {"status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Optimize fields using YAML configuration")
    parser.add_argument("--config", required=True, help="Path to field_config.yaml")
    parser.add_argument("--data", required=True, help="Path to DataFrame CSV")
    parser.add_argument("--output", default="./data/processed", help="Output directory")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--fields", nargs="+", help="Specific fields to optimize")
    parser.add_argument("--group", help="Field group to optimize")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--no_logging", action="store_true")

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} records")

    # Convert to DSPy datasets
    trainset, valset, testset = df_to_dspy(df, train_frac=args.train_frac, val_frac=args.val_frac)

    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    save_datasets(trainset, valset, testset, output_dir)

    # Setup LMs
    opt_config = config.get("optimization", {})
    lm, reflection_lm = setup_language_models(
        args.api_key, opt_config.get("student_model", "openai/gpt-5-mini"), opt_config.get("reflection_model", "openai/gpt-5-mini")
    )

    # Determine fields
    field_configs = config["fields"]
    fields_to_optimize = determine_fields_to_optimize(config, args)

    print(f"\n{'='*70}")
    print(f"Optimizing {len(fields_to_optimize)} fields")
    print(f"{'='*70}\n")

    # Optimize each field
    results = {}
    for i, field_name in enumerate(tqdm(fields_to_optimize, desc="Optimizing fields"), 1):
        if field_name not in field_configs:
            print(f"Warning: Field '{field_name}' not in config, skipping")
            continue

        print(f"\n[{i}/{len(fields_to_optimize)}] {field_name}")

        result = optimize_single_field(
            field_name=field_name,
            field_config=field_configs[field_name],
            trainset=trainset,
            valset=valset,
            reflection_lm=reflection_lm,
            output_dir=output_dir,
            enable_logging=not args.no_logging,
        )

        results[field_name] = result

    # Save summary
    save_optimization_summary(results, output_dir / "optimization_summary.json")


if __name__ == "__main__":
    main()