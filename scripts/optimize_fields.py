#!/usr/bin/env python3
"""
Field optimizer using YAML configuration and matcher-based metrics.

Usage:
    python optimize_fields.py --config configs/fields_config.yaml --data data.json --api_key KEY

    # Parallel execution (4 workers):
    python optimize_fields.py --config configs/fields_config.yaml --data data.json --api_key KEY --n_workers 4
"""

import sys
from pathlib import Path

# Add scripts directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import dspy
from tqdm import tqdm

from optimization.data_utils import json_to_dspy, save_datasets
from optimization.field_optimizer import FieldOptimizer
from optimization.optimizer_utils import load_config, save_optimization_summary


def setup_language_models(api_key: str, student_model: str, reflection_model: str) -> tuple[dspy.LM, dspy.LM]:
    """Setup DSPy language models."""
    lm = dspy.LM(student_model, api_key=api_key, temperature=1.0, max_tokens=20000)
    dspy.settings.configure(lm=lm)
    
    reflection_lm = dspy.LM(reflection_model, api_key=api_key, temperature=1.0, max_tokens=20000)
    
    return lm, reflection_lm


def determine_fields(config: dict, args) -> list[str]:
    """Determine which fields to optimize."""
    if args.group:
        if args.group not in config.get("field_groups", {}):
            raise ValueError(f"Group '{args.group}' not found in config")
        return config["field_groups"][args.group]
    
    if args.fields:
        return args.fields
    
    return list(config["fields"].keys())


def optimize_field_worker(field_name: str, field_config: dict, trainset: list, valset: list,
                          api_key: str, student_model: str, reflection_model: str,
                          output_dir: Path, enable_logging: bool) -> tuple[str, dict]:
    """Optimize a single field (worker function for parallel execution)."""
    try:
        # CRITICAL: Setup path in worker process (ProcessPoolExecutor creates new processes)
        import sys
        import os

        # Setup LMs in worker process
        lm = dspy.LM(student_model, api_key=api_key, temperature=1.0, max_tokens=20000)
        dspy.settings.configure(lm=lm)
        
        reflection_lm = dspy.LM(reflection_model, api_key=api_key, temperature=1.0, max_tokens=20000)
        
        # Create log directory
        log_dir = output_dir / "logs" / field_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimize
        optimizer = FieldOptimizer(field_name, field_config, reflection_lm)
        optimized = optimizer.optimize(trainset, valset, enable_logging=enable_logging, log_dir=str(log_dir))
        
        # Save
        output_path = output_dir / f"{field_name}_optimized.json"
        optimized.save(str(output_path))
        
        return field_name, {"status": "success", "path": str(output_path)}
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return field_name, {"status": "failed", "error": str(e), "traceback": error_detail}


def optimize_field(field_name: str, field_config: dict, trainset: list, valset: list,
                   reflection_lm: dspy.LM, output_dir: Path, enable_logging: bool) -> dict:
    """Optimize a single field (sequential execution)."""
    log_dir = output_dir / "logs" / field_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        optimizer = FieldOptimizer(field_name, field_config, reflection_lm)
        optimized = optimizer.optimize(trainset, valset, enable_logging=enable_logging, log_dir=str(log_dir))
        
        output_path = output_dir / f"{field_name}_optimized.json"
        optimized.save(str(output_path))
        
        return {"status": "success", "path": str(output_path)}
    
    except Exception as e:
        import traceback
        print(f"✗ Error optimizing {field_name}: {e}")
        print(traceback.format_exc())
        return {"status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Optimize fields using YAML configuration")
    parser.add_argument("--config", required=True, help="Path to fields_config.yaml")
    parser.add_argument("--data", required=True, help="Path to dataset JSON")
    parser.add_argument("--output", default="./data/processed", help="Output directory")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--fields", nargs="+", help="Specific fields to optimize")
    parser.add_argument("--group", help="Field group to optimize (e.g., wave1_all)")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--no_logging", action="store_true")
    parser.add_argument("--n_workers", type=int, default=16, help="Number of parallel workers (default: 1 = sequential)")
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Load data
    print(f"Loading data from {args.data}...")
    trainset, valset, testset = json_to_dspy(Path(args.data), train_frac=args.train_frac, val_frac=args.val_frac)
    
    # Setup output
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    save_datasets(trainset, valset, testset, output_dir)
    
    # Get model names from config
    opt_config = config.get("optimization", {})
    student_model = opt_config.get("student_model", "openai/gpt-5-mini")
    reflection_model = opt_config.get("reflection_model", "openai/gpt-5-mini")
    
    # Setup LMs (only for sequential mode)
    if args.n_workers == 1:
        lm, reflection_lm = setup_language_models(args.api_key, student_model, reflection_model)
    
    # Determine fields
    fields_to_optimize = determine_fields(config, args)
    
    mode = "parallel" if args.n_workers > 1 else "sequential"
    print(f"\n{'='*70}")
    print(f"Optimizing {len(fields_to_optimize)} fields ({mode} mode, {args.n_workers} workers)")
    print(f"{'='*70}\n")
    
    results = {}
    
    if args.n_workers > 1:
        # PARALLEL EXECUTION
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(
                    optimize_field_worker,
                    field_name,
                    config["fields"][field_name],
                    trainset,
                    valset,
                    args.api_key,
                    student_model,
                    reflection_model,
                    output_dir,
                    not args.no_logging
                ): field_name
                for field_name in fields_to_optimize
                if field_name in config["fields"]
            }
            
            for future in as_completed(futures):
                field_name = futures[future]
                name, result = future.result()
                results[name] = result
                
                if result["status"] == "success":
                    print(f"✓ {name}: {result['path']}")
                else:
                    print(f"✗ {name}: {result['error']}")
    
    else:
        # SEQUENTIAL EXECUTION
        for i, field_name in enumerate(tqdm(fields_to_optimize, desc="Optimizing"), 1):
            if field_name not in config["fields"]:
                print(f"Warning: Field '{field_name}' not in config, skipping")
                continue
            
            print(f"\n[{i}/{len(fields_to_optimize)}] {field_name}")
            
            results[field_name] = optimize_field(
                field_name, config["fields"][field_name], trainset, valset,
                reflection_lm, output_dir, not args.no_logging
            )
    
    # Save summary
    save_optimization_summary(results, output_dir / "optimization_summary.json")


if __name__ == "__main__":
    main()