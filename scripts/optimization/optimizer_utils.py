"""Utilities for optimizer: logging, results management, config loading."""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


# ============================================================================
# LOGGING UTILITIES
# ============================================================================


@contextmanager
def dspy_logging(log_dir: str = "./data/interim"):
    """Context manager for clean DSPy logging."""
    log_filename = f"{log_dir}/dspy_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    dspy_logger = logging.getLogger("dspy")
    original_handlers = dspy_logger.handlers.copy()
    original_level = dspy_logger.level
    original_propagate = dspy_logger.propagate

    dspy_logger.handlers.clear()
    dspy_logger.setLevel(logging.INFO)
    dspy_logger.propagate = False

    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    dspy_logger.addHandler(file_handler)

    print(f"✓ Logging to: {log_filename}")

    try:
        yield log_filename
    finally:
        file_handler.flush()
        file_handler.close()
        dspy_logger.handlers.clear()
        dspy_logger.handlers.extend(original_handlers)
        dspy_logger.setLevel(original_level)
        dspy_logger.propagate = original_propagate


# ============================================================================
# CONFIG UTILITIES
# ============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# RESULTS UTILITIES
# ============================================================================


def save_gepa_results(detailed_results, path: Path, field_name: str, field_type: str):
    """Save GEPA optimization results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "field_name": field_name,
        "field_type": field_type,
        "num_candidates": len(detailed_results.candidates),
        "val_aggregate_scores": detailed_results.val_aggregate_scores,
        "best_idx": detailed_results.best_idx,
        "total_metric_calls": detailed_results.total_metric_calls,
        "candidate_instructions": [
            {
                "index": i,
                "score": detailed_results.val_aggregate_scores[i],
                "instructions": c.predict.signature.instructions if hasattr(c, "predict") else "",
                "instruction_length": len(c.predict.signature.instructions) if hasattr(c, "predict") else 0,
            }
            for i, c in enumerate(detailed_results.candidates)
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def handle_tied_candidates(optimized_program, results):
    """Handle tie-breaking for GEPA candidates.
    
    Args:
        optimized_program: The optimized program from GEPA
        results: The detailed_results attribute
        
    Returns:
        tuple: (best_program, num_tied, best_score)
    """
    if not hasattr(optimized_program, "detailed_results"):
        return optimized_program, 0, None

    best_score = results.val_aggregate_scores[results.best_idx]

    # Find all candidates tied at best score
    tied_indices = [i for i, score in enumerate(results.val_aggregate_scores) if abs(score - best_score) < 1e-6]

    if len(tied_indices) > 1:
        # Select latest (most evolved) if tie
        best_idx = max(tied_indices)
        print(f"✓ Found {len(tied_indices)} tied at {best_score:.3f}, using latest (#{best_idx})")
        return results.candidates[best_idx], len(tied_indices), best_score

    return optimized_program, 1, best_score


def save_optimization_summary(results: Dict[str, Any], output_path: Path):
    """Save optimization summary to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    print(f"✓ Successful: {successful}/{len(results)}")
    print(f"✗ Failed: {failed}/{len(results)}")

    if failed > 0:
        failed_fields = [k for k, v in results.items() if v["status"] == "failed"]
        print(f"\nFailed fields: {', '.join(failed_fields)}")

    print(f"\n✓ Summary saved to: {output_path}")
