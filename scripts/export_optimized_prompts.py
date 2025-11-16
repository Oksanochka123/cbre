import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

#!/usr/bin/env python3
"""
Export Optimized Prompts Script

This script processes optimization logs from prompt optimization runs and exports
the best-performing prompts for each field to JSON files with their metadata.

Usage:
    python export_optimized_prompts.py \
        --input optimizer_logs/optimized_all \
        --output prompts/exported_prompts \
        --config configs/fields_config.yaml




### Output JSON Format

Each output file has the following structure:

```json
{
  "field_name": "property_name",
  "type": "string",
  "matcher": "StringMatcher",
  "json_ref": "STATIC::Gen Info 1::Gen Info 1|Property Information|Property Name",
  "params": {
    "threshold": 0.85
  },
  "instructions": "Task overview\n- Input: A single field named...",
  "instructions_length": 6548,
  "score": 0.510902057815743
}
```
"""


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_fields_config(config_path: Path) -> dict[str, Any]:
    """Load the fields configuration from YAML file.

    Args:
        config_path: Path to the fields_config.yaml file

    Returns:
        Dictionary containing the fields configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    logger.info(f"Loading fields config from {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "fields" not in config:
        raise ValueError("Invalid config file: 'fields' key not found")

    logger.info(f"Loaded config with {len(config['fields'])} fields")
    return config


def load_gepa_results(field_name: str, logs_dir: Path) -> dict[str, Any]:
    """Load GEPA results for a specific field.

    Args:
        field_name: Name of the field
        logs_dir: Path to the logs directory containing field subdirectories

    Returns:
        Dictionary containing GEPA results

    Raises:
        FileNotFoundError: If GEPA results file doesn't exist
    """
    gepa_file = logs_dir / field_name / f"gepa_results_{field_name}.json"

    if not gepa_file.exists():
        raise FileNotFoundError(f"GEPA results not found for field '{field_name}': {gepa_file}")

    logger.debug(f"Loading GEPA results from {gepa_file}")

    with open(gepa_file) as f:
        results = json.load(f)

    return results


def extract_best_prompt(field_name: str, gepa_results: dict[str, Any], field_config: dict[str, Any]) -> dict[str, Any]:
    """Extract the best-performing prompt and its metadata.

    Args:
        field_name: Name of the field
        gepa_results: GEPA results dictionary
        field_config: Configuration for this field from fields_config.yaml

    Returns:
        Dictionary with field metadata and best prompt
    """
    best_idx = gepa_results.get("best_idx")

    if best_idx is None:
        raise ValueError(f"No best_idx found in GEPA results for field '{field_name}'")

    # Find the candidate with the best index
    candidates = gepa_results.get("candidate_instructions", [])
    best_candidate = None

    for candidate in candidates:
        if candidate.get("index") == best_idx:
            best_candidate = candidate
            break

    if best_candidate is None:
        raise ValueError(f"Could not find candidate with index {best_idx} for field '{field_name}'")

    # Extract field metadata from config
    field_type = field_config.get("type", "unknown")
    matcher = field_config.get("matcher", "unknown")
    json_ref = field_config.get("json_ref", "")
    params = field_config.get("params", {})

    # Build output structure
    output = {
        "field_name": field_name,
        "type": field_type,
        "matcher": matcher,
        "json_ref": json_ref,
        "params": params,
        "instructions": best_candidate.get("instructions", ""),
        "instructions_length": best_candidate.get("instruction_length", 0),
        "score": best_candidate.get("score", 0.0),
    }

    return output


def export_prompt(field_data: dict[str, Any], output_dir: Path) -> Path:
    """Export a field's prompt data to a JSON file.

    Args:
        field_data: Dictionary containing field metadata and prompt
        output_dir: Directory to save the output file

    Returns:
        Path to the created file
    """
    field_name = field_data["field_name"]
    output_file = output_dir / f"{field_name}.json"

    logger.debug(f"Writing {field_name} to {output_file}")

    with open(output_file, "w") as f:
        json.dump(field_data, f, indent=2, ensure_ascii=False)

    return output_file


def process_optimization_logs(input_dir: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Process all optimization logs and export best prompts.

    Args:
        input_dir: Path to the optimizer logs directory (e.g., optimizer_logs/optimized_all)
        config_path: Path to fields_config.yaml
        output_dir: Path to output directory for exported prompts

    Returns:
        Dictionary with processing statistics
    """
    # Load configuration
    config = load_fields_config(config_path)
    fields_config = config["fields"]

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Locate logs directory
    logs_dir = input_dir / "logs"

    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    # Statistics
    stats = {"total_fields": 0, "successful": 0, "failed": 0, "skipped": 0, "failed_fields": []}

    # Process each field in the config
    for field_name, field_config in fields_config.items():
        stats["total_fields"] += 1

        try:
            logger.info(f"Processing field: {field_name}")

            # Load GEPA results
            gepa_results = load_gepa_results(field_name, logs_dir)

            # Extract best prompt
            field_data = extract_best_prompt(field_name, gepa_results, field_config)

            # Export to file
            _ = export_prompt(field_data, output_dir)

            stats["successful"] += 1
            logger.info(
                f"✓ {field_name}: score={field_data['score']:.4f}, " f"length={field_data['instructions_length']}"
            )

        except FileNotFoundError as e:
            logger.warning(f"⊘ {field_name}: {e}")
            stats["skipped"] += 1

        except Exception as e:
            logger.error(f"✗ {field_name}: {e}")
            stats["failed"] += 1
            stats["failed_fields"].append(field_name)

    return stats


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Export optimized prompts from optimization logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from default locations
  python export_optimized_prompts.py \\
      --input optimizer_logs/optimized_all \\
      --output prompts/exported_prompts

  # Specify custom config file
  python export_optimized_prompts.py \\
      --input optimizer_logs/optimized_all \\
      --output prompts/exported_prompts \\
      --config configs/custom_fields_config.yaml
        """,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to optimizer logs directory (e.g., optimizer_logs/optimized_all)",
    )

    parser.add_argument(
        "--output", type=Path, default=None, help="Path to output directory (default: prompts/{timestamp})"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fields_config.yaml"),
        help="Path to fields config file (default: configs/fields_config.yaml)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default output directory with timestamp if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path("prompts") / timestamp

    # Validate paths
    if not args.input.exists():
        logger.error(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    if not args.config.exists():
        logger.error(f"Config file does not exist: {args.config}")
        sys.exit(1)

    # Process logs
    logger.info("=" * 80)
    logger.info("Starting prompt export")
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 80)

    try:
        stats = process_optimization_logs(args.input, args.config, args.output)

        # Print summary
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total fields:    {stats['total_fields']}")
        logger.info(f"✓ Successful:    {stats['successful']}")
        logger.info(f"⊘ Skipped:       {stats['skipped']}")
        logger.info(f"✗ Failed:        {stats['failed']}")

        if stats["failed_fields"]:
            logger.info(f"\nFailed fields: {', '.join(stats['failed_fields'])}")

        logger.info("=" * 80)
        logger.info(f"Output saved to: {args.output.absolute()}")

        # Exit with appropriate code
        if stats["failed"] > 0:
            sys.exit(1)
        elif stats["successful"] == 0:
            logger.error("No prompts were successfully exported!")
            sys.exit(1)
        else:
            logger.info("✓ Export completed successfully")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
