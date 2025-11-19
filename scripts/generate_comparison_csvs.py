"""
Generate Comparison CSV Tables for Field Extraction

This script generates individual CSV files for each lease agreement,
comparing ground truth values with predicted values for all fields.

Usage:
    python generate_comparison_csvs.py \
        --ground-truth data/interim \
        --predictions data/predictions/predictions_1 \
        --config configs/fields_config.yaml \
        --output results/comparisons
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import yaml
from components.json_ref_resolver import JsonRefResolver


class ComparisonCSVGenerator:
    """Generator for comparison CSV tables."""

    def __init__(
        self,
        ground_truth_dir: Path,
        predictions_dir: Path,
        fields_config: dict,
        output_dir: Path,
    ):
        """Initialize the generator.

        Args:
            ground_truth_dir: Directory containing ground truth JSONs
            predictions_dir: Directory containing prediction JSONs
            fields_config: Fields configuration dictionary
            output_dir: Directory to save CSV files
        """
        self.ground_truth_dir = Path(ground_truth_dir)
        self.predictions_dir = Path(predictions_dir)
        self.fields_config = fields_config.get("fields", {})
        self.output_dir = Path(output_dir)

    def load_json_file(self, file_path: Path) -> dict | None:
        """Load a JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data or None if failed
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading JSON from {file_path}: {e}")
            return None

    def find_ground_truth_json(self, lease_folder: Path) -> Path | None:
        """Find ground truth JSON in a lease folder.

        Args:
            lease_folder: Path to lease subfolder

        Returns:
            Path to ground truth JSON or None
        """
        json_files = [f for f in lease_folder.glob("*.json") if not f.name.endswith("_meta.json")]

        if not json_files:
            return None

        if len(json_files) == 1:
            return json_files[0]

        # Multiple files - prefer most recent
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return json_files[0]

    def extract_field_value(self, data: dict, field_name: str) -> any:
        """Extract field value from JSON using json_ref.

        Args:
            data: JSON data (ground truth or prediction)
            field_name: Name of the field

        Returns:
            Extracted value or None
        """
        field_config = self.fields_config.get(field_name, {})
        json_ref = field_config.get("json_ref", "")

        if not json_ref or json_ref == "MISSING":
            return None

        try:
            return JsonRefResolver.resolve(data, json_ref)
        except Exception as e:
            logging.debug(f"Error extracting {field_name}: {e}")
            return None

    def format_value(self, value: any) -> str:
        """Format a value for CSV output.

        Args:
            value: Value to format

        Returns:
            Formatted string representation
        """
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def generate_lease_csv(
        self,
        lease_name: str,
        ground_truth: dict,
        prediction: dict,
    ) -> list[dict]:
        """Generate comparison data for a single lease.

        Args:
            lease_name: Name of the lease
            ground_truth: Ground truth JSON
            prediction: Prediction JSON

        Returns:
            List of row dictionaries for CSV
        """
        rows = []

        for field_name in sorted(self.fields_config.keys()):
            gt_value = self.extract_field_value(ground_truth, field_name)
            pred_value = self.extract_field_value(prediction, field_name)

            rows.append(
                {
                    "field_name": field_name,
                    "ground_truth": self.format_value(gt_value),
                    "predicted": self.format_value(pred_value),
                }
            )

        return rows

    def sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename.

        Args:
            name: Original name

        Returns:
            Sanitized filename
        """
        # Replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        sanitized = name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, "_")
        return sanitized

    def generate_all(self) -> tuple[int, int]:
        """Generate CSV files for all leases.

        Returns:
            Tuple of (processed count, skipped count)
        """
        logging.info("Starting CSV generation...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Find all lease folders in ground truth directory
        gt_folders = [f for f in self.ground_truth_dir.iterdir() if f.is_dir() and not f.name.startswith(".")]

        logging.info(f"Found {len(gt_folders)} lease folders")

        processed = 0
        skipped = 0

        for gt_folder in gt_folders:
            lease_name = gt_folder.name

            # Find ground truth JSON
            gt_json_path = self.find_ground_truth_json(gt_folder)
            if not gt_json_path:
                logging.warning(f"No ground truth JSON for {lease_name}, skipping")
                skipped += 1
                continue

            # Find prediction JSON
            pred_json_path = self.predictions_dir / lease_name / "predicted_fields.json"
            if not pred_json_path.exists():
                logging.warning(f"No prediction JSON for {lease_name}, skipping")
                skipped += 1
                continue

            # Load both JSONs
            gt_data = self.load_json_file(gt_json_path)
            pred_data = self.load_json_file(pred_json_path)

            if gt_data is None or pred_data is None:
                logging.error(f"Failed to load data for {lease_name}, skipping")
                skipped += 1
                continue

            # Generate comparison data
            rows = self.generate_lease_csv(lease_name, gt_data, pred_data)

            # Write CSV file
            safe_filename = self.sanitize_filename(lease_name)
            output_path = self.output_dir / f"{safe_filename}.csv"

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["field_name", "ground_truth", "predicted"])
                writer.writeheader()
                writer.writerows(rows)

            processed += 1
            logging.info(f"Generated CSV {processed}/{len(gt_folders)}: {lease_name}")

        logging.info(f"Generation complete: {processed} processed, {skipped} skipped")

        return processed, skipped


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate comparison CSV tables for field extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_comparison_csvs.py \\
      --ground-truth data/interim \\
      --predictions data/predictions/predictions_1 \\
      --output results/comparisons

  # With custom config
  python generate_comparison_csvs.py \\
      --ground-truth data/interim \\
      --predictions data/predictions/predictions_1 \\
      --config configs/fields_config.yaml \\
      --output results/comparisons
        """,
    )

    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/interim"),
        help="Path to ground truth directory (default: data/interim)",
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions directory",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/fields_config.yaml"),
        help="Path to fields config file (default: configs/fields_config.yaml)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/comparisons"),
        help="Path to output directory for CSV files (default: results/comparisons)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logging.info("=" * 60)
    logging.info("GENERATING COMPARISON CSV TABLES")
    logging.info("=" * 60)

    # Validate inputs
    if not args.ground_truth.exists():
        logging.error(f"Ground truth directory not found: {args.ground_truth}")
        sys.exit(1)

    if not args.predictions.exists():
        logging.error(f"Predictions directory not found: {args.predictions}")
        sys.exit(1)

    if not args.config.exists():
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load config
    with open(args.config, encoding="utf-8") as f:
        fields_config = yaml.safe_load(f)

    # Run generation
    try:
        generator = ComparisonCSVGenerator(
            ground_truth_dir=args.ground_truth,
            predictions_dir=args.predictions,
            fields_config=fields_config,
            output_dir=args.output,
        )

        processed, skipped = generator.generate_all()

        logging.info("=" * 60)
        logging.info(f"GENERATION COMPLETE: {processed} CSV files created")
        logging.info(f"Output directory: {args.output}")
        logging.info("=" * 60)

        sys.exit(0)

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
