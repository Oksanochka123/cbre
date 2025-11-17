#!/usr/bin/env python3
"""
Evaluation Script for Field Extraction

This script evaluates LLM predictions against ground truth using field-specific matchers.

Usage:
    python evaluate_predictions.py \\
        --ground-truth data/interim \\
        --predictions data/predictions \\
        --config configs/fields_config.yaml \\
        --output results/evaluation.csv
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from components.json_ref_resolver import JsonRefResolver
from matchers.matcher_registry import MatcherRegistry


class FieldEvaluator:
    """Evaluator for field extraction predictions."""

    def __init__(
        self,
        ground_truth_dir: Path,
        predictions_dir: Path,
        fields_config: dict,
    ):
        """Initialize the evaluator.

        Args:
            ground_truth_dir: Directory containing ground truth JSONs
            predictions_dir: Directory containing prediction JSONs
            fields_config: Fields configuration dictionary
        """
        self.ground_truth_dir = Path(ground_truth_dir)
        self.predictions_dir = Path(predictions_dir)
        self.fields_config = fields_config.get("fields", {})

        # Initialize matcher registry
        self.matchers = self._create_matchers()

        logging.info(f"Initialized evaluator with {len(self.matchers)} field matchers")

    def _create_matchers(self) -> dict:
        """Create matchers for all fields based on config.

        Returns:
            Dictionary mapping field names to matcher instances
        """
        matchers = {}

        for field_name, field_config in self.fields_config.items():
            matcher_type = field_config.get("matcher", "StringMatcher")
            field_type = field_config.get("type", "string")
            params = field_config.get("params", {})

            try:
                # Map matcher class names to types
                matcher_type_map = {
                    "StringMatcher": "string",
                    "DateMatcher": "date",
                    "NumberMatcher": "number",
                    "FloatMatcher": "float",
                    "BooleanMatcher": "boolean",
                    "EnumMatcher": "enum",
                    "PhoneMatcher": "phone",
                    "AddressMatcher": "address",
                    "JSONMatcher": "json",
                }

                type_key = matcher_type_map.get(matcher_type, field_type)
                matcher = MatcherRegistry.create(field_name, type_key, **params)
                matchers[field_name] = matcher

                logging.debug(f"Created {matcher_type} for field: {field_name}")

            except Exception as e:
                logging.warning(f"Failed to create matcher for {field_name}: {e}")
                # Fallback to string matcher
                matchers[field_name] = MatcherRegistry.create(field_name, "string")

        return matchers

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
        # Find JSON files, excluding _meta.json
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
            logging.debug(f"No json_ref for field {field_name}")
            return None

        try:
            value = JsonRefResolver.resolve(data, json_ref)
            return value
        except Exception as e:
            logging.debug(f"Error extracting {field_name} using json_ref '{json_ref}': {e}")
            return None

    def evaluate_lease(
        self,
        lease_name: str,
        ground_truth: dict,
        prediction: dict,
    ) -> dict:
        """Evaluate predictions for a single lease.

        Args:
            lease_name: Name of the lease
            ground_truth: Ground truth JSON
            prediction: Prediction JSON

        Returns:
            Dictionary of field results
        """
        results = {}

        for field_name, matcher in self.matchers.items():
            try:
                # Extract values
                gt_value = self.extract_field_value(ground_truth, field_name)
                pred_value = self.extract_field_value(prediction, field_name)

                # Handle None cases
                if gt_value is None and pred_value is None:
                    score, feedback = 1.0, "Both None"
                elif gt_value is None:
                    score, feedback = 0.0, "Hallucination (GT is None)"
                elif pred_value is None:
                    score, feedback = 0.0, "Missing (Pred is None)"
                else:
                    # Use matcher
                    score, feedback = matcher.match(gt_value, pred_value)

                results[field_name] = {
                    "score": score,
                    "feedback": feedback,
                    "gt_value": gt_value,
                    "pred_value": pred_value,
                }

                logging.debug(f"{lease_name} - {field_name}: {score:.2f} - {feedback[:50]}")

            except Exception as e:
                logging.error(f"Error evaluating {field_name} for {lease_name}: {e}")
                results[field_name] = {
                    "score": 0.0,
                    "feedback": f"Error: {str(e)}",
                    "gt_value": None,
                    "pred_value": None,
                }

        return results

    def evaluate_all(self) -> dict:
        """Evaluate all leases.

        Returns:
            Dictionary with evaluation results
        """
        logging.info("Starting evaluation...")

        # Find all lease folders in ground truth directory
        gt_folders = [f for f in self.ground_truth_dir.iterdir() if f.is_dir() and not f.name.startswith(".")]

        logging.info(f"Found {len(gt_folders)} lease folders in ground truth")

        # Accumulate results
        all_results = {}
        field_stats = defaultdict(lambda: {"total": 0, "correct": 0, "scores": []})

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

            # Evaluate this lease
            lease_results = self.evaluate_lease(lease_name, gt_data, pred_data)
            all_results[lease_name] = lease_results

            # Update field statistics
            for field_name, result in lease_results.items():
                score = result["score"]
                field_stats[field_name]["total"] += 1
                field_stats[field_name]["scores"].append(score)

                # Consider "correct" if score >= 0.95 (to account for fuzzy matching)
                if score >= 0.95:
                    field_stats[field_name]["correct"] += 1

            processed += 1
            logging.info(f"Evaluated {processed}/{len(gt_folders)}: {lease_name}")

        logging.info(f"Evaluation complete: {processed} processed, {skipped} skipped")

        return {
            "lease_results": all_results,
            "field_stats": dict(field_stats),
            "processed": processed,
            "skipped": skipped,
        }

    def generate_csv_report(self, results: dict, output_path: Path) -> None:
        """Generate CSV report with per-field metrics.

        Args:
            results: Evaluation results
            output_path: Path to save CSV file
        """
        logging.info(f"Generating CSV report: {output_path}")

        field_stats = results["field_stats"]

        # Prepare CSV data
        rows = []
        for field_name in sorted(field_stats.keys()):
            stats = field_stats[field_name]
            total = stats["total"]
            correct = stats["correct"]
            accuracy = correct / total if total > 0 else 0.0

            # Get field config
            field_config = self.fields_config.get(field_name, {})
            json_ref = field_config.get("json_ref", "")
            matcher = field_config.get("matcher", "Unknown")

            rows.append(
                {
                    "field_name": field_name,
                    "accuracy": f"{accuracy:.4f}",
                    "correct_docs": correct,
                    "total_docs": total,
                    "json_ref": json_ref,
                    "matcher": matcher,
                }
            )

        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "field_name",
                "accuracy",
                "correct_docs",
                "total_docs",
                "json_ref",
                "matcher",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logging.info(f"CSV report saved to {output_path}")

    def generate_summary_report(self, results: dict, output_path: Path) -> None:
        """Generate summary report.

        Args:
            results: Evaluation results
            output_path: Path to save summary file
        """
        logging.info(f"Generating summary report: {output_path}")

        field_stats = results["field_stats"]
        processed = results["processed"]
        skipped = results["skipped"]

        # Calculate overall statistics
        total_fields = len(field_stats)
        total_evaluations = sum(s["total"] for s in field_stats.values())
        total_correct = sum(s["correct"] for s in field_stats.values())
        overall_accuracy = total_correct / total_evaluations if total_evaluations > 0 else 0.0

        # Calculate per-field accuracies
        field_accuracies = []
        for field_name, stats in field_stats.items():
            total = stats["total"]
            correct = stats["correct"]
            accuracy = correct / total if total > 0 else 0.0
            field_accuracies.append((field_name, accuracy, correct, total))

        # Sort by accuracy
        field_accuracies.sort(key=lambda x: x[1], reverse=True)

        # Build summary text
        summary = []
        summary.append("=" * 80)
        summary.append("FIELD EXTRACTION EVALUATION SUMMARY")
        summary.append("=" * 80)
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")

        summary.append("OVERALL STATISTICS")
        summary.append("-" * 80)
        summary.append(f"Total leases processed:     {processed}")
        summary.append(f"Leases skipped:             {skipped}")
        summary.append(f"Total fields evaluated:     {total_fields}")
        summary.append(f"Total field evaluations:    {total_evaluations}")
        summary.append(f"Total correct:              {total_correct}")
        summary.append(f"Overall accuracy:           {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
        summary.append("")

        summary.append("TOP 10 BEST PERFORMING FIELDS")
        summary.append("-" * 80)
        summary.append(f"{'Field':<40} {'Accuracy':>10} {'Correct/Total':>15}")
        summary.append("-" * 80)
        for field_name, accuracy, correct, total in field_accuracies[:10]:
            summary.append(f"{field_name:<40} {accuracy:>10.4f} {f'{correct}/{total}':>15}")
        summary.append("")

        summary.append("BOTTOM 10 WORST PERFORMING FIELDS")
        summary.append("-" * 80)
        summary.append(f"{'Field':<40} {'Accuracy':>10} {'Correct/Total':>15}")
        summary.append("-" * 80)
        for field_name, accuracy, correct, total in field_accuracies[-10:]:
            summary.append(f"{field_name:<40} {accuracy:>10.4f} {f'{correct}/{total}':>15}")
        summary.append("")

        summary.append("=" * 80)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(summary))

        logging.info(f"Summary report saved to {output_path}")

        # Also log to console
        for line in summary:
            logging.info(line)


def setup_logging(log_file: Path | None = None) -> None:
    """Setup logging configuration.

    Args:
        log_file: Optional path to log file
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate field extraction predictions against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python evaluate_predictions.py \\
      --ground-truth data/interim \\
      --predictions data/predictions \\
      --output results/evaluation.csv

  # With custom config
  python evaluate_predictions.py \\
      --ground-truth data/interim \\
      --predictions data/predictions \\
      --config configs/fields_config.yaml \\
      --output results/evaluation.csv
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
        default=Path("results/evaluation.csv"),
        help="Path to output CSV file (default: results/evaluation.csv)",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to summary report file (default: same dir as CSV with .txt extension)",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to log file (default: None, logs to console only)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    logging.info("=" * 80)
    logging.info("STARTING FIELD EXTRACTION EVALUATION")
    logging.info("=" * 80)

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

    # Run evaluation
    try:
        evaluator = FieldEvaluator(
            ground_truth_dir=args.ground_truth,
            predictions_dir=args.predictions,
            fields_config=fields_config,
        )

        results = evaluator.evaluate_all()

        # Generate CSV report
        evaluator.generate_csv_report(results, args.output)

        # Generate summary report
        summary_path = args.summary or args.output.with_suffix(".txt")
        evaluator.generate_summary_report(results, summary_path)

        logging.info("=" * 80)
        logging.info("EVALUATION COMPLETE")
        logging.info("=" * 80)

        sys.exit(0)

    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
