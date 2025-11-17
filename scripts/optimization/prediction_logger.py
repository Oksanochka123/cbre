"""Prediction logging utility for tracking extraction performance."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import csv


logger = logging.getLogger(__name__)


class PredictionLogger:
    """Logs predictions during optimization for debugging and analysis."""

    def __init__(
        self,
        log_dir: Path,
        field_name: str,
        mode: str = "csv",
    ):
        """
        Initialize prediction logger.

        Args:
            log_dir: Directory to write logs
            field_name: Name of field being optimized
            mode: "csv" or "json" format
        """
        self.log_dir = Path(log_dir)
        self.field_name = field_name
        self.mode = mode
        self.prediction_count = 0

        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if mode == "csv":
            self.log_file = self.log_dir / f"predictions_{field_name}_{timestamp}.csv"
            self.csv_writer = None
            self.csv_file = None
            self._init_csv()
        else:
            self.log_file = self.log_dir / f"predictions_{field_name}_{timestamp}.jsonl"

        self.metadata_log_file = self.log_dir / f"metadata_{field_name}_{timestamp}.log"
        self.metadata_file = open(self.metadata_log_file, "w")

        self.buffer: list[Dict[str, Any]] = []

        logger.info(f"PredictionLogger initialized for {field_name}")

    def _init_csv(self):
        """Initialize CSV file with headers."""
        self.csv_file = open(self.log_file, "w", newline="", buffering=1)
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "field_name",
                "gold",
                "predicted",
                "score",
                "feedback",
                "timestamp",
            ],
        )
        self.csv_writer.writeheader()

    def log_prediction(
        self,
        gold: Any,
        predicted: Any,
        score: float,
        feedback: str,
        **kwargs,  # Accept but ignore iteration/example_idx
    ):
        """Log a single prediction."""
        gold_str = self._serialize(gold)
        pred_str = self._serialize(predicted)

        record = {
            "field_name": self.field_name,
            "gold": gold_str,
            "predicted": pred_str,
            "score": f"{score:.4f}",
            "feedback": feedback.replace("\n", " | "),
            "timestamp": datetime.now().isoformat(),
        }

        self.buffer.append(record)
        self.prediction_count += 1

        if len(self.buffer) >= 1:
            self.flush()

    def flush(self):
        """Write buffered predictions to log file."""
        if not self.buffer:
            return

        if self.mode == "csv":
            for record in self.buffer:
                self.csv_writer.writerow(record)
            self.csv_file.flush()
        else:
            with open(self.log_file, "a") as f:
                for record in self.buffer:
                    f.write(json.dumps(record) + "\n")

        logger.debug(f"Flushed {len(self.buffer)} predictions to {self.log_file.name}")
        self.buffer = []

    def write_initial_signature(self, signature_description: str):
        """Write initial signature to metadata log file.

        Args:
            signature_description: Description of the initial signature
        """
        if self.metadata_file:
            self.metadata_file.write(f"{'='*70}\n")
            self.metadata_file.write(f"INITIAL SIGNATURE for {self.field_name}\n")
            self.metadata_file.write(f"{'='*70}\n\n")
            self.metadata_file.write(signature_description)
            self.metadata_file.write(f"\n{'='*70}\n")
            self.metadata_file.flush()
            logger.info(f"Initial signature written to {self.metadata_log_file.name}")

    @staticmethod
    def _serialize(value: Any) -> str:
        """Serialize value to string for logging."""
        if value is None:
            return "None"

        if isinstance(value, str):
            return value[:500]  # Truncate long strings

        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value)[:500]  # Truncate JSON
            except:
                return str(value)[:500]

        return str(value)[:500]

    def close(self):
        """Close log file and flush remaining data."""
        self.flush()
        if self.csv_file:
            self.csv_file.close()
        if self.metadata_file:
            self.metadata_file.close()
        logger.info(f"Closed log files: {self.log_file}, {self.metadata_log_file}")

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        self.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics from log file."""
        if self.mode != "csv":
            return {}

        # Read log file
        scores = []
        pass_count = 0
        fail_count = 0

        try:
            with open(self.log_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        score = float(row.get("score", 0))
                        scores.append(score)
                        if score >= 0.95:
                            pass_count += 1
                        else:
                            fail_count += 1
                    except ValueError:
                        pass
        except FileNotFoundError:
            return {}

        if not scores:
            return {}

        return {
            "total_predictions": len(scores),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": pass_count / len(scores) if scores else 0,
        }


class MultiFieldPredictionLogger:
    """
    Manages multiple PredictionLogger instances for multiple fields.
    Useful for tracking all fields during optimization.
    """

    def __init__(self, log_dir: Path, field_names: list[str], mode: str = "csv"):
        """
        Initialize multi-field logger.

        Args:
            log_dir: Base directory for all logs
            field_names: List of field names to track
            mode: "csv" or "json"
        """
        self.log_dir = Path(log_dir)
        self.field_names = field_names
        self.mode = mode
        self.loggers: Dict[str, PredictionLogger] = {}

        # Create logger for each field
        for field_name in field_names:
            self.loggers[field_name] = PredictionLogger(
                log_dir=self.log_dir / field_name,
                field_name=field_name,
                mode=mode,
            )

        logger.info(f"MultiFieldPredictionLogger initialized for {len(field_names)} fields")

    def log_prediction(
        self,
        field_name: str,
        gold: Any,
        predicted: Any,
        score: float,
        feedback: str,
        **kwargs,
    ):
        """Log prediction for a specific field."""
        if field_name in self.loggers:
            self.loggers[field_name].log_prediction(
                gold=gold,
                predicted=predicted,
                score=score,
                feedback=feedback,
                **kwargs,
            )

    def flush_all(self):
        """Flush all loggers."""
        for logger in self.loggers.values():
            logger.flush()

    def close_all(self):
        """Close all loggers."""
        for logger in self.loggers.values():
            logger.close()

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all fields."""
        return {
            field_name: logger.get_statistics()
            for field_name, logger in self.loggers.items()
        }

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit."""
        self.close_all()
