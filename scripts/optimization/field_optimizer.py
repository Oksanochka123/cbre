"""Field optimizer using matcher-based metrics and GEPA."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy

from optimization.metrics import MatcherMetric
from optimization.optimizer_utils import dspy_logging, handle_tied_candidates, save_gepa_results
from optimization.signature_utils import create_signature
from optimization.prediction_logger import PredictionLogger

logger = logging.getLogger(__name__)


class FieldOptimizer:
    """Optimizer for a single field using matcher-based metrics."""

    def __init__(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        reflection_lm: Optional[dspy.LM] = None,
        signature: Optional[type] = None,
        enable_prediction_logging: bool = True,
    ):
        """Initialize optimizer for a single field.

        Args:
            field_name: Name of the field
            field_config: Field configuration from YAML
            reflection_lm: LM for GEPA reflection (also used as judge_lm for JSON)
            signature: Optional custom signature
            enable_prediction_logging: Whether to log predictions during optimization
        """
        self.field_name = field_name
        self.field_config = field_config
        self.field_type = field_config["type"]
        self.signature = signature
        self.reflection_lm = reflection_lm
        self.enable_prediction_logging = enable_prediction_logging
        self.prediction_logger = None

        # Build metric (will add logger later if needed)
        self.metric = MatcherMetric(field_name, field_config, judge_lm=reflection_lm)

    def optimize(
        self,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None,
        program: Optional[dspy.Module] = None,
        enable_logging: bool = True,
        log_dir: str = "./data/interim",
        **optimizer_kwargs,
    ) -> dspy.Module:
        """Optimize field extraction using GEPA.

        Args:
            trainset: Training examples
            valset: Validation examples
            program: Optional pre-built program (default: creates ChainOfThought)
            enable_logging: Whether to enable DSPy logging
            log_dir: Directory for logs and results
            **optimizer_kwargs: Additional kwargs for GEPA

        Returns:
            Optimized DSPy module
        """
        # Create program if not provided
        if program is None:
            if self.signature is None:
                # Extract valid_values for enum fields
                valid_values = None
                if self.field_type == "enum" and "params" in self.field_config:
                    valid_values = self.field_config["params"].get("valid_values")

                self.signature = create_signature(
                    self.field_name,
                    self.field_type,
                    trainset,
                    valid_values=valid_values,
                )
            program = dspy.ChainOfThought(self.signature)

        print(f"\n{'='*60}")
        print(f"Optimizing: {self.field_name} ({self.field_type})")
        print(f"Train: {len(trainset)}, Val: {len(valset) if valset else 0}")
        print(f"{'='*60}\n")

        # Build signature description
        signature_description = ""
        if hasattr(self.signature, '_output_fields'):
            for field_name_out, field_obj in self.signature._output_fields.items():
                if hasattr(field_obj, 'desc') and field_obj.desc:
                    signature_description += f"\nField: {field_name_out}\n"
                    signature_description += f"Description:\n{field_obj.desc}\n"
        elif hasattr(self.signature, '__dict__'):
            signature_description += f"Signature: {self.signature.__dict__}\n"

        # Log initial signature to DSPy logger
        logger.info(f"\n{'='*70}")
        logger.info(f"INITIAL SIGNATURE for {self.field_name}:")
        logger.info(f"{'='*70}")
        logger.info(signature_description)
        logger.info(f"{'='*70}\n")

        # Initialize prediction logger if enabled
        # Save predictions in the same log directory as other logs
        log_dir_path = Path(log_dir)
        if self.enable_prediction_logging:
            self.prediction_logger = PredictionLogger(
                log_dir=log_dir_path,  # Save directly in log_dir, not in subdirectory
                field_name=self.field_name,
                mode="csv",
            )
            # Write initial signature to prediction logger's metadata file
            self.prediction_logger.write_initial_signature(signature_description)
            # Update metric with logger
            self.metric.prediction_logger = self.prediction_logger
            print(f"Prediction logging enabled: {self.prediction_logger.log_file}")
            print(f"Metadata log: {self.prediction_logger.metadata_log_file}")

        # Setup GEPA with defaults
        gepa_kwargs = {
            "auto": "light",
            "metric": self.metric,
            "track_stats": True,
            "track_best_outputs": True,
            "seed": 42,
        }

        if self.reflection_lm is not None:
            gepa_kwargs["reflection_lm"] = self.reflection_lm

        # Override with user-provided kwargs
        gepa_kwargs.update(optimizer_kwargs)

        optimizer = dspy.GEPA(**gepa_kwargs)

        # Run optimization
        if enable_logging:
            with dspy_logging(log_dir):
                optimized = optimizer.compile(student=program, trainset=trainset, valset=valset)
        else:
            optimized = optimizer.compile(student=program, trainset=trainset, valset=valset)

        # Close prediction logger
        if self.prediction_logger:
            self.prediction_logger.close()
            stats = self.prediction_logger.get_statistics()
            print(f"\nPrediction Log Statistics:")
            print(f"Total Predictions: {stats.get('total_predictions', 0)}")
            print(f"Average Score: {stats.get('avg_score', 0):.4f}")
            print(f"Pass Rate (≥0.95): {stats.get('pass_rate', 0)*100:.1f}%")

        # Handle tied candidates and save results
        if hasattr(optimized, "detailed_results"):
            results = optimized.detailed_results

            # Handle ties
            optimized, num_tied, best_score = handle_tied_candidates(optimized, results)

            # Save GEPA results
            results_path = log_dir_path / f"gepa_results_{self.field_name}.json"
            save_gepa_results(results, results_path, self.field_name, self.field_type)
            print(f"GEPA results saved to: {results_path}")

        print(f"Optimization complete for {self.field_name}\n")

        # Save optimized program
        program_path = log_dir_path / f"optimized_{self.field_name}.json"
        program_path.parent.mkdir(parents=True, exist_ok=True)
        optimized.save(str(program_path))
        print(f"Optimized program saved to: {program_path}")

        return optimized