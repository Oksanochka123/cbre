"""Field optimizer using matcher-based metrics and GEPA."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy

from scripts.metrics import MatcherMetric
from scripts.optimizer_utils import dspy_logging, handle_tied_candidates, save_gepa_results
from scripts.signature_utils import create_signature


class FieldOptimizer:
    """Optimizer for a single field using matcher-based metrics."""

    def __init__(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        reflection_lm: Optional[dspy.LM] = None,
        signature: Optional[type] = None,
    ):
        """Initialize optimizer for a single field.

        Args:
            field_name: Name of the field
            field_config: Field configuration from YAML
            reflection_lm: LM for GEPA reflection (also used as judge_lm for JSON)
            signature: Optional custom signature
        """
        self.field_name = field_name
        self.field_config = field_config
        self.field_type = field_config["type"]
        self.signature = signature
        self.reflection_lm = reflection_lm

        # Build metric
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
                self.signature = create_signature(self.field_name, self.field_type, trainset)
            program = dspy.ChainOfThought(self.signature)

        print(f"\n{'='*60}")
        print(f"Optimizing: {self.field_name} ({self.field_type})")
        print(f"Train: {len(trainset)}, Val: {len(valset) if valset else 0}")
        print(f"{'='*60}\n")

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

        # Handle tied candidates and save results
        if hasattr(optimized, "detailed_results"):
            results = optimized.detailed_results

            # Handle ties
            optimized, num_tied, best_score = handle_tied_candidates(optimized, results)

            # Save GEPA results
            results_path = Path(log_dir) / f"gepa_results_{self.field_name}.json"
            save_gepa_results(results, results_path, self.field_name, self.field_type)
            print(f"✓ GEPA results saved to: {results_path}")

        print(f"✓ Optimization complete for {self.field_name}\n")

        # Save optimized program
        program_path = Path(log_dir) / f"optimized_{self.field_name}.json"
        program_path.parent.mkdir(parents=True, exist_ok=True)
        optimized.save(str(program_path))
        print(f"✓ Optimized program saved to: {program_path}")

        return optimized