"""Evaluator for optimized field extraction programs."""

from typing import Any, Dict, List, Optional
import logging

import dspy
import numpy as np

from optimization.metrics import MatcherMetric

logger = logging.getLogger(__name__)


class FieldEvaluator:
    """Evaluator for optimized field extraction programs."""

    def __init__(self, field_name: str, field_config: Dict[str, Any], judge_lm: Optional[dspy.LM] = None):
        """Initialize evaluator.
        
        Args:
            field_name: Name of the field
            field_config: Field configuration from YAML
            judge_lm: Optional LLM for JSON judging
        """
        self.field_name = field_name
        self.field_type = field_config["type"]
        self.metric = MatcherMetric(field_name, field_config, judge_lm=judge_lm)

    def evaluate(self, program: dspy.Module, testset: List[dspy.Example], verbose: bool = True) -> Dict[str, Any]:
        """Evaluate optimized program on test set.
        
        Args:
            program: Optimized DSPy module
            testset: Test examples
            verbose: Whether to print detailed feedback
            
        Returns:
            Dictionary with evaluation metrics and results
        """
        scores = []
        feedbacks = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating {self.field_name} on {len(testset)} examples...")
            print(f"{'='*60}\n")

        for idx, example in enumerate(testset):
            try:
                pred = program(document_text=example.document_text)
                result = self.metric(example, pred)
                scores.append(result.score)
                feedbacks.append(result.feedback)

                if verbose and result.score < 1.0:
                    print(f"\nExample {idx + 1}:")
                    print(result.feedback)
                    print("-" * 60)

            except Exception as e:
                import traceback
                scores.append(0.0)
                error_msg = f"Error: {e}"
                error_detail = traceback.format_exc()
                feedbacks.append(error_msg)
                # Always log errors, not just when verbose
                logger.error(f"Example {idx + 1} evaluation failed: {error_msg}\n{error_detail}")
                if verbose:
                    print(f"\nExample {idx + 1}: {error_msg}")
                    print("-" * 60)

        # Compute statistics
        results = {
            "field_name": self.field_name,
            "field_type": self.field_type,
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "perfect": sum(1 for s in scores if s >= 0.99),
            "good": sum(1 for s in scores if s >= 0.8),
            "moderate": sum(1 for s in scores if 0.5 <= s < 0.8),
            "poor": sum(1 for s in scores if s < 0.5),
            "total": len(scores),
            "scores": scores,
            "feedbacks": feedbacks,
        }

        if verbose:
            self._print_summary(results)

        return results

    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY: {self.field_name}")
        print(f"{'='*60}")
        print(f"Mean Score:    {results['mean']:.3f}")
        print(f"Median Score:  {results['median']:.3f}")
        print(f"Std Dev:       {results['std']:.3f}")
        print(f"Min/Max:       {results['min']:.3f} / {results['max']:.3f}")
        print(f"\nScore Distribution:")
        print(
            f"  Perfect (≥0.99):  {results['perfect']}/{results['total']} "
            f"({100*results['perfect']/results['total']:.1f}%)"
        )
        print(
            f"  Good (≥0.80):     {results['good']}/{results['total']} "
            f"({100*results['good']/results['total']:.1f}%)"
        )
        print(
            f"  Moderate (0.50-0.80): {results['moderate']}/{results['total']} "
            f"({100*results['moderate']/results['total']:.1f}%)"
        )
        print(
            f"  Poor (<0.50):     {results['poor']}/{results['total']} "
            f"({100*results['poor']/results['total']:.1f}%)"
        )
        print(f"{'='*60}\n")