"""Matcher-based metrics for DSPy optimization."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import dspy

from matchers import MatcherRegistry

logger = logging.getLogger(__name__)


class MatcherMetric:
    """DSPy metric using matchers from config."""

    def __init__(
        self,
        field_name: str,
        field_config: Dict[str, Any],
        judge_lm: Optional[dspy.LM] = None,
        prediction_logger: Optional[Any] = None,
        example_idx: int = 0,
    ):
        """Initialize metric from field config.

        Args:
            field_name: Name of the field
            field_config: Field configuration from YAML
            judge_lm: Optional LLM for JSON judging
            prediction_logger: Optional PredictionLogger for tracking predictions
            example_idx: Current example index (updated before each evaluation)
        """
        self.field_name = field_name
        self.field_type = field_config["type"]
        self.matcher_name = field_config.get("matcher", "StringMatcher")
        self.params = field_config.get("params", {})
        self.prediction_logger = prediction_logger

        if self.field_type == "json" and judge_lm is not None:
            self.params["judge_lm"] = judge_lm

        self.matcher = MatcherRegistry.create(field_name, self.field_type, **self.params)

    def __call__(self, gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
        """Evaluate prediction using matcher."""
        result = self.matcher(gold, pred, trace)

        if self.prediction_logger is not None:
            try:
                gold_val = self.matcher._extract(gold, self.field_name)
                pred_val = self.matcher._extract(pred, self.field_name)

                self.prediction_logger.log_prediction(
                    gold=gold_val,
                    predicted=pred_val,
                    score=result.score,
                    feedback=result.feedback,
                )
            except Exception as e:
                logger.warning(f"Failed to log prediction for {self.field_name}: {e}")

        return result.score
