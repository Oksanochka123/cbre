"""Matcher-based metrics for DSPy optimization."""

from typing import Any, Dict, Optional

import dspy

from matchers import MatcherRegistry


class MatcherMetric:
    """DSPy metric using matchers from config."""

    def __init__(self, field_name: str, field_config: Dict[str, Any], judge_lm: Optional[dspy.LM] = None):
        """Initialize metric from field config.

        Args:
            field_name: Name of the field
            field_config: Field configuration from YAML
            judge_lm: Optional LLM for JSON judging
        """
        self.field_name = field_name
        self.field_type = field_config["type"]
        self.matcher_name = field_config.get("matcher", "StringMatcher")
        self.params = field_config.get("params", {})

        # Pass judge_lm to JSON matchers
        if self.field_type == "json" and judge_lm is not None:
            self.params["judge_lm"] = judge_lm

        # Create matcher
        self.matcher = MatcherRegistry.create(field_name, self.field_type, **self.params)

    def __call__(self, example: dspy.Example, pred: dspy.Prediction, trace=None) -> dspy.Prediction:
        """Evaluate prediction using matcher."""
        return self.matcher(example, pred, trace)