"""JSON list matcher."""

from typing import Any

import dspy

from scripts.matchers.base_matcher import BaseMatcher


class JSONMatcher(BaseMatcher):
    """Match JSON lists with optional schema-based field matchers."""

    def __init__(
        self,
        field_name: str,
        judge_lm: dspy.LM | None = None,
        programmatic_weight: float = 0.3,
        llm_weight: float = 0.7,
        field_schema: dict[str, dict] | None = None,
    ):
        """
        Initialize JSONMatcher.

        Args:
            field_name: Name of the field
            judge_lm: Optional LLM judge
            programmatic_weight: Weight for programmatic scoring
            llm_weight: Weight for LLM scoring
            field_schema: Optional schema dict for building field matchers.
                         Format: {"field_name": {"type": "string", ...}}
                         Example: {
                             "unit_number": {"type": "string"},
                             "start_date": {"type": "date"},
                             "amount": {"type": "float", "tolerance": 0.001},
                             "frequency": {"type": "enum", "valid_values": ["Annual", "Monthly"]}
                         }
        """
        super().__init__(field_name)
        self.judge_lm = judge_lm
        self.programmatic_weight = programmatic_weight
        self.llm_weight = llm_weight
        self.field_schema = field_schema

        # Build field matchers from schema
        self.field_matchers = self._build_field_matchers() if field_schema else None

    def _build_field_matchers(self) -> dict[str, Any]:
        """Build field matchers from schema configuration."""
        # Lazy import to avoid circular dependency
        from scripts.matchers.matcher_registry import MatcherRegistry

        if not self.field_schema:
            return {}

        matchers = {}
        for field_name, config in self.field_schema.items():
            config_copy = config.copy()  # Don't mutate original
            field_type = config_copy.pop("type")
            matchers[field_name] = MatcherRegistry.create(field_name, field_type, **config_copy)

        return matchers

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        from scripts.json_metrics import hybrid_json_score, json_match_score

        if self.judge_lm:
            score, details = hybrid_json_score(
                gold,
                pred,
                judge_lm=self.judge_lm,
                programmatic_weight=self.programmatic_weight,
                llm_weight=self.llm_weight,
                field_matchers=self.field_matchers,
            )
            prog_det = details.get("programmatic_details", {})
            if "error" in prog_det:
                return score, f"⚠ {self.field_name}: {prog_det['error']}"

            n_matched = prog_det.get("n_matched", 0)
            feedback = (
                f"{'✓' if score >= 0.95 else '✗'} {self.field_name}:\n"
                f"  Prog: {details['programmatic_score']:.2f}, "
                f"LLM: {details['llm_score']:.2f}, Final: {score:.2f}\n"
                f"  {prog_det['n_gold']}g/{prog_det['n_pred']}p/{n_matched}m"
            )
        else:
            score, details = json_match_score(gold, pred, field_matchers=self.field_matchers)
            if "error" in details:
                return score, f"⚠ {self.field_name}: {details['error']}"

            n_matched = details.get("n_matched", 0)
            feedback = (
                f"{'✓' if score >= 0.95 else '✗'} {self.field_name}:\n"
                f"  Score: {score:.2f}\n"
                f"  {details['n_gold']}g/{details['n_pred']}p/{n_matched}m"
            )

        return score, feedback
