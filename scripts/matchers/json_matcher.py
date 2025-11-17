"""JSON list matcher."""

from typing import Any

import dspy

from matchers.base_matcher import BaseMatcher


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
        from matchers.matcher_registry import MatcherRegistry

        if not self.field_schema:
            return {}

        matchers = {}
        for field_name, config in self.field_schema.items():
            config_copy = config.copy()  # Don't mutate original
            field_type = config_copy.pop("type")
            matchers[field_name] = MatcherRegistry.create(field_name, field_type, **config_copy)

        return matchers

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        from json_metrics import hybrid_json_score, json_match_score, parse_json_safe

        # Parse data for value-showing feedback
        gold_parsed = parse_json_safe(gold)
        pred_parsed = parse_json_safe(pred)

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

            feedback = self._build_detailed_feedback(score, details, prog_det)
        else:
            score, details = json_match_score(gold, pred, field_matchers=self.field_matchers)
            if "error" in details:
                return score, f"ERROR {self.field_name}: {details['error']}"

            feedback = self._build_detailed_feedback(score, details, details)

        return score, feedback

    def _build_detailed_feedback(self, score: float, full_details: dict, prog_details: dict) -> str:
        """Build concise, actionable feedback for optimization."""

        n_gold = prog_details.get("n_gold", 0)
        n_pred = prog_details.get("n_pred", 0)
        n_matched = prog_details.get("n_matched", 0)

        status = "PASS" if score >= 0.95 else "FAIL"
        lines = [f"{status} {self.field_name}: {score:.3f}"]

        # Add LLM reasoning if available (from hybrid scoring)
        llm_reasoning = full_details.get("llm_reasoning")
        if llm_reasoning and llm_reasoning != "No LLM judge provided":
            lines.append(f"  LLM: {llm_reasoning}")

        # Record count summary
        lines.append(f"  Records: {n_gold} expected, {n_pred} extracted, {n_matched} matched")

        # Count issues
        if n_gold != n_pred:
            diff = n_pred - n_gold
            if diff < 0:
                lines.append(f"  Missing {-diff} records - extract all table rows")
            else:
                lines.append(f"  Hallucinated {diff} extra records - verify table boundaries")

        # Field-level details
        record_details = prog_details.get("record_details", [])
        if record_details and n_matched > 0:
            matched_rec = next((r for r in record_details if r.get("similarity", 0) > 0), None)

            if matched_rec and "details" in matched_rec:
                field_scores = matched_rec["details"].get("field_scores", {})
                failing = [(f, s) for f, s in field_scores.items() if s < 0.8]

                if failing:
                    failing.sort(key=lambda x: x[1])
                    lines.append(f"  Failing fields ({len(failing)}/{len(field_scores)}):")

                    # Show top 8 worst fields
                    for field, fscore in failing[:8]:
                        lines.append(f"    - {field}: {fscore:.2f}")

                    if len(failing) > 8:
                        lines.append(f"    - and {len(failing) - 8} more")

        # Unmatched records
        if n_matched < n_gold:
            lines.append(f"  Unmatched gold records: {n_gold - n_matched}")
        if n_matched < n_pred:
            lines.append(f"  Unmatched pred records: {n_pred - n_matched} (hallucinations)")

        # Priority actions
        actions = []
        if n_pred < n_gold:
            actions.append(f"1. Extract all {n_gold} rows (missing {n_gold - n_pred})")
        elif n_pred > n_gold:
            actions.append(f"1. Remove {n_pred - n_gold} extra rows")

        if record_details and n_matched > 0:
            matched_rec = next((r for r in record_details if r.get("similarity", 0) > 0), None)
            if matched_rec and "details" in matched_rec:
                field_scores = matched_rec["details"].get("field_scores", {})
                failing = [(f, s) for f, s in field_scores.items() if s < 0.8]
                if failing:
                    worst_field, worst_score = failing[0]
                    actions.append(f"2. Fix '{worst_field}' field (score {worst_score:.2f})")

        if score < 0.8:
            actions.append("3. Verify schema field names match exactly")

        if actions:
            lines.append("  Actions: " + " | ".join(actions))

        return "\n".join(lines)