"""JSON list matcher."""

from typing import Any

import dspy

from scripts.matchers.base_matcher import BaseMatcher


class JSONMatcher(BaseMatcher):
    """Match JSON lists."""

    def __init__(
        self,
        field_name: str,
        judge_lm: dspy.LM | None = None,
        programmatic_weight: float = 0.3,
        llm_weight: float = 0.7,
        **kwargs,
    ):
        super().__init__(field_name, **kwargs)
        self.judge_lm = judge_lm
        self.programmatic_weight = programmatic_weight
        self.llm_weight = llm_weight

    def match(self, gold: Any, pred: Any) -> tuple[float, str]:
        from scripts.json_metrics import hybrid_json_score, json_match_score

        if self.judge_lm:
            score, details = hybrid_json_score(
                gold,
                pred,
                judge_lm=self.judge_lm,
                programmatic_weight=self.programmatic_weight,
                llm_weight=self.llm_weight,
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
            score, details = json_match_score(gold, pred)
            if "error" in details:
                return score, f"⚠ {self.field_name}: {details['error']}"

            n_matched = details.get("n_matched", 0)
            feedback = (
                f"{'✓' if score >= 0.95 else '✗'} {self.field_name}:\n"
                f"  Score: {score:.2f}\n"
                f"  {details['n_gold']}g/{details['n_pred']}p/{n_matched}m"
            )

        return score, feedback
