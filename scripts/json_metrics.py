import json
from difflib import SequenceMatcher
from typing import Any

import dspy
import numpy as np
from scipy.optimize import linear_sum_assignment

# ============================================================================
# UTILITIES
# ============================================================================


def parse_json_safe(json_str: Any) -> list[dict] | None:
    """Parse JSON string to list of dicts. Returns None on failure."""
    if json_str is None:
        return None

    # Already parsed
    if isinstance(json_str, list):
        return json_str

    # Parse string
    try:
        parsed = json.loads(str(json_str))
        # Ensure it's a list
        if isinstance(parsed, dict):
            return [parsed]
        return parsed if isinstance(parsed, list) else None
    except (json.JSONDecodeError, ValueError, TypeError):
        # Try to handle Python syntax (single quotes, None instead of null)
        try:
            import ast

            parsed = ast.literal_eval(str(json_str))
            # Ensure it's a list
            if isinstance(parsed, dict):
                return [parsed]
            return parsed if isinstance(parsed, list) else None
        except (ValueError, SyntaxError):
            return None


def normalize_value(val: Any) -> str:
    """Normalize value for comparison."""
    if val is None:
        return ""
    return str(val).strip().lower()


# ============================================================================
# FIELD-LEVEL COMPARISON
# ============================================================================


def simple_field_match(gold_val: Any, pred_val: Any) -> float:
    """
    Simple exact field comparison (used when no field_matchers provided).
    Returns 1.0 for exact match after normalization, 0.0 otherwise.
    """
    # Both None: perfect match
    if gold_val is None and pred_val is None:
        return 1.0

    # One is None: complete failure
    if gold_val is None or pred_val is None:
        return 0.0

    # Normalize for comparison
    g_norm = normalize_value(gold_val)
    p_norm = normalize_value(pred_val)

    # Exact match after normalization
    return 1.0 if g_norm == p_norm else 0.0


def semantic_field_match(gold_val: Any, pred_val: Any) -> float:
    """
    Compare two field values with semantic awareness.
    Returns score in [0, 1].

    Handles:
    - None values
    - Type coercion (string "1" vs int 1)
    - Numeric similarity
    - String similarity

    Does NOT handle:
    - Boolean-to-int coercion (true -> 1) - agent should fix this
    """
    # Both None: perfect match
    if gold_val is None and pred_val is None:
        return 1.0

    # One is None: complete failure
    if gold_val is None or pred_val is None:
        return 0.0

    # Normalize for comparison
    g_norm = normalize_value(gold_val)
    p_norm = normalize_value(pred_val)

    # Exact match after normalization
    if g_norm == p_norm:
        return 1.0

    # Try numeric comparison (handles "1" vs 1, but NOT true vs 1)
    try:
        g_num = float(g_norm)
        p_num = float(p_norm)

        # Exact numeric match
        if abs(g_num - p_num) < 1e-6:
            return 1.0

        # Similarity based on relative error
        rel_err = abs(g_num - p_num) / max(abs(g_num), 1e-9)
        return max(0.0, np.exp(-rel_err * 5))  # Exponential decay
    except (ValueError, AttributeError):
        pass

    # String similarity as last resort (heavily penalized)
    similarity = SequenceMatcher(None, g_norm, p_norm).ratio()
    return similarity * 0.5  # Max 0.5 for partial string matches


def evaluate_record_pair(
    gold_rec: dict, pred_rec: dict, field_matchers: dict[str, Any] | None = None
) -> tuple[float, dict]:
    """
    Evaluate match between a single gold-pred record pair.

    Args:
        gold_rec: Ground truth record
        pred_rec: Predicted record
        field_matchers: Optional dict of {field_name: matcher} for compositional matching.
                       Each matcher should have a .match(gold, pred) -> (score, feedback) method.
                       If provided, ALL fields must have a matcher defined (will raise ValueError if missing).

    Returns:
        score: float [0, 1]
        details: dict with field-level breakdown

    Raises:
        ValueError: If field_matchers is provided but a field is missing a matcher
    """
    all_keys = set(gold_rec.keys()) | set(pred_rec.keys())

    if not all_keys:
        return 1.0, {"field_scores": {}, "avg_score": 1.0}

    field_scores = {}
    for key in all_keys:
        gold_val = gold_rec.get(key)
        pred_val = pred_rec.get(key)

        # Use compositional matcher if provided
        if field_matchers is not None:
            if key not in field_matchers:
                raise ValueError(
                    f"Field '{key}' is missing from field_matchers. "
                    f"When using field_matchers, ALL fields must have a matcher defined. "
                    f"Available matchers: {list(field_matchers.keys())}"
                )
            matcher = field_matchers[key]
            score, _ = matcher.match(gold_val, pred_val)
            field_scores[key] = score
        else:
            # Fallback to simple exact matching when no matchers provided
            field_scores[key] = simple_field_match(gold_val, pred_val)

    avg_score = sum(field_scores.values()) / len(field_scores)

    details = {
        "field_scores": field_scores,
        "avg_score": avg_score,
        "perfect_fields": sum(1 for s in field_scores.values() if s >= 0.99),
        "failed_fields": sum(1 for s in field_scores.values() if s == 0.0),
        "total_fields": len(field_scores),
    }

    return avg_score, details


# ============================================================================
# RECORD MATCHING (Hungarian Algorithm)
# ============================================================================


def compute_record_similarity(gold_rec: dict, pred_rec: dict, field_matchers: dict[str, Any] | None = None) -> float:
    """Compute similarity between two records (used for matching)."""
    score, _ = evaluate_record_pair(gold_rec, pred_rec, field_matchers)
    return score


def match_records_hungarian(
    gold_records: list[dict], pred_records: list[dict], field_matchers: dict[str, Any] | None = None
) -> list[tuple[int, int, float]]:
    """
    Match gold and predicted records using Hungarian algorithm.

    Args:
        gold_records: List of ground truth records
        pred_records: List of predicted records
        field_matchers: Optional dict of {field_name: matcher} for compositional matching

    Returns:
        List of (gold_idx, pred_idx, similarity_score) tuples
    """
    n_gold = len(gold_records)
    n_pred = len(pred_records)

    if n_gold == 0 or n_pred == 0:
        return []

    # Build cost matrix (negative similarity for minimization)
    cost_matrix = np.zeros((n_gold, n_pred))
    for i, gold_rec in enumerate(gold_records):
        for j, pred_rec in enumerate(pred_records):
            similarity = compute_record_similarity(gold_rec, pred_rec, field_matchers)
            cost_matrix[i, j] = -similarity  # Negative for minimization

    # Solve assignment problem
    gold_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # Build matches with scores
    matches = []
    for i, j in zip(gold_indices, pred_indices, strict=False):
        similarity = -cost_matrix[i, j]  # Convert back to positive
        matches.append((i, j, similarity))

    return matches


# ============================================================================
# MAIN JSON METRIC
# ============================================================================


def json_match_score(
    ground_truth: Any,
    predicted: Any,
    count_weight: float = 0.4,
    match_weight: float = 0.6,
    field_matchers: dict[str, Any] | None = None,
) -> tuple[float, dict]:
    """
    Compare JSON lists of records with proper record matching.

    Scoring:
    1. Count penalty (quadratic): penalizes record count mismatch
    2. Record matching: uses Hungarian algorithm to match records
    3. Field-level comparison: evaluates matched record pairs

    Args:
        ground_truth: JSON string or parsed list of dicts
        predicted: JSON string or parsed list of dicts
        count_weight: Weight for record count component (default: 0.4)
        match_weight: Weight for record matching component (default: 0.6)

    Returns:
        score: float [0, 1]
        details: dict with analysis
    """
    gold_records = parse_json_safe(ground_truth)
    pred_records = parse_json_safe(predicted)

    # Parse errors
    if gold_records is None and pred_records is None:
        return 1.0, {"error": "Both failed to parse"}

    if gold_records is None:
        return 0.0, {"error": "Ground truth failed to parse"}

    if pred_records is None:
        return 0.0, {"error": "Prediction failed to parse"}

    n_gold = len(gold_records)
    n_pred = len(pred_records)

    # Empty case
    if n_gold == 0 and n_pred == 0:
        return 1.0, {"count_score": 1.0, "match_score": 1.0, "final_score": 1.0, "n_gold": 0, "n_pred": 0}

    # Empty prediction when should have data
    if n_pred == 0 and n_gold > 0:
        return 0.0, {
            "error": "Empty prediction",
            "count_score": 0.0,
            "match_score": 0.0,
            "final_score": 0.0,
            "n_gold": n_gold,
            "n_pred": 0,
        }

    # Empty gold (shouldn't happen, but handle gracefully)
    if n_gold == 0 and n_pred > 0:
        return 0.0, {
            "error": "Hallucinated records",
            "count_score": 0.0,
            "match_score": 0.0,
            "final_score": 0.0,
            "n_gold": 0,
            "n_pred": n_pred,
        }

    # COMPONENT 1: Count score (quadratic penalty)
    count_ratio = min(n_gold, n_pred) / max(n_gold, n_pred)
    count_score = count_ratio**2

    # COMPONENT 2: Record matching score
    matches = match_records_hungarian(gold_records, pred_records, field_matchers)

    if not matches:
        match_score = 0.0
        record_details = []
    else:
        # Evaluate matched pairs
        match_scores = []
        record_details = []

        for gold_idx, pred_idx, similarity in matches:
            gold_rec = gold_records[gold_idx]
            pred_rec = pred_records[pred_idx]

            rec_score, rec_details = evaluate_record_pair(gold_rec, pred_rec, field_matchers)
            match_scores.append(rec_score)

            record_details.append(
                {
                    "gold_idx": gold_idx,
                    "pred_idx": pred_idx,
                    "similarity": similarity,
                    "score": rec_score,
                    "details": rec_details,
                }
            )

        # Average match score across all gold records
        match_score = sum(match_scores) / len(match_scores)

    # Penalty for unmatched records
    n_matched = len(matches)
    n_unmatched_gold = n_gold - n_matched
    n_unmatched_pred = n_pred - n_matched

    # FINAL SCORE: Multiplicative combination
    # This ensures both count and quality matter
    final_score = count_score * match_score

    details = {
        "count_score": count_score,
        "match_score": match_score,
        "final_score": final_score,
        "n_gold": n_gold,
        "n_pred": n_pred,
        "n_matched": n_matched,
        "n_unmatched_gold": n_unmatched_gold,
        "n_unmatched_pred": n_unmatched_pred,
        "record_details": record_details,
    }

    return final_score, details


# ============================================================================
# LLM JUDGE FOR JSON
# ============================================================================


class JSONJudge(dspy.Signature):
    """Judge JSON extraction quality."""

    rubric = dspy.InputField(desc="Scoring rubric")
    gold_json = dspy.InputField(desc="Ground truth JSON")
    pred_json = dspy.InputField(desc="Predicted JSON")

    reasoning = dspy.OutputField(desc="Analysis of match quality")
    score = dspy.OutputField(desc="Score from 0.0 to 1.0")


def llm_judge_json(
    gold_data: Any, pred_data: Any, judge_lm: dspy.LM | None = None, rubric: str | None = None
) -> tuple[float, str]:
    """
    Use LLM to judge JSON extraction quality.

    Args:
        gold_data: Ground truth (JSON string or parsed)
        pred_data: Prediction (JSON string or parsed)
        judge_lm: DSPy LM to use for judging
        rubric: Custom scoring rubric

    Returns:
        score: float [0, 1]
        reasoning: str explanation
    """
    if rubric is None:
        rubric = """
Score JSON extraction quality using this rubric:

1. RECORD COUNT (40% weight):
   - Exact match: 1.0
   - Off by 1 record: 0.6
   - Off by 2 records: 0.3
   - Off by 3+ records: 0.1
   - Empty when should have data: 0.0

2. RECORD MATCHING (60% weight):
   For each gold record, find the best matching predicted record:
   - All fields correct: 1.0 per record
   - 1-2 fields incorrect: 0.7 per record
   - 3-4 fields incorrect: 0.4 per record
   - 5+ fields incorrect: 0.1 per record
   - No matching record found: 0.0 per record

   Average the scores across all gold records.

FINAL SCORE = count_score × match_score

Field comparison rules:
- None/null fields: Must match exactly (both None or both have values)
- String/number type differences: OK if values are semantically equivalent ("1" vs 1)
- Boolean vs integer: NOT OK - must match types (true ≠ 1, agent should fix type)
- Dates: Minor format differences OK if date is correct
- Minor typos/formatting: Penalize but don't fail completely

Calculate the final score and provide clear reasoning about what's wrong.
"""

    judge = dspy.ChainOfThought(JSONJudge)

    # Format JSON strings
    if not isinstance(gold_data, str):
        gold_str = json.dumps(gold_data, indent=2, ensure_ascii=False)
    else:
        gold_str = gold_data

    if not isinstance(pred_data, str):
        pred_str = json.dumps(pred_data, indent=2, ensure_ascii=False)
    else:
        pred_str = pred_data

    # Run judge with optional LM context
    if judge_lm is not None:
        with dspy.context(lm=judge_lm):
            result = judge(rubric=rubric, gold_json=gold_str, pred_json=pred_str)
    else:
        result = judge(rubric=rubric, gold_json=gold_str, pred_json=pred_str)

    # Parse score
    try:
        score = float(result.score)
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except (ValueError, AttributeError):
        score = 0.0

    reasoning = getattr(result, "reasoning", "No reasoning provided")

    return score, reasoning


# ============================================================================
# HYBRID: COMBINE PROGRAMMATIC + LLM
# ============================================================================


def hybrid_json_score(
    ground_truth: Any,
    predicted: Any,
    judge_lm: dspy.LM | None = None,
    programmatic_weight: float = 0.2,
    llm_weight: float = 0.8,
    field_matchers: dict[str, Any] | None = None,
) -> tuple[float, dict]:
    """
    Combine programmatic record-matching and LLM scoring.

    Args:
        ground_truth: JSON data
        predicted: JSON data
        judge_lm: LLM for judging
        programmatic_weight: Weight for programmatic score
        llm_weight: Weight for LLM score
        field_matchers: Optional dict of {field_name: matcher} for compositional matching

    Returns:
        score: Combined score [0, 1]
        details: Full breakdown
    """
    # Programmatic score
    prog_score, prog_details = json_match_score(ground_truth, predicted, field_matchers=field_matchers)

    # LLM score
    if judge_lm is not None:
        llm_score, llm_reasoning = llm_judge_json(ground_truth, predicted, judge_lm)
    else:
        llm_score = 0.0
        llm_reasoning = "No LLM judge provided"

    # Normalize weights
    total_weight = programmatic_weight + llm_weight
    norm_prog_w = programmatic_weight / total_weight
    norm_llm_w = llm_weight / total_weight

    # Combine
    final_score = norm_prog_w * prog_score + norm_llm_w * llm_score

    details = {
        "programmatic_score": prog_score,
        "llm_score": llm_score,
        "final_score": final_score,
        "programmatic_details": prog_details,
        "llm_reasoning": llm_reasoning,
    }

    return final_score, details
