"""Enhanced feedback structure with rich error context for GEPA optimization.

This module provides structured feedback that helps the GEPA optimizer understand
not just that something failed, but WHY it failed and what type was expected.
"""

from dataclasses import dataclass, asdict
from typing import Any, Optional
import json
import ast


@dataclass
class EnhancedFeedback:
    """Rich feedback structure for optimization.

    Attributes:
        score: Numeric score [0, 1]
        feedback_text: Human-readable feedback string (for logging)
        expected_type: The type that was expected (e.g., "json", "string", "number")
        actual_type: The type that was actually returned (e.g., "string", "dict")
        parsing_error: Specific parsing error message (if applicable)
        recovery_suggestion: Actionable suggestion to fix the issue
        is_valid: Whether the output format is valid
    """
    score: float
    feedback_text: str
    expected_type: str
    actual_type: str
    parsing_error: Optional[str] = None
    recovery_suggestion: Optional[str] = None
    is_valid: bool = True

    def to_feedback_string(self) -> str:
        """Convert to DSPy-compatible feedback string."""
        return self.feedback_text

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return asdict(self)


def try_parse_json_with_feedback(
    value: Any, field_name: str
) -> tuple[Optional[list[dict]], EnhancedFeedback]:
    """
    Attempt to parse JSON value and provide detailed feedback on failure.

    Args:
        value: The value to parse (string, list, dict, etc.)
        field_name: Name of the field being parsed

    Returns:
        (parsed_value, feedback) where parsed_value is None on parse failure
    """
    # Already parsed
    if isinstance(value, list):
        return value, EnhancedFeedback(
            score=1.0,
            feedback_text=f"✓ {field_name}: Valid JSON",
            expected_type="json",
            actual_type="list",
            is_valid=True,
        )

    if isinstance(value, dict):
        return [value], EnhancedFeedback(
            score=1.0,
            feedback_text=f"✓ {field_name}: Valid JSON",
            expected_type="json",
            actual_type="dict",
            is_valid=True,
        )

    # Null case
    if value is None:
        return None, EnhancedFeedback(
            score=0.0,
            feedback_text=f"✗ {field_name}: JSON parse failed",
            expected_type="json",
            actual_type="None",
            parsing_error="Null value",
            recovery_suggestion='Ensure output is a valid JSON array like [{"field": "value"}]',
            is_valid=False,
        )

    # Try to parse string
    value_str = str(value).strip()

    if not value_str:
        return None, EnhancedFeedback(
            score=0.0,
            feedback_text=f"✗ {field_name}: JSON parse failed (empty string)",
            expected_type="json",
            actual_type="string",
            parsing_error="Empty string provided",
            recovery_suggestion='Return a valid JSON array like [{"field": "value"}] or null',
            is_valid=False,
        )

    # Attempt standard JSON parsing
    try:
        parsed = json.loads(value_str)
        if isinstance(parsed, dict):
            return [parsed], EnhancedFeedback(
                score=1.0,
                feedback_text=f"✓ {field_name}: Valid JSON",
                expected_type="json",
                actual_type="dict",
                is_valid=True,
            )
        elif isinstance(parsed, list):
            return parsed, EnhancedFeedback(
                score=1.0,
                feedback_text=f"✓ {field_name}: Valid JSON",
                expected_type="json",
                actual_type="list",
                is_valid=True,
            )
        else:
            return None, EnhancedFeedback(
                score=0.0,
                feedback_text=f"✗ {field_name}: JSON parse failed (wrong structure)",
                expected_type="json (array of objects)",
                actual_type=type(parsed).__name__,
                parsing_error=f"Expected array or object, got {type(parsed).__name__}",
                recovery_suggestion=f'Ensure output is a JSON array [...] or object {{...}}, not a {type(parsed).__name__}',
                is_valid=False,
            )
    except json.JSONDecodeError as e:
        # Try Python literal syntax
        try:
            parsed = ast.literal_eval(value_str)
            if isinstance(parsed, dict):
                return [parsed], EnhancedFeedback(
                    score=1.0,
                    feedback_text=f"✓ {field_name}: Valid JSON (Python syntax)",
                    expected_type="json",
                    actual_type="dict",
                    is_valid=True,
                )
            elif isinstance(parsed, list):
                return parsed, EnhancedFeedback(
                    score=1.0,
                    feedback_text=f"✓ {field_name}: Valid JSON (Python syntax)",
                    expected_type="json",
                    actual_type="list",
                    is_valid=True,
                )
        except (ValueError, SyntaxError):
            pass

        # Parse failed - provide detailed error info
        # Show a snippet of what was received
        snippet = value_str[:80] if len(value_str) <= 80 else value_str[:77] + "..."

        return None, EnhancedFeedback(
            score=0.0,
            feedback_text=f"✗ {field_name}: Invalid JSON - parse failed",
            expected_type='json array [{...}, {...}]',
            actual_type="string (invalid JSON)",
            parsing_error=f"JSON decode error at position {e.pos}: {e.msg}",
            recovery_suggestion=f'Got plain text instead of JSON. Ensure output is ONLY valid JSON like [{{"key": "value"}}] with no explanations or markdown. Received: {snippet}',
            is_valid=False,
        )


def try_parse_value_with_feedback(
    value: Any,
    field_name: str,
    expected_type: str,
    type_hints: Optional[dict] = None,
) -> EnhancedFeedback:
    """
    Validate a value matches expected type and provide detailed feedback.

    Args:
        value: The value to validate
        field_name: Name of the field
        expected_type: Expected type name (e.g., "string", "number", "date", "boolean")
        type_hints: Optional dict with type-specific validation hints

    Returns:
        EnhancedFeedback with validation result
    """
    from components.constants import NULL_VALUES

    # Check for null
    is_null = (
        value is None or
        (isinstance(value, str) and value.strip().lower() in NULL_VALUES) or
        (isinstance(value, (list, dict)) and len(value) == 0)
    )

    if is_null:
        return EnhancedFeedback(
            score=1.0,
            feedback_text=f"✓ {field_name}: Correctly null",
            expected_type=expected_type,
            actual_type="null",
            is_valid=True,
        )

    type_hints = type_hints or {}

    # Type-specific validation
    if expected_type == "string":
        if isinstance(value, str):
            return EnhancedFeedback(
                score=1.0,
                feedback_text=f"✓ {field_name}: Valid string",
                expected_type="string",
                actual_type="string",
                is_valid=True,
            )
        elif isinstance(value, (dict, list)):
            received = "JSON object" if isinstance(value, dict) else "JSON array"
            return EnhancedFeedback(
                score=0.0,
                feedback_text=f"✗ {field_name}: Type mismatch (got {received})",
                expected_type="raw string",
                actual_type=received,
                parsing_error=f"Received {received} instead of plain string",
                recovery_suggestion=f'Return ONLY the raw string value, not {received}. Examples: "John Doe" or null, never {{"value": "John Doe"}}',
                is_valid=False,
            )
        else:
            return EnhancedFeedback(
                score=0.0,
                feedback_text=f"✗ {field_name}: Type mismatch (got {type(value).__name__})",
                expected_type="string",
                actual_type=type(value).__name__,
                parsing_error=f"Expected string, got {type(value).__name__}",
                recovery_suggestion=f"Return a string value or null, not {type(value).__name__}",
                is_valid=False,
            )

    elif expected_type == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return EnhancedFeedback(
                score=1.0,
                feedback_text=f"✓ {field_name}: Valid number",
                expected_type="number",
                actual_type=type(value).__name__,
                is_valid=True,
            )
        elif isinstance(value, str):
            try:
                float(value)
                return EnhancedFeedback(
                    score=1.0,
                    feedback_text=f"✓ {field_name}: Valid number (string)",
                    expected_type="number",
                    actual_type="string",
                    is_valid=True,
                )
            except ValueError:
                return EnhancedFeedback(
                    score=0.0,
                    feedback_text=f"✗ {field_name}: Not a valid number",
                    expected_type="number",
                    actual_type="string (non-numeric)",
                    parsing_error=f"String '{value}' cannot be converted to number",
                    recovery_suggestion=f"Return ONLY a numeric value like 50000 or null, not '{value}'",
                    is_valid=False,
                )
        elif isinstance(value, (dict, list)):
            received = "JSON object" if isinstance(value, dict) else "JSON array"
            return EnhancedFeedback(
                score=0.0,
                feedback_text=f"✗ {field_name}: Type mismatch (got {received})",
                expected_type="number",
                actual_type=received,
                parsing_error=f"Received {received} instead of plain number",
                recovery_suggestion=f'Return ONLY the numeric value, not {received}. Example: 50000 or null, not {{"value": 50000}}',
                is_valid=False,
            )
        else:
            return EnhancedFeedback(
                score=0.0,
                feedback_text=f"✗ {field_name}: Type mismatch (got {type(value).__name__})",
                expected_type="number",
                actual_type=type(value).__name__,
                parsing_error=f"Expected number, got {type(value).__name__}",
                recovery_suggestion=f"Return a numeric value or null, not {type(value).__name__}",
                is_valid=False,
            )

    elif expected_type == "boolean":
        if isinstance(value, bool):
            return EnhancedFeedback(
                score=1.0,
                feedback_text=f"✓ {field_name}: Valid boolean",
                expected_type="boolean",
                actual_type="bool",
                is_valid=True,
            )
        elif isinstance(value, (dict, list)):
            received = "JSON object" if isinstance(value, dict) else "JSON array"
            return EnhancedFeedback(
                score=0.0,
                feedback_text=f"✗ {field_name}: Type mismatch (got {received})",
                expected_type="boolean (true/false)",
                actual_type=received,
                parsing_error=f"Received {received} instead of plain boolean",
                recovery_suggestion=f'Return ONLY true, false, or null. Do not wrap in {received}. Examples: true | false | null',
                is_valid=False,
            )
        else:
            return EnhancedFeedback(
                score=0.0,
                feedback_text=f"✗ {field_name}: Type mismatch (got {type(value).__name__})",
                expected_type="boolean",
                actual_type=type(value).__name__,
                parsing_error=f"Expected boolean, got {type(value).__name__}",
                recovery_suggestion="Return true, false, or null",
                is_valid=False,
            )

    elif expected_type == "date":
        if isinstance(value, str):
            import re
            # Simple ISO date validation
            if re.match(r"^\d{4}-\d{2}-\d{2}$", value.strip()):
                return EnhancedFeedback(
                    score=1.0,
                    feedback_text=f"✓ {field_name}: Valid date",
                    expected_type="ISO date (YYYY-MM-DD)",
                    actual_type="string",
                    is_valid=True,
                )
            else:
                return EnhancedFeedback(
                    score=0.0,
                    feedback_text=f"✗ {field_name}: Invalid date format",
                    expected_type="ISO date (YYYY-MM-DD)",
                    actual_type="string (invalid format)",
                    parsing_error=f"Date '{value}' doesn't match format YYYY-MM-DD",
                    recovery_suggestion=f"Return date in ISO format YYYY-MM-DD (e.g., 2025-01-15), not '{value}'",
                    is_valid=False,
                )
        elif isinstance(value, (dict, list)):
            received = "JSON object" if isinstance(value, dict) else "JSON array"
            return EnhancedFeedback(
                score=0.0,
                feedback_text=f"✗ {field_name}: Type mismatch (got {received})",
                expected_type="ISO date string (YYYY-MM-DD)",
                actual_type=received,
                parsing_error=f"Received {received} instead of plain date string",
                recovery_suggestion=f'Return ONLY the date string in ISO format like "2025-01-15", not {received}',
                is_valid=False,
            )
        else:
            return EnhancedFeedback(
                score=0.0,
                feedback_text=f"✗ {field_name}: Type mismatch (got {type(value).__name__})",
                expected_type="ISO date (YYYY-MM-DD)",
                actual_type=type(value).__name__,
                parsing_error=f"Expected date string, got {type(value).__name__}",
                recovery_suggestion="Return a date string in format YYYY-MM-DD or null",
                is_valid=False,
            )

    elif expected_type == "json":
        # JSON arrays/objects
        return EnhancedFeedback(
            score=1.0,
            feedback_text=f"✓ {field_name}: Valid JSON",
            expected_type="json",
            actual_type=type(value).__name__,
            is_valid=True,
        )

    # Unknown type - be permissive
    return EnhancedFeedback(
        score=1.0,
        feedback_text=f"✓ {field_name}: Valid value",
        expected_type=expected_type,
        actual_type=type(value).__name__,
        is_valid=True,
    )


def format_feedback_with_context(
    feedback: EnhancedFeedback,
    field_name: str,
    field_type: str,
) -> str:
    """
    Format enhanced feedback with full context for the optimizer.

    Args:
        feedback: EnhancedFeedback object
        field_name: Name of field
        field_type: Type of field

    Returns:
        Formatted feedback string
    """
    lines = [feedback.feedback_text]

    if not feedback.is_valid:
        lines.append(f"  Expected type: {feedback.expected_type}")
        lines.append(f"  Received type: {feedback.actual_type}")

        if feedback.parsing_error:
            lines.append(f"  Error: {feedback.parsing_error}")

        if feedback.recovery_suggestion:
            lines.append(f"  Fix: {feedback.recovery_suggestion}")

    return "\n".join(lines)
