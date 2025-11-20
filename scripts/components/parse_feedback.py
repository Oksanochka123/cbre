"""Simple feedback for when parsing fails.

After attempting to parse/validate with the elaborate logic,
if it fails, we return feedback explaining the type mismatch.
"""

from typing import Any
import json
from json_metrics import parse_json_safe


def is_json_string(value: Any) -> bool:
    """Check if a value is a string that looks like JSON.

    Returns True if it's a string that starts with { or [.
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    return (s.startswith('{') and s.endswith('}')) or \
           (s.startswith('[') and s.endswith(']'))


def get_json_structure_type(value: str) -> str:
    """Get the structure type of a JSON string."""
    if not isinstance(value, str):
        return "unknown"
    s = value.strip()
    if s.startswith('{'):
        return "JSON object"
    elif s.startswith('['):
        return "JSON array"
    return "unknown"


def get_type_name(value: Any) -> str:
    """Get human-readable type name."""
    if isinstance(value, bool):
        return "boolean"
    elif isinstance(value, dict):
        return "JSON object"
    elif isinstance(value, list):
        return "JSON array"
    elif isinstance(value, str):
        # Check if it's a JSON string
        if is_json_string(value):
            return get_json_structure_type(value)
        return "string"
    elif isinstance(value, (int, float)):
        return "number"
    elif value is None:
        return "null"
    else:
        return type(value).__name__


def _extract_json_wrapped_value(value: str) -> str | None:
    """Try to extract the wrapped value from a JSON object.

    If a non-JSON field type receives {"field": "value"}, suggest the unwrapped value.
    Uses parse_json_safe from json_metrics for consistent parsing.
    """
    if not isinstance(value, str):
        return None

    parsed = parse_json_safe(value)
    if parsed is None:
        return None

    # Extract first value from parsed list/dict
    if isinstance(parsed, list) and parsed:
        first_item = parsed[0]
        if isinstance(first_item, dict) and first_item:
            # {"field": "value"} -> extract the value
            return str(list(first_item.values())[0])
        return str(first_item)

    return None


def format_parse_error_feedback(
    field_name: str,
    expected_type: str,
    actual_value: Any,
    parse_error: str | None = None,
) -> str:
    """Format feedback when parsing/validation fails.

    Args:
        field_name: Name of the field
        expected_type: What type was expected (e.g., "string", "number", "json")
        actual_value: What was actually received
        parse_error: Optional specific error message

    Returns:
        Formatted feedback string
    """
    actual_type = get_type_name(actual_value)

    lines = [f"✗ {field_name}: Type mismatch"]
    lines.append(f"  Expected: {expected_type}")
    lines.append(f"  Got: {actual_type}")

    if parse_error:
        lines.append(f"  Error: {parse_error}")

    # Add recovery suggestions based on type mismatch
    if expected_type == "string" and actual_type in ("JSON object", "JSON array"):
        # Try to extract the wrapped value and suggest it
        extracted = _extract_json_wrapped_value(actual_value)
        lines.append(f"  Fix: Return ONLY the raw string value, not {actual_type}")
        if extracted:
            lines.append(f"  Example: Instead of {actual_value}, return: {extracted}")
    elif expected_type == "number" and actual_type in ("JSON object", "JSON array"):
        extracted = _extract_json_wrapped_value(actual_value)
        lines.append(f"  Fix: Return ONLY the numeric value, not {actual_type}")
        if extracted:
            lines.append(f"  Example: Instead of {actual_value}, return: {extracted}")
    elif expected_type == "boolean" and actual_type in ("JSON object", "JSON array"):
        extracted = _extract_json_wrapped_value(actual_value)
        lines.append(f"  Fix: Return ONLY true, false, or null. Do not wrap in {actual_type}")
        if extracted:
            lines.append(f"  Example: Instead of {actual_value}, return: {extracted}")
    elif expected_type == "date" and actual_type in ("JSON object", "JSON array"):
        extracted = _extract_json_wrapped_value(actual_value)
        lines.append(f"  Fix: Return ONLY the date string in ISO format (YYYY-MM-DD), not {actual_type}")
        if extracted:
            lines.append(f"  Example: Instead of {actual_value}, return: {extracted}")
    elif expected_type == "address" and actual_type in ("JSON object", "JSON array"):
        extracted = _extract_json_wrapped_value(actual_value)
        lines.append(f"  Fix: Return ONLY the raw address string, not {actual_type}")
        if extracted:
            lines.append(f"  Example: Instead of {actual_value}, return: {extracted}")
    elif expected_type == "json" and actual_type == "string":
        lines.append(f"  Fix: Return ONLY valid JSON like [{{...}}], not plain text")

    return "\n".join(lines)
