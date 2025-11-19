"""Type enforcement templates for atomic field extraction.

This module provides type-specific output format enforcement instructions
that can be prepended or appended to prompts to ensure structured output
from LLMs.
"""

# Type enforcement templates for critical output format instructions
TYPE_ENFORCERS = {
    "string": """
================================================================================
CRITICAL OUTPUT FORMAT - THIS OVERRIDES ALL PREVIOUS INSTRUCTIONS
================================================================================
IGNORE any previous instructions about JSON format, field names, or structured output.

REQUIRED OUTPUT FORMAT:
• Return ONLY the raw string value
• NO JSON objects, NO field names as keys, NO braces {}, NO formatting like "field_name: value"

CORRECT examples: "John Doe" | "5412 W. Plano Parkway" | null
WRONG examples: {"guarantor_name": "John Doe"} | guarantor_name: "John Doe" | {"value": "John Doe", "reasoning": "..."}

If you cannot find the value, return: null
REMINDER: Output ONLY the raw string value or null. Nothing else.
================================================================================
""",
    "number": """
================================================================================
CRITICAL OUTPUT FORMAT - THIS OVERRIDES ALL PREVIOUS INSTRUCTIONS
================================================================================
IGNORE any previous instructions about JSON format, field names, or structured output.

REQUIRED OUTPUT FORMAT:
• Return ONLY the numeric value (integer)
• NO JSON objects, NO field names as keys, NO braces {}, NO formatting like "field_name: value"

CORRECT examples: 50000 | 256495 | null
WRONG examples: {"ti_total_cost": 50000} | ti_total_cost: 50000 | building_square_footage: 256495 | {"value": 50000, "reasoning": "..."}

If you cannot find the value, return: null
REMINDER: Output ONLY the numeric value or null. Nothing else.
================================================================================
""",
    "float": """
================================================================================
CRITICAL OUTPUT FORMAT - THIS OVERRIDES ALL PREVIOUS INSTRUCTIONS
================================================================================
IGNORE any previous instructions about JSON format, field names, or structured output.

REQUIRED OUTPUT FORMAT:
• Return ONLY the numeric value (float/decimal)
• NO JSON objects, NO field names as keys, NO braces {}, NO formatting like "field_name: value"

CORRECT examples: 0.85 | 0.2544 | 1250.50 | null
WRONG examples: {"project_prorata_pct": 0.85} | project_prorata_pct: 0.85 | {"value": 0.2544} | 0.2544 (as decimal)

If you cannot find the value, return: null
REMINDER: Output ONLY the numeric value or null. Nothing else.
================================================================================
""",
    "boolean": """
================================================================================
CRITICAL OUTPUT FORMAT - THIS OVERRIDES ALL PREVIOUS INSTRUCTIONS
================================================================================
IGNORE any previous instructions about JSON format, field names, explanations, or structured output.

REQUIRED OUTPUT FORMAT:
• Return ONLY: true, false, or null
• NO JSON objects, NO field names, NO explanations or reasoning, NO braces {}

CORRECT examples: true | false | null
WRONG examples: {"tenant_property_insurance_required": true} | tenant_property_insurance_required: true | {"value": true, "explanation": "..."}

If you cannot determine the value, return: null
REMINDER: Output ONLY true, false, or null. Nothing else.
================================================================================
""",
    "date": """
================================================================================
CRITICAL OUTPUT FORMAT - THIS OVERRIDES ALL PREVIOUS INSTRUCTIONS
================================================================================
IGNORE any previous instructions about JSON format, field names, or structured output.

REQUIRED OUTPUT FORMAT:
• Return ONLY the date string in ISO format (YYYY-MM-DD)
• NO JSON objects, NO field names as keys, NO braces {}, NO formatting like "field_name: value"

CORRECT examples: "2025-01-15" | "2026-04-30" | null
WRONG examples: {"insurance_term_start": "2025-01-15"} | insurance_term_start: "2025-01-15" | contraction_expiration_date: "2026-04-30"

If you cannot find the date, return: null
REMINDER: Output ONLY the ISO date string or null. Nothing else.
================================================================================
""",
    "enum": """
================================================================================
CRITICAL OUTPUT FORMAT - THIS OVERRIDES ALL PREVIOUS INSTRUCTIONS
================================================================================
IGNORE any previous instructions about JSON format, field names, or structured output.

REQUIRED OUTPUT FORMAT:
• Return ONLY the enum value string
• NO JSON objects, NO field names as keys, NO braces {}, NO formatting like "field_name: value"

CORRECT examples: "PSF" | "Total" | "Monthly" | null
WRONG examples: {"ti_calc_method": "PSF"} | ti_calc_method: "PSF" | {"value": "Total"}

If the enum value is not found, return: null
REMINDER: Output ONLY the enum value string or null. Nothing else.
================================================================================
""",
    "address": """
================================================================================
CRITICAL OUTPUT FORMAT - THIS OVERRIDES ALL PREVIOUS INSTRUCTIONS
================================================================================
IGNORE any previous instructions about JSON format, field names, or structured output.

REQUIRED OUTPUT FORMAT:
• Return ONLY the raw address string
• NO JSON objects, NO field names as keys, NO braces {}, NO formatting like "field_name: value"

CORRECT examples: "123 Main St, Suite 100, New York, NY 10001" | "5412 W. Plano Parkway" | null
WRONG examples: {"legal_notice_address": "123 Main St"} | legal_notice_address: "123 Main St" | {"value": "123 Main St"}

If you cannot find the address, return: null
REMINDER: Output ONLY the raw address string or null. Nothing else.
================================================================================
""",
    "phone": """
================================================================================
CRITICAL OUTPUT FORMAT - THIS OVERRIDES ALL PREVIOUS INSTRUCTIONS
================================================================================
IGNORE any previous instructions about JSON format, field names, or structured output.

REQUIRED OUTPUT FORMAT:
• Return ONLY the phone number string
• NO JSON objects, NO field names as keys, NO braces {}, NO formatting like "field_name: value"

CORRECT examples: "(555) 123-4567" | "555-123-4567" | "5551234567" | null
WRONG examples: {"tenant_phone": "(555) 123-4567"} | tenant_phone: "(555) 123-4567" | {"value": "(555) 123-4567"}

If you cannot find the phone number, return: null
REMINDER: Output ONLY the phone number string or null. Nothing else.
================================================================================
""",
    "json": """
================================================================================
CRITICAL OUTPUT FORMAT - THIS OVERRIDES ALL PREVIOUS INSTRUCTIONS
================================================================================
IGNORE any previous instructions about field names, plain text, or non-JSON formats.

REQUIRED OUTPUT FORMAT:
• Return ONLY valid JSON (array or object)
• NO markdown code blocks, NO explanations, NO plain text
• The JSON must be parseable and complete

CORRECT examples: [{"key": "value"}] | {"nested": {"data": true}} | null
WRONG examples: ```json\n[...]\n``` | "Here's the JSON: {...}" | {"incomplete":

If you cannot extract the data, return: null
REMINDER: Output ONLY valid JSON or null. Nothing else.
================================================================================
""",
}


def get_enforcer(field_type: str) -> str | None:
    """Get the type enforcement template for a field type.

    Args:
        field_type: The field data type (e.g., 'string', 'number', 'date')

    Returns:
        The enforcement template string, or None if type is not found
    """
    return TYPE_ENFORCERS.get(field_type)


def has_enforcer(field_type: str) -> bool:
    """Check if an enforcer exists for a field type.

    Args:
        field_type: The field data type

    Returns:
        True if an enforcer exists for this type
    """
    return field_type in TYPE_ENFORCERS


def apply_enforcer(
    instructions: str,
    field_type: str,
    position: str = "both"
) -> str:
    """Apply type enforcement to instructions.

    Args:
        instructions: The original instructions string
        field_type: The field data type
        position: Where to add enforcement - "prepend", "append", or "both"

    Returns:
        Instructions with enforcement applied, or original if no enforcer found

    Raises:
        ValueError: If position is invalid
    """
    enforcer = get_enforcer(field_type)
    if not enforcer:
        return instructions

    if position == "prepend":
        return enforcer + "\n\n" + instructions
    elif position == "append":
        return instructions + "\n" + enforcer
    elif position == "both":
        return enforcer + "\n\n" + instructions + "\n" + enforcer
    else:
        raise ValueError(f"Invalid position: {position}. Must be 'prepend', 'append', or 'both'")


def is_already_enforced(instructions: str) -> bool:
    """Check if instructions already contain type enforcement.

    Args:
        instructions: The instructions string to check

    Returns:
        True if enforcement markers are already present
    """
    return "CRITICAL OUTPUT FORMAT" in instructions


def list_supported_types() -> list[str]:
    """Get list of all supported field types.

    Returns:
        List of supported field types sorted alphabetically
    """
    return sorted(TYPE_ENFORCERS.keys())
