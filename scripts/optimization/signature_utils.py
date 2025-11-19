"""Signature generation utilities for DSPy."""

from typing import Any, List, Optional

import dspy


def extract_value(obj: Any, field_name: str) -> Any:
    """Extract field value from object or dict.

    Args:
        obj: Object or dictionary
        field_name: Name of field to extract

    Returns:
        Field value or None
    """
    if isinstance(obj, dict):
        return obj.get(field_name)
    return getattr(obj, field_name, None)


def create_signature(
    field_name: str,
    field_type: str,
    examples: List[dspy.Example],
    valid_values: Optional[List[str]] = None,
) -> type:
    """Auto-generate DSPy signature from field info and examples.

    Args:
        field_name: Name of the field to extract
        field_type: Type of field (date, string, number, enum, json, etc.)
        examples: Example data to extract samples from
        valid_values: For enum types, list of valid values

    Returns:
        DSPy Signature class
    """
    # Extract sample values
    samples = []
    for ex in examples[:3]:
        val = extract_value(ex, field_name)
        if val is not None and val != "":
            samples.append(str(val))

    # Build description based on type with explicit DATA TYPE signature
    desc_parts = []

    # 1. Task/Field name (consistent across all types)
    field_label = field_name.replace('_', ' ')

    # 2. DATA TYPE signature (explicit type info before examples)
    type_signatures = {
        "date": (
            f"Extract {field_label} as date\n"
            f"DATA TYPE: date (ISO YYYY-MM-DD format)\n"
            f"Return: String like '2023-01-15' or null"
        ),
        "number": (
            f"Extract {field_label} as number\n"
            f"DATA TYPE: number (integer, supports currency like $1M, 1K)\n"
            f"Return: Numeric value or null"
        ),
        "float": (
            f"Extract {field_label} as float\n"
            f"DATA TYPE: float (decimal, supports percentages like 5%, $1,234.56, fractions)\n"
            f"Return: Decimal number like 3.5, 12.0, or null"
        ),
        "boolean": (
            f"Extract {field_label} as boolean\n"
            f"DATA TYPE: boolean (true/false)\n"
            f"Return: 'true' or 'false' (lowercase string), or null if not found"
        ),
        "enum": (
            f"Extract {field_label} as constrained value\n"
            f"DATA TYPE: enum\n"
            f"VALID VALUES: {valid_values if valid_values else ['Yes', 'No']}\n"
            f"Return: Exactly one of the valid values, or null if not found"
        ),
        "json": (
            f"Extract {field_label} as structured data\n"
            f"DATA TYPE: json (array of objects)\n"
            f"Return: JSON array like [{{'key': 'value'}}] or empty array []"
        ),
        "phone": (
            f"Extract {field_label} as phone number\n"
            f"DATA TYPE: phone (digits with optional formatting)\n"
            f"Return: Phone number like '(555) 123-4567' or null"
        ),
        "address": (
            f"Extract {field_label} as address\n"
            f"DATA TYPE: address (street address)\n"
            f"Return: Full or partial address string, or null"
        ),
        "string": (
            f"Extract {field_label} as text\n"
            f"DATA TYPE: string (text)\n"
            f"Return: Text value or null"
        ),
    }

    desc = type_signatures.get(field_type, f"Extract {field_label}")

    # 3. CRITICAL - NULL HANDLING section
    desc += "\n\nCRITICAL - NULL HANDLING:\nIf field is NOT found in document:"
    desc += "\n  → Return: null (NOT empty string, NOT 'unknown', NOT default)"
    desc += "\n  → Do NOT guess or invent values"
    desc += "\n  → Do NOT extract similar fields instead"

    # 4. Add examples section (well-separated with clear formatting)
    if samples:
        desc += "\n\n" + "─" * 70
        desc += "\nEXAMPLES:\n"

        # Add sample values as reference
        desc += f"Sample values from data: {'; '.join(samples[:])}"

    # Create signature class with enhanced description
    sig = type(
        f"Extract_{field_name}",
        (dspy.Signature,),
        {
            "document_text": dspy.InputField(desc="Lease document text"),
            field_name: dspy.OutputField(desc=desc),
        },
    )

    return sig