"""Signature generation utilities for DSPy."""

from typing import Any, List

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


def create_signature(field_name: str, field_type: str, examples: List[dspy.Example]) -> type:
    """Auto-generate DSPy signature from field info and examples.
    
    Args:
        field_name: Name of the field to extract
        field_type: Type of field (date, string, number, etc.)
        examples: Example data to extract samples from
        
    Returns:
        DSPy Signature class
    """
    # Extract sample values
    samples = []
    for ex in examples[:2]:
        val = extract_value(ex, field_name)
        if val:
            samples.append(str(val))

    # Build description based on type
    desc_map = {
        "date": f"Extract {field_name.replace('_', ' ')} in YYYY-MM-DD format",
        "number": f"Extract numeric {field_name.replace('_', ' ')}",
        "float": f"Extract {field_name.replace('_', ' ')} as a float number",
        "enum": f"Extract {field_name.replace('_', ' ')}",
        "json": f"Extract {field_name.replace('_', ' ')} as a JSON list of dictionaries",
        "phone": f"Extract {field_name.replace('_', ' ')} as phone number",
        "address": f"Extract {field_name.replace('_', ' ')} as complete address",
        "string": f"Extract {field_name.replace('_', ' ')}",
    }
    desc = desc_map.get(field_type, f"Extract {field_name}")

    # Add examples to description
    if samples and field_type not in ["json", "phone", "address"]:
        desc += f". Examples: {', '.join(samples[:2])}"
    elif samples and field_type == "json":
        desc += ". Return valid JSON format."
    elif samples and field_type in ["phone", "address"]:
        desc += f". Example: {samples[0]}"

    # Create signature class
    sig = type(
        f"Extract_{field_name}",
        (dspy.Signature,),
        {
            "document_text": dspy.InputField(desc="Lease document text"),
            field_name: dspy.OutputField(desc=desc),
        },
    )

    return sig