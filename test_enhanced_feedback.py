#!/usr/bin/env python3
"""Quick test of enhanced feedback system."""

import sys
sys.path.insert(0, '/mloscratch/homes/protsenk/prompt_optimization/field_extraction/scripts')

from components.feedback import (
    try_parse_json_with_feedback,
    try_parse_value_with_feedback,
    format_feedback_with_context,
)

def test_json_feedback():
    """Test JSON parsing feedback."""
    print("=" * 80)
    print("JSON FEEDBACK TESTS")
    print("=" * 80)

    test_cases = [
        ('percentage_rent_table', '[{"key": "value"}]', "Valid JSON array"),
        ('percentage_rent_table', 'some plain text describing data', "Plain text instead of JSON"),
        ('percentage_rent_table', '{"incomplete":', "Malformed JSON"),
        ('percentage_rent_table', '', "Empty string"),
        ('percentage_rent_table', None, "Null value"),
    ]

    for field_name, value, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input: {repr(value)[:60]}")
        parsed, feedback = try_parse_json_with_feedback(value, field_name)
        formatted = format_feedback_with_context(feedback, field_name, "json")
        print(f"  Score: {feedback.score}")
        print(f"  Valid: {feedback.is_valid}")
        print(f"  Feedback:\n{formatted}")

def test_string_feedback():
    """Test string type validation feedback."""
    print("\n" + "=" * 80)
    print("STRING TYPE FEEDBACK TESTS")
    print("=" * 80)

    test_cases = [
        ('guarantor_name', 'John Doe', "Valid string"),
        ('guarantor_name', '{"name": "John Doe"}', "JSON object instead of string"),
        ('guarantor_name', '[{"name": "John"}]', "JSON array instead of string"),
        ('guarantor_name', 123, "Number instead of string"),
    ]

    for field_name, value, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input: {repr(value)}")
        feedback = try_parse_value_with_feedback(value, field_name, "string")
        formatted = format_feedback_with_context(feedback, field_name, "string")
        print(f"  Score: {feedback.score}")
        print(f"  Valid: {feedback.is_valid}")
        print(f"  Feedback:\n{formatted}")

def test_number_feedback():
    """Test number type validation feedback."""
    print("\n" + "=" * 80)
    print("NUMBER TYPE FEEDBACK TESTS")
    print("=" * 80)

    test_cases = [
        ('building_square_footage', '256495', "Valid number string"),
        ('building_square_footage', 256495, "Valid number"),
        ('building_square_footage', '{"value": 256495}', "JSON object instead of number"),
        ('building_square_footage', 'not a number', "Non-numeric string"),
    ]

    for field_name, value, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input: {repr(value)}")
        feedback = try_parse_value_with_feedback(value, field_name, "number")
        formatted = format_feedback_with_context(feedback, field_name, "number")
        print(f"  Score: {feedback.score}")
        print(f"  Valid: {feedback.is_valid}")
        print(f"  Feedback:\n{formatted}")

def test_boolean_feedback():
    """Test boolean type validation feedback."""
    print("\n" + "=" * 80)
    print("BOOLEAN TYPE FEEDBACK TESTS")
    print("=" * 80)

    test_cases = [
        ('tenant_insurance_required', True, "Valid boolean"),
        ('tenant_insurance_required', 'true', "Valid boolean string"),
        ('tenant_insurance_required', '{"value": true}', "JSON object instead of boolean"),
        ('tenant_insurance_required', 1, "Integer instead of boolean"),
    ]

    for field_name, value, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input: {repr(value)}")
        feedback = try_parse_value_with_feedback(value, field_name, "boolean")
        formatted = format_feedback_with_context(feedback, field_name, "boolean")
        print(f"  Score: {feedback.score}")
        print(f"  Valid: {feedback.is_valid}")
        print(f"  Feedback:\n{formatted}")

def test_date_feedback():
    """Test date type validation feedback."""
    print("\n" + "=" * 80)
    print("DATE TYPE FEEDBACK TESTS")
    print("=" * 80)

    test_cases = [
        ('execution_date', '2025-01-15', "Valid ISO date"),
        ('execution_date', '{"date": "2025-01-15"}', "JSON object instead of date string"),
        ('execution_date', '2025/01/15', "Wrong date format"),
    ]

    for field_name, value, description in test_cases:
        print(f"\n{description}:")
        print(f"  Input: {repr(value)}")
        feedback = try_parse_value_with_feedback(value, field_name, "date")
        formatted = format_feedback_with_context(feedback, field_name, "date")
        print(f"  Score: {feedback.score}")
        print(f"  Valid: {feedback.is_valid}")
        print(f"  Feedback:\n{formatted}")

if __name__ == "__main__":
    test_json_feedback()
    test_string_feedback()
    test_number_feedback()
    test_boolean_feedback()
    test_date_feedback()
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)
