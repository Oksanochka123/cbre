"""Unit tests for JSONMatcher."""

import json

import pytest

from scripts.matchers import JSONMatcher


class TestJSONMatcher:
    """Test JSON list matching."""

    def test_empty_lists(self):
        m = JSONMatcher("items")
        score, _ = m.match("[]", "[]")
        assert score == 1.0

    def test_single_record_match(self):
        m = JSONMatcher("items")
        gold = '[{"name": "A", "value": 1}]'
        pred = '[{"name": "A", "value": 1}]'
        score, _ = m.match(gold, pred)
        assert score >= 0.95

    def test_record_count_mismatch(self):
        m = JSONMatcher("items")
        gold = '[{"name": "A"}, {"name": "B"}]'
        pred = '[{"name": "A"}]'
        score, _ = m.match(gold, pred)
        assert 0.0 < score < 0.7

    def test_field_mismatch(self):
        m = JSONMatcher("items")
        gold = '[{"name": "A", "value": 1}]'
        pred = '[{"name": "A", "value": 2}]'
        score, _ = m.match(gold, pred)
        assert 0.4 < score < 0.9


class TestJSONMatcherWithSchema:
    """Test JSON matcher with schema-based field matchers."""

    def test_schema_with_string_matcher(self):
        schema = {"name": {"type": "string"}, "description": {"type": "string"}}
        m = JSONMatcher("items", field_schema=schema)

        gold = [{"name": "Alice", "description": "Engineer"}]
        pred = [{"name": "alice", "description": "engineer"}]
        score, _ = m.match(json.dumps(gold), json.dumps(pred))
        assert score == 1.0  # Case-insensitive match

    def test_schema_with_number_matcher(self):
        schema = {"unit_number": {"type": "string"}, "rent": {"type": "number"}}
        m = JSONMatcher("leases", field_schema=schema)

        gold = [{"unit_number": "101", "rent": 1000}]
        pred = [{"unit_number": "101", "rent": 1001}]  # Within 0.5% margin
        score, _ = m.match(json.dumps(gold), json.dumps(pred))
        assert score == 1.0

    def test_schema_with_date_matcher(self):
        schema = {"unit": {"type": "string"}, "start_date": {"type": "date"}}
        m = JSONMatcher("leases", field_schema=schema)

        gold = [{"unit": "101", "start_date": "2024-01-15"}]
        pred = [{"unit": "101", "start_date": "01/15/2024"}]  # Different format
        score, _ = m.match(json.dumps(gold), json.dumps(pred))
        assert score == 1.0

    def test_schema_with_enum_matcher(self):
        schema = {
            "unit": {"type": "string"},
            "frequency": {"type": "enum", "valid_values": ["Annual", "Monthly", "Quarterly"]},
        }
        m = JSONMatcher("billings", field_schema=schema)

        gold = [{"unit": "101", "frequency": "Monthly"}]
        pred = [{"unit": "101", "frequency": "monthly"}]  # Case variation
        score, _ = m.match(json.dumps(gold), json.dumps(pred))
        assert score == 1.0

    def test_schema_with_float_matcher(self):
        schema = {"unit": {"type": "string"}, "rate": {"type": "float", "tolerance": 0.01}}
        m = JSONMatcher("rates", field_schema=schema)

        gold = [{"unit": "101", "rate": 3.14159}]
        pred = [{"unit": "101", "rate": 3.14260}]  # Within tolerance
        score, _ = m.match(json.dumps(gold), json.dumps(pred))
        assert score == 1.0

    def test_schema_missing_field_raises_error(self):
        # Schema only defines 'name', but data has 'value' too
        schema = {"name": {"type": "string"}}
        m = JSONMatcher("items", field_schema=schema)

        gold = [{"name": "A", "value": 1}]
        pred = [{"name": "A", "value": 1}]

        with pytest.raises(ValueError, match="Field 'value' is missing from field_matchers"):
            m.match(json.dumps(gold), json.dumps(pred))

    def test_schema_complex_record(self):
        schema = {
            "unit_number": {"type": "string"},
            "tenant_name": {"type": "string"},
            "rent_amount": {"type": "number"},
            "start_date": {"type": "date"},
            "lease_type": {"type": "enum", "valid_values": ["Gross", "Net", "Modified Gross"]},
        }
        m = JSONMatcher("leases", field_schema=schema)

        gold = [
            {
                "unit_number": "101",
                "tenant_name": "John Doe",
                "rent_amount": 2500,
                "start_date": "2024-01-01",
                "lease_type": "Gross",
            }
        ]
        pred = [
            {
                "unit_number": "101",
                "tenant_name": "john doe",  # Case difference
                "rent_amount": 2501,  # Within margin
                "start_date": "01/01/2024",  # Format difference
                "lease_type": "gross",  # Case difference
            }
        ]
        score, _ = m.match(json.dumps(gold), json.dumps(pred))
        assert score == 1.0
