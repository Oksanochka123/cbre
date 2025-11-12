"""Unit tests for field matchers."""

import pytest

from scripts.matchers import (
    AddressMatcher,
    BooleanMatcher,
    DateMatcher,
    EnumMatcher,
    FloatMatcher,
    JSONMatcher,
    MatcherRegistry,
    NumberMatcher,
    PhoneMatcher,
    StringMatcher,
)


class TestPhoneMatcher:
    """Test phone number matching."""

    def test_exact_match(self):
        m = PhoneMatcher("phone")
        score, _ = m.match("555-123-4567", "(555) 123-4567")
        assert score == 1.0

    def test_partial_match(self):
        m = PhoneMatcher("phone")
        # Country code difference now gets full match (improved logic)
        score, _ = m.match("1-555-123-4567", "555-123-4567")
        assert score == 1.0

    def test_invalid_placeholder(self):
        m = PhoneMatcher("phone")
        assert m._normalize("unknown") is None
        assert m._normalize("000-000-0000") is None
        assert m._normalize("888-888-8888") is None

    def test_format_variations(self):
        m = PhoneMatcher("phone")
        score, _ = m.match("+1 (555) 123-4567", "555.123.4567")
        assert score == 1.0

    def test_international_format(self):
        m = PhoneMatcher("phone")
        score, _ = m.match("+1-555-123-4567", "555-123-4567")
        assert score == 1.0

    def test_different_phones(self):
        m = PhoneMatcher("phone")
        score, _ = m.match("555-123-4567", "555-987-6543")
        assert score == 0.0

    def test_empty_phone(self):
        m = PhoneMatcher("phone")
        assert m._normalize("") is None
        assert m._normalize("   ") is None

    def test_short_phone(self):
        m = PhoneMatcher("phone")
        # Too short
        assert m._normalize("123") is None

        # Valid 7-digit
        assert m._normalize("123-4567") is not None


class TestDateMatcher:
    """Test date matching."""

    def test_exact_match(self):
        m = DateMatcher("date")
        score, _ = m.match("2024-01-15", "01/15/2024")
        assert score == 1.0

    def test_one_day_tolerance(self):
        m = DateMatcher("date")
        score, _ = m.match("2024-01-15", "2024-01-16")
        assert score == 0.9

    def test_parse_failure(self):
        m = DateMatcher("date")
        score, _ = m.match("invalid", "2024-01-15")
        assert score == 0.0

    def test_format_variations(self):
        m = DateMatcher("date")
        score, _ = m.match("January 15, 2024", "01/15/2024")
        assert score == 1.0

    def test_european_vs_us_format(self):
        m = DateMatcher("date")
        # Same date, different formats
        score, _ = m.match("2024-03-15", "03/15/2024")  # ISO vs US
        assert score == 1.0

    def test_month_name_formats(self):
        m = DateMatcher("date")
        score, _ = m.match("Jan 15, 2024", "January 15, 2024")
        assert score == 1.0

    def test_different_dates(self):
        m = DateMatcher("date")
        score, _ = m.match("2024-01-15", "2024-02-15")
        assert score == 0.0

    def test_year_difference(self):
        m = DateMatcher("date")
        score, _ = m.match("2024-01-15", "2023-01-15")
        assert score == 0.0


class TestNumberMatcher:
    """Test numeric matching."""

    def test_exact_match(self):
        m = NumberMatcher("num")
        score, _ = m.match("1000", "1000")
        assert score == 1.0

    def test_comma_handling(self):
        m = NumberMatcher("num")
        score, _ = m.match("1,000", "1000")
        assert score == 1.0

        score, _ = m.match("1,000,000", "1000000")
        assert score == 1.0

    def test_decimal_handling(self):
        m = NumberMatcher("num")
        score, _ = m.match("1000.0", "1000")
        assert score == 1.0

        score, _ = m.match("1000.00", "1000")
        assert score == 1.0

    def test_wrong_number(self):
        m = NumberMatcher("num")
        # Within 0.5% margin (1/1000 = 0.1%)
        score, _ = m.match("1000", "1001")
        assert score == 1.0

        score, _ = m.match("1000", "999")
        assert score == 1.0  # Within 0.5% margin

        score, _ = m.match("5000", "5050")
        assert score == 0.0  # 1% error, outside margin

    def test_large_numbers(self):
        m = NumberMatcher("num")
        score, _ = m.match("1000000", "1,000,000")
        assert score == 1.0

        score, _ = m.match("1000000", "1000001")
        assert score == 1.0  # Within 0.5% margin

    def test_negative_numbers(self):
        m = NumberMatcher("num")
        score, _ = m.match("-1000", "-1000")
        assert score == 1.0

        score, _ = m.match("-1000", "1000")
        assert score == 0.0

    def test_zero(self):
        m = NumberMatcher("num")
        score, _ = m.match("0", "0")
        assert score == 1.0

        score, _ = m.match("0", "1")
        assert score == 0.0

    def test_parse_error(self):
        m = NumberMatcher("num")
        score, _ = m.match("not_a_number", "1000")
        assert score == 0.0

        score, _ = m.match("1000", "abc")
        assert score == 0.0


class TestFloatMatcher:
    """Test float matching with tolerance."""

    def test_within_tolerance(self):
        m = FloatMatcher("val", tolerance=0.001)
        score, _ = m.match("3.14159", "3.14160")
        assert score == 1.0

    def test_outside_tolerance(self):
        m = FloatMatcher("val", tolerance=0.001)
        score, _ = m.match("3.14159", "3.15")
        assert score == 0.0  # Binary: outside tolerance = 0

    def test_exact_match(self):
        m = FloatMatcher("val")
        score, _ = m.match("1.5", "1.5")
        assert score == 1.0

    def test_default_tolerance(self):
        m = FloatMatcher("val")  # default 5e-4
        score, _ = m.match("1.0", "1.0004")
        assert score == 1.0

        score, _ = m.match("1.0", "1.001")
        assert score == 0.0

    def test_scientific_notation(self):
        m = FloatMatcher("val", tolerance=1e-6)
        score, _ = m.match("1e-5", "0.00001")
        assert score == 1.0

    def test_very_small_numbers(self):
        m = FloatMatcher("val", tolerance=1e-10)
        score, _ = m.match("0.0000000001", "1e-10")
        assert score == 1.0


class TestStringMatcher:
    """Test string matching."""

    def test_exact_match(self):
        m = StringMatcher("name")
        score, _ = m.match("John Doe", "john doe")
        assert score == 1.0

    def test_high_similarity(self):
        m = StringMatcher("name")
        score, _ = m.match("John Doe", "John Do")
        assert 0.15 < score < 0.25  # sim_weight * similarity

    def test_low_similarity(self):
        m = StringMatcher("name")
        score, _ = m.match("John Doe", "Jane Smith")
        assert 0.0 <= score < 0.1


class TestBooleanMatcher:
    """Test boolean matching."""

    def test_true_values(self):
        m = BooleanMatcher("flag")
        for val in ["yes", "true", "True", "Y", "1"]:
            score, _ = m.match("yes", val)
            assert score == 1.0

    def test_false_values(self):
        m = BooleanMatcher("flag")
        for val in ["no", "false", "False", "N", "0"]:
            score, _ = m.match("no", val)
            assert score == 1.0

    def test_mismatch(self):
        m = BooleanMatcher("flag")
        score, _ = m.match("yes", "no")
        assert score == 0.0

    def test_invalid(self):
        m = BooleanMatcher("flag")
        score, _ = m.match("yes", "maybe")
        assert score == 0.0


class TestEnumMatcher:
    """Test enum matching."""

    def test_preset_yes_no(self):
        m = EnumMatcher("field", preset="yes_no")
        score, _ = m.match("yes", "Yes")
        assert score == 1.0

        score, _ = m.match("no", "NO")
        assert score == 1.0

    def test_preset_yes_no_na(self):
        m = EnumMatcher("field", preset="yes_no_na")
        score, _ = m.match("n/a", "N/A")
        assert score == 1.0

    def test_custom_values(self):
        m = EnumMatcher("org_type", valid_values=["LLC", "Corporation", "Individual"])
        score, _ = m.match("LLC", "llc")
        assert score == 1.0

    def test_case_variations(self):
        m = EnumMatcher("lease_type", valid_values=["Gross", "Net", "Triple Net"])
        score, _ = m.match("Gross", "GROSS")
        assert score == 1.0

        score, _ = m.match("triple net", "Triple Net")
        assert score == 1.0

    def test_typo_detection(self):
        m = EnumMatcher("field", valid_values=["Office", "Retail", "Industrial"])
        score, feedback = m.match("Office", "Ofice")
        assert 0.7 <= score <= 0.95
        assert "office" in feedback.lower()

    def test_typo_threshold(self):
        m = EnumMatcher("field", valid_values=["Office"], fuzzy_threshold=0.90)
        score, _ = m.match("Office", "Ofice")
        assert score > 0.0

        score, _ = m.match("Office", "Retail")
        assert score == 0.0

    def test_invalid_value(self):
        m = EnumMatcher("field", valid_values=["A", "B", "C"])
        score, feedback = m.match("A", "Z")
        assert score == 0.0
        assert "invalid" in feedback.lower()

    def test_both_valid_but_different(self):
        m = EnumMatcher("field", valid_values=["Gross", "Net"])
        score, feedback = m.match("Gross", "Net")
        assert score == 0.0
        assert "gross" in feedback.lower() and "net" in feedback.lower()

    def test_null_handling(self):
        m = EnumMatcher("field", valid_values=["A", "B"])

        score, _ = m.match("N/A", "na")
        assert score == 1.0

        score, _ = m.match("Unknown", "TBD")
        assert score == 1.0

        score, _ = m.match("A", "N/A")
        assert score == 0.0

    def test_disable_null_handling(self):
        m = EnumMatcher("field", valid_values=["N/A", "Unknown", "A"], treat_null_as_none=False)
        score, _ = m.match("N/A", "n/a")
        assert score == 1.0

    def test_case_sensitive_mode(self):
        m = EnumMatcher("field", valid_values=["LLC", "Inc"], case_sensitive=True)

        score, _ = m.match("LLC", "LLC")
        assert score == 1.0

        score, _ = m.match("LLC", "llc")
        assert score == 0.0

    def test_messy_yes_no(self):
        m = EnumMatcher("field", preset="yes_no_na")

        for variant in ["Yes", "YES", "yes"]:
            score, _ = m.match("yes", variant)
            assert score >= 0.95

        score, _ = m.match("yes", "Y")
        assert score == 0.0

        for variant in ["N/A", "na", "n/a", "NA"]:
            score, _ = m.match("n/a", variant)
            assert score >= 0.95

    def test_lease_types(self):
        """Test realistic lease type enums."""
        m = EnumMatcher("lease_type", valid_values=["Gross", "Net", "Modified Gross", "Triple Net"])

        score, _ = m.match("Triple Net", "triple net")
        assert score == 1.0

        score, _ = m.match("Modified Gross", "modified gross")
        assert score == 1.0

        # Typo
        score, _ = m.match("Gross", "Gros")
        assert 0.7 <= score <= 0.95

    def test_org_types(self):
        """Test organization type enums."""
        m = EnumMatcher(
            "org_type",
            valid_values=[
                "LLC",
                "Corporation",
                "Individual",
                "Partnership",
                "LLP",
                "Government",
                "Charitable Org",
                "Inc.",
                "Other",
            ],
        )

        score, _ = m.match("LLC", "llc")
        assert score == 1.0

        score, _ = m.match("Charitable Org", "charitable org")
        assert score == 1.0

        # Invalid
        score, _ = m.match("LLC", "Unknown Entity Type")
        assert score == 0.0

    def test_empty_string_as_null(self):
        """Test empty string handling."""
        m = EnumMatcher("field", valid_values=["A", "B"])
        score, _ = m.match("", "")
        assert score == 1.0

    def test_zero_as_null(self):
        """Test '0' as null."""
        m = EnumMatcher("field", valid_values=["1", "2", "3"])
        score, _ = m.match("0", "0")
        assert score == 1.0

    def test_whitespace_handling(self):
        """Test whitespace normalization."""
        m = EnumMatcher("field", valid_values=["Option A", "Option B"])
        score, _ = m.match("Option A", "  option a  ")
        assert score == 1.0

    def test_multiple_word_enum(self):
        """Test multi-word enum values."""
        m = EnumMatcher("field", valid_values=["Net Lease", "Gross Lease", "Triple Net Lease"])
        score, _ = m.match("Triple Net Lease", "triple net lease")
        assert score == 1.0

        # Typo in multi-word
        score, _ = m.match("Triple Net Lease", "triple net lase")
        assert 0.7 <= score <= 0.95


class TestAddressMatcher:
    """Test address matching."""

    def test_exact_match(self):
        m = AddressMatcher("addr")
        score, _ = m.match("123 Main St", "123 Main Street")
        assert score == 1.0

    def test_abbreviation_normalization(self):
        m = AddressMatcher("addr")
        score, _ = m.match("456 Oak Ave", "456 Oak Avenue")
        assert score == 1.0

    def test_punctuation_handling(self):
        m = AddressMatcher("addr")
        score, _ = m.match("123 Main St.", "123 Main St")
        assert score == 1.0

    def test_high_similarity(self):
        m = AddressMatcher("addr")
        score, _ = m.match("123 Main St", "123 Main Street Apt 1")
        assert score >= 0.7


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


class TestMatcherRegistry:
    """Test matcher registry."""

    def test_create_phone(self):
        m = MatcherRegistry.create("phone", "phone")
        assert isinstance(m, PhoneMatcher)

    def test_create_date(self):
        m = MatcherRegistry.create("date", "date")
        assert isinstance(m, DateMatcher)

    def test_create_number(self):
        m = MatcherRegistry.create("num", "number")
        assert isinstance(m, NumberMatcher)

    def test_create_string(self):
        m = MatcherRegistry.create("name", "string")
        assert isinstance(m, StringMatcher)

    def test_create_enum(self):
        m = MatcherRegistry.create("type", "enum", valid_values=["A", "B"])
        assert isinstance(m, EnumMatcher)

    def test_unknown_type_defaults_to_string(self):
        m = MatcherRegistry.create("field", "unknown_type")
        assert isinstance(m, StringMatcher)

    def test_register_custom(self):
        class CustomMatcher(StringMatcher):
            pass

        MatcherRegistry.register("custom", CustomMatcher)
        m = MatcherRegistry.create("field", "custom")
        assert isinstance(m, CustomMatcher)


class TestDSPyInterface:
    """Test DSPy metric interface."""

    def test_both_none(self):
        m = PhoneMatcher("phone")
        result = m({"phone": None}, {"phone": None})
        assert result.score == 1.0

    def test_hallucination(self):
        m = PhoneMatcher("phone")
        result = m({"phone": None}, {"phone": "555-1234"})
        assert result.score == 0.0

    def test_missing(self):
        m = PhoneMatcher("phone")
        result = m({"phone": "555-1234"}, {"phone": None})
        assert result.score == 0.0

    def test_normal_match(self):
        m = PhoneMatcher("phone")
        result = m({"phone": "555-1234"}, {"phone": "555-1234"})
        assert result.score == 1.0

    def test_dict_extraction(self):
        m = StringMatcher("name")
        result = m({"name": "Alice"}, {"name": "alice"})
        assert result.score == 1.0

    def test_object_extraction(self):
        m = StringMatcher("name")

        class Example:
            name = "Alice"

        class Pred:
            name = "alice"

        result = m(Example(), Pred())
        assert result.score == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
