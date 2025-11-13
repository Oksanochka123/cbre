"""Unit tests for MatcherRegistry."""

from scripts.matchers import (
    DateMatcher,
    EnumMatcher,
    MatcherRegistry,
    NumberMatcher,
    PhoneMatcher,
    StringMatcher,
)


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
