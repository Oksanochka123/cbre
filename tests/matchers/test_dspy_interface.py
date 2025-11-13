"""Unit tests for DSPy metric interface."""

from scripts.matchers import PhoneMatcher, StringMatcher


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
