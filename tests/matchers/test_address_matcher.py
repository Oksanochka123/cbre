"""Unit tests for AddressMatcher."""

from scripts.matchers import AddressMatcher


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
