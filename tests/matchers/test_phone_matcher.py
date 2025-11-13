"""Unit tests for PhoneMatcher."""

from scripts.matchers import PhoneMatcher


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
