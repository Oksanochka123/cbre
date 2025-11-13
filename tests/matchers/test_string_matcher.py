"""Unit tests for StringMatcher."""

from scripts.matchers import StringMatcher


class TestStringMatcher:
    """Test string matching."""

    def test_exact_match(self):
        m = StringMatcher("name")
        score, _ = m.match("John Doe", "john doe")
        assert score == 1.0

    def test_high_similarity(self):
        m = StringMatcher("name")
        score, _ = m.match("John Doe", "John Do")
        assert score >= 0.6  # Above threshold, returns similarity ratio

    def test_low_similarity(self):
        m = StringMatcher("name")
        score, _ = m.match("John Doe", "Jane Smith")
        assert score == 0.0  # Below threshold

    def test_custom_threshold(self):
        m = StringMatcher("name", threshold=0.8)
        # High similarity above threshold
        score, _ = m.match("John Doe", "John Do")  # ~0.88 similarity
        assert score >= 0.8

        # Lower similarity below threshold
        score, _ = m.match("John", "Jane")  # ~0.5 similarity
        assert score == 0.0
