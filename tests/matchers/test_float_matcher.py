"""Unit tests for FloatMatcher."""

from scripts.matchers import FloatMatcher


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
