"""Unit tests for BooleanMatcher."""

from scripts.matchers import BooleanMatcher


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
