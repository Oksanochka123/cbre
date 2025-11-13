"""Unit tests for DateMatcher."""

from scripts.matchers import DateMatcher


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
