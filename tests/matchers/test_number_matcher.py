"""Unit tests for NumberMatcher."""

from scripts.matchers import NumberMatcher


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

    # Currency format tests
    def test_currency_symbol_usd(self):
        m = NumberMatcher("num")
        score, _ = m.match("$1000", "1000")
        assert score == 1.0

        score, _ = m.match("1000", "$1000")
        assert score == 1.0

    def test_currency_code_usd(self):
        m = NumberMatcher("num")
        score, _ = m.match("USD 1000", "1000")
        assert score == 1.0

        score, _ = m.match("USD1000", "1000")
        assert score == 1.0

    def test_currency_with_millions(self):
        m = NumberMatcher("num")
        score, _ = m.match("$1M", "1000000")
        assert score == 1.0

        score, _ = m.match("USD 1M", "1000000")
        assert score == 1.0

        score, _ = m.match("1M", "1000000")
        assert score == 1.0

    def test_currency_with_thousands(self):
        m = NumberMatcher("num")
        score, _ = m.match("$500K", "500000")
        assert score == 1.0

        score, _ = m.match("USD 500K", "500000")
        assert score == 1.0

    def test_currency_with_billions(self):
        m = NumberMatcher("num")
        score, _ = m.match("$1.5B", "1500000000")
        assert score == 1.0

        score, _ = m.match("USD 2B", "2000000000")
        assert score == 1.0

    def test_currency_with_decimals(self):
        m = NumberMatcher("num")
        score, _ = m.match("$1.5M", "1500000")
        assert score == 1.0

        score, _ = m.match("USD 2.5M", "2500000")
        assert score == 1.0

    def test_other_currencies(self):
        m = NumberMatcher("num")
        score, _ = m.match("EUR 1000", "1000")
        assert score == 1.0

        score, _ = m.match("GBP 500K", "500000")
        assert score == 1.0

        score, _ = m.match("€1M", "1000000")
        assert score == 1.0

        score, _ = m.match("£500K", "500000")
        assert score == 1.0
