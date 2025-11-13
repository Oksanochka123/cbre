"""Unit tests for EnumMatcher."""

from scripts.matchers import EnumMatcher


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
