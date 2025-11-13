"""Unit tests for JSONMatcher."""

from scripts.matchers import JSONMatcher


class TestJSONMatcher:
    """Test JSON list matching."""

    def test_empty_lists(self):
        m = JSONMatcher("items")
        score, _ = m.match("[]", "[]")
        assert score == 1.0

    def test_single_record_match(self):
        m = JSONMatcher("items")
        gold = '[{"name": "A", "value": 1}]'
        pred = '[{"name": "A", "value": 1}]'
        score, _ = m.match(gold, pred)
        assert score >= 0.95

    def test_record_count_mismatch(self):
        m = JSONMatcher("items")
        gold = '[{"name": "A"}, {"name": "B"}]'
        pred = '[{"name": "A"}]'
        score, _ = m.match(gold, pred)
        assert 0.0 < score < 0.7

    def test_field_mismatch(self):
        m = JSONMatcher("items")
        gold = '[{"name": "A", "value": 1}]'
        pred = '[{"name": "A", "value": 2}]'
        score, _ = m.match(gold, pred)
        assert 0.4 < score < 0.9
