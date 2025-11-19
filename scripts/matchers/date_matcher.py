"""Date matcher with fuzzy parsing support."""

import re
from datetime import datetime

from matchers.base_matcher import BaseMatcher


class DateMatcher(BaseMatcher):
    """Match dates with flexible format support and null handling."""

    FORMATS = [
        "%Y-%m-%d %H:%M:%S",  # ISO with time
        "%Y-%m-%d",           # ISO
        "%m/%d/%Y",           # US format
        "%d/%m/%Y",           # European format
        "%B %d, %Y",          # "January 15, 2023"
        "%b %d, %Y",          # "Jan 15, 2023"
        "%Y/%m/%d",           # Alternative
        "%m-%d-%Y",           # Dashed
        "%d %B %Y",           # "15 January 2023"
        "%d %b %Y",           # "15 Jan 2023"
        "%B %d %Y",           # "January 15 2023"
        "%b %d %Y",           # "Jan 15 2023"
    ]

    MONTH_MAP = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
    }

    def _parse(self, date_str: str) -> datetime | None:
        """
        Parse date from various formats.

        Supports:
        - ISO: 2023-01-15, 2023-01-15 10:30:00
        - US: 01/15/2023, 1-15-2023
        - European: 15/01/2023
        - Text: January 15, 2023 / Jan 15, 2023
        - Null values: None, "None", "null", "", "n/a"
        """
        if self._is_null(date_str):
            return None

        s = str(date_str).strip()

        # Try strict formats first
        for fmt in self.FORMATS:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue

        # Try fuzzy regex-based parsing
        # Pattern: "Month Day, Year" or "Month Day Year"
        match = re.search(
            r'(january|february|march|april|may|june|july|august|september|october|november|december|'
            r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2}),?\s+(\d{4})',
            s, re.IGNORECASE
        )
        if match:
            month_str, day, year = match.groups()
            try:
                month = self.MONTH_MAP.get(month_str.lower(), 0)
                if 1 <= month <= 12 and 1 <= int(day) <= 31 and 1900 <= int(year) <= 2100:
                    return datetime(int(year), month, int(day))
            except (ValueError, TypeError):
                pass

        # Pattern: "Day Month Year" or "Day Month, Year"
        match = re.search(
            r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december|'
            r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec),?\s+(\d{4})',
            s, re.IGNORECASE
        )
        if match:
            day, month_str, year = match.groups()
            try:
                month = self.MONTH_MAP.get(month_str.lower(), 0)
                if 1 <= month <= 12 and 1 <= int(day) <= 31 and 1900 <= int(year) <= 2100:
                    return datetime(int(year), month, int(day))
            except (ValueError, TypeError):
                pass

        return None

    def match(self, gold: str, pred: str) -> tuple[float, str]:
        # Null handling first
        gold_null = self._is_null(gold)
        pred_null = self._is_null(pred)

        if gold_null and pred_null:
            return 1.0, f"✓ {self.field_name}: Both null"
        if gold_null and not pred_null:
            return 0.0, f"✗ {self.field_name}: Hallucination (predicted {pred} when gold is null)"
        if not gold_null and pred_null:
            return 0.0, f"✗ {self.field_name}: Missing (expected {gold}, got null)"

        # Parse dates
        g_date = self._parse(gold)
        p_date = self._parse(pred)

        if g_date is None or p_date is None:
            return 0.0, f"✗ {self.field_name}: Parse error | {gold} → {pred}"

        if g_date == p_date:
            return 1.0, f"✓ {self.field_name}: {gold} → {pred}"

        days_diff = abs((g_date - p_date).days)
        if days_diff <= 1:
            return 0.9, f"~ {self.field_name}: Off by {days_diff} day | {gold} → {pred}"

        return 0.0, f"✗ {self.field_name}: Off by {days_diff} days | {gold} → {pred}"
