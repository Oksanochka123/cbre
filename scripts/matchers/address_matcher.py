"""Address matcher with component-based matching."""

import re
from difflib import SequenceMatcher

from matchers.base_matcher import BaseMatcher


class AddressMatcher(BaseMatcher):
    """Match addresses with component parsing and fuzzy matching."""

    def __init__(self, field_name: str, threshold: float = 0.6):
        super().__init__(field_name)
        self.threshold = threshold

    def _normalize(self, addr: str) -> str:
        """Normalize address for comparison."""
        s = str(addr).strip().lower()
        # Expand abbreviations
        s = re.sub(r"\bst\b", "street", s)
        s = re.sub(r"\bave\b", "avenue", s)
        s = re.sub(r"\brd\b", "road", s)
        s = re.sub(r"\bdr\b", "drive", s)
        s = re.sub(r"\bblvd\b", "boulevard", s)
        s = re.sub(r"\bln\b", "lane", s)
        s = re.sub(r"\bpk(wy)?\b", "parkway", s)
        s = re.sub(r"\bct\b", "court", s)
        s = re.sub(r"\bterr\b", "terrace", s)
        # Remove punctuation
        s = re.sub(r"[.,#'\"()-]", " ", s)
        return " ".join(s.split())

    def _strip_company_info(self, addr: str) -> str:
        """Remove company names, c/o, Attn, etc."""
        s = str(addr).strip()
        # Remove "c/o Company Name" patterns
        s = re.sub(r'\bc/o\b[^,]*(?:,|$)', ' ', s, flags=re.IGNORECASE)
        # Remove "Attn: Name" patterns
        s = re.sub(r'Attn:.*', '', s, flags=re.IGNORECASE)
        # Remove common company suffixes
        s = re.sub(r'\b(LLC|INC|Inc|Inc\.|Company|Corp|Corporation|Ltd|Limited)\b', '', s, flags=re.IGNORECASE)
        return " ".join(s.split())

    def _extract_zip(self, addr: str) -> str | None:
        """Extract ZIP code from address."""
        match = re.search(r'\b\d{5}\b', addr)
        return match.group(0) if match else None

    def _extract_state(self, addr: str) -> str | None:
        """Extract state abbreviation from address."""
        match = re.search(r'\b([A-Z]{2})\b', addr)
        return match.group(1) if match else None

    def match(self, gold: str, pred: str) -> tuple[float, str]:
        # Null handling first
        gold_null = self._is_null(gold)
        pred_null = self._is_null(pred)

        if gold_null and pred_null:
            return 1.0, f"✓ {self.field_name}: Both null"
        if gold_null and not pred_null:
            return 0.0, f"✗ {self.field_name}: Hallucination (predicted {pred[:50]} when gold is null)"
        if not gold_null and pred_null:
            return 0.0, f"✗ {self.field_name}: Missing (expected {gold[:50]}, got null)"

        # Normalize both addresses
        g_norm = self._normalize(gold)
        p_norm = self._normalize(pred)

        # Try exact match
        if g_norm == p_norm:
            return 1.0, f"✓ {self.field_name}: Exact match"

        # Extract components
        g_zip = self._extract_zip(gold)
        p_zip = self._extract_zip(pred)
        g_state = self._extract_state(gold)
        p_state = self._extract_state(pred)

        # Strip company info from prediction (it often has extra detail)
        p_stripped = self._strip_company_info(pred)
        p_stripped_norm = self._normalize(p_stripped)

        # Check if core address is in prediction (prediction is more detailed)
        if g_norm in p_stripped_norm or g_norm in p_norm:
            # Gold address appears as substring in prediction
            return 0.8, f"✓ {self.field_name}: Gold embedded in prediction"

        # Component-based matching
        score = 0.0

        # Match ZIP code
        if g_zip and p_zip and g_zip == p_zip:
            score += 0.3
        # Match state
        if g_state and p_state and g_state == p_state:
            score += 0.2

        # String similarity on full normalized addresses
        sim = SequenceMatcher(None, g_norm, p_norm).ratio()
        score += sim * 0.5

        if score >= self.threshold:
            return score, f"✗ {self.field_name}: {score:.2f} sim | '{gold[:40]}' vs '{pred[:40]}'"

        return 0.0, f"✗ {self.field_name}: {score:.2f} sim (too low) | '{gold[:40]}' vs '{pred[:40]}'"
